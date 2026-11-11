"""TRC-v5: calibrated selective cross-branch evidence correction.

The deployed action is still factorized into permission, branch direction and
strength.  Unlike v4d, the permission score is not trained with a dataset
prior.  Explicit response competition statistics supplement the learned map
encoder so that the decision heads can distinguish C/T peak structure before
choosing a hard action.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from lib.models.heads.target_reference_erc import CHANNELS, build_evidence


RESPONSE_STAT_DIM = 15


class CalibratedSelectiveTargetReferenceERCPrototype(nn.Module):
    """Hard deployed policy with straight-through training surrogates."""

    def __init__(self, hidden=32, alpha_max=0.5, temperature=1.0,
                 permission_threshold=0.5,
                 permission_initial_probability=0.5,
                 direction_t_threshold=0.5):
        super().__init__()
        if hidden <= 0 or alpha_max <= 0 or temperature <= 0:
            raise ValueError('Positive hidden/alpha/temperature required')
        if not 0 < permission_threshold < 1:
            raise ValueError('permission_threshold must be in (0, 1)')
        if not 0 < permission_initial_probability < 1:
            raise ValueError('permission_initial_probability must be in (0, 1)')
        if not 0 < direction_t_threshold < 1:
            raise ValueError('direction_t_threshold must be in (0, 1)')
        self.alpha_max = float(alpha_max)
        self.temperature = float(temperature)
        self.permission_threshold = float(permission_threshold)
        self.direction_t_threshold = float(direction_t_threshold)
        self.encoder = nn.Sequential(
            nn.Conv2d(len(CHANNELS), hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        decision_dim = 3 * hidden + RESPONSE_STAT_DIM
        self.permission_head = nn.Sequential(
            nn.Linear(decision_dim, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(decision_dim, hidden), nn.GELU(), nn.Linear(hidden, 2),
        )
        self.strength_head = nn.Sequential(
            nn.Linear(decision_dim, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )
        for head in (self.permission_head, self.direction_head, self.strength_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        initial_logit = self.temperature * math.log(
            permission_initial_probability / (1.0 - permission_initial_probability)
        )
        nn.init.constant_(self.permission_head[-1].bias, initial_logit)

    @property
    def reachable_alpha_supremum(self):
        return self.alpha_max

    @staticmethod
    def _learned_statistics(features):
        return torch.cat([
            features.mean((-2, -1)),
            features.amax((-2, -1)),
            features.flatten(2).std(-1, unbiased=False),
        ], -1)

    @staticmethod
    def _response_statistics(evidence):
        """Dataset-independent C/T competition features in stable ranges."""
        log_c = evidence['log_c'].detach().flatten(1)
        log_t = evidence['log_t'].detach().flatten(1)
        d = evidence['d'].detach().flatten(1)
        hann = evidence['hann'].detach().flatten(1)
        batch, count = log_c.shape
        height, width = evidence['c'].shape[-2:]
        pc, pt = log_c.softmax(-1), log_t.softmax(-1)
        eps = torch.finfo(pc.dtype).eps
        entropy_scale = max(math.log(max(count, 2)), eps)
        entropy_c = -(pc * pc.clamp_min(eps).log()).sum(-1) / entropy_scale
        entropy_t = -(pt * pt.clamp_min(eps).log()).sum(-1) / entropy_scale

        top_c, index_c = pc.max(-1)
        top_t, index_t = pt.max(-1)
        if count > 1:
            top2_c = pc.topk(2, dim=-1).values[:, 1]
            top2_t = pt.topk(2, dim=-1).values[:, 1]
        else:
            top2_c, top2_t = top_c, top_t
        concentration_scale = math.log1p(max(count, 2))
        concentration_c = torch.log1p(top_c * count) / concentration_scale
        concentration_t = torch.log1p(top_t * count) / concentration_scale
        margin_c = (top_c - top2_c) / top_c.clamp_min(eps)
        margin_t = (top_t - top2_t) / top_t.clamp_min(eps)
        cross_c_at_t = pc.gather(1, index_t[:, None])[:, 0] / top_c.clamp_min(eps)
        cross_t_at_c = pt.gather(1, index_c[:, None])[:, 0] / top_t.clamp_min(eps)

        x_c = (index_c % width).to(pc) / max(width - 1, 1)
        y_c = torch.div(index_c, width, rounding_mode='floor').to(pc) / max(height - 1, 1)
        x_t = (index_t % width).to(pc) / max(width - 1, 1)
        y_t = torch.div(index_t, width, rounding_mode='floor').to(pc) / max(height - 1, 1)
        dx, dy = (x_c - x_t).abs(), (y_c - y_t).abs()
        distance = (dx.square() + dy.square()).sqrt() / math.sqrt(2.0)

        total_variation = 0.5 * (pc - pt).abs().sum(-1)
        mixture = 0.5 * (pc + pt)
        js = 0.5 * (
            (pc * (pc.clamp_min(eps) / mixture.clamp_min(eps)).log()).sum(-1)
            + (pt * (pt.clamp_min(eps) / mixture.clamp_min(eps)).log()).sum(-1)
        ) / math.log(2.0)

        log_base = log_c + log_t + hann.clamp_min(1e-30).log()
        if count > 1:
            base_top2 = log_base.topk(2, dim=-1).values
            base_margin = torch.tanh((base_top2[:, 0] - base_top2[:, 1]).clamp_min(0) / 4.0)
        else:
            base_margin = log_base.new_zeros(batch)
        base_index = log_base.argmax(-1)
        d_scale = d.abs().amax(-1).clamp_min(1.0)
        d_at_base = d.gather(1, base_index[:, None])[:, 0] / d_scale

        statistics = torch.stack([
            entropy_c, entropy_t,
            concentration_c, concentration_t,
            margin_c, margin_t,
            cross_c_at_t, cross_t_at_c,
            dx, dy, distance,
            total_variation, js,
            base_margin, d_at_base,
        ], -1)
        if statistics.shape[-1] != RESPONSE_STAT_DIM:
            raise RuntimeError('Unexpected response statistic dimension')
        return statistics.nan_to_num().clamp(-1.0, 1.0)

    def forward(self, evidence, mode='FULL_ERC'):
        if mode not in ('FULL_ERC', 'ERC_OFF'):
            raise ValueError('Unsupported mode')
        features = self.encoder(evidence['channels'].detach())
        response_statistics = self._response_statistics(evidence)
        decision_features = torch.cat([
            self._learned_statistics(features), response_statistics,
        ], -1)

        permission_logit = self.permission_head(decision_features)[:, 0]
        permission_probability = torch.sigmoid(permission_logit / self.temperature)
        hard_permission = permission_probability.gt(
            self.permission_threshold
        ).to(permission_probability)
        permission = (
            hard_permission + permission_probability - permission_probability.detach()
            if self.training else hard_permission
        )

        direction_logits = self.direction_head(decision_features)
        direction_probability = (direction_logits / self.temperature).softmax(-1)
        # With two classes, p(T) > 0.5 is exactly the former argmax policy
        # (including the C decision on ties).  Keeping the boundary explicit
        # lets later designs calibrate deployment without changing the learned
        # probability objective or any earlier version's default behaviour.
        hard_direction_index = direction_probability[:, 1].gt(
            self.direction_t_threshold
        ).long()
        hard_direction_policy = F.one_hot(
            hard_direction_index, 2
        ).to(direction_probability)
        direction_policy = (
            hard_direction_policy + direction_probability - direction_probability.detach()
            if self.training else hard_direction_policy
        )
        direction = direction_policy[:, 0] - direction_policy[:, 1]
        soft_direction = direction_probability[:, 0] - direction_probability[:, 1]
        strength_logit = self.strength_head(decision_features)[:, 0]
        strength = strength_logit.sigmoid()

        raw_alpha_scalar = self.alpha_max * permission * direction * strength
        raw_soft_alpha_scalar = (
            self.alpha_max * permission_probability * soft_direction * strength
        )
        active = evidence['active']
        alpha_scalar = torch.where(
            active, raw_alpha_scalar, torch.zeros_like(raw_alpha_scalar)
        )
        soft_alpha_scalar = torch.where(
            active, raw_soft_alpha_scalar, torch.zeros_like(raw_soft_alpha_scalar)
        )
        alpha = alpha_scalar[:, None, None].expand_as(evidence['d'])
        soft_alpha = soft_alpha_scalar[:, None, None].expand_as(evidence['d'])
        correction = alpha * evidence['d']
        factor = correction.exp()
        if mode == 'ERC_OFF':
            alpha = torch.zeros_like(alpha)
            correction = torch.zeros_like(correction)
            factor = torch.ones_like(factor)

        hard_action = torch.where(
            hard_permission.bool(), hard_direction_index + 1,
            torch.zeros_like(hard_direction_index),
        )
        action_probability = torch.stack([
            1.0 - permission_probability,
            permission_probability * direction_probability[:, 0],
            permission_probability * direction_probability[:, 1],
        ], -1)
        c, t, hann = evidence['c'], evidence['t'], evidence['hann']
        base = (c * hann) * t
        log_base = (
            evidence['log_c'] + evidence['log_t']
            + hann.clamp_min(1e-30).log()
        )
        return {
            'features': features,
            'response_statistics': response_statistics,
            'decision_features': decision_features,
            'permission_logit': permission_logit,
            'permission_probability': permission_probability,
            'permission_threshold': self.permission_threshold,
            'permission_temperature': self.temperature,
            'hard_permission': hard_permission,
            'direction_logits': direction_logits,
            'direction_probability': direction_probability,
            'direction_t_threshold': self.direction_t_threshold,
            'hard_direction_index': hard_direction_index,
            'strength_logit': strength_logit,
            'strength': strength,
            'action_probability': action_probability,
            'hard_action': hard_action,
            'alpha_scalar': alpha_scalar,
            'soft_alpha_scalar': soft_alpha_scalar,
            'alpha': alpha,
            'soft_alpha': soft_alpha,
            'factor': factor,
            'base': base,
            'log_base': log_base,
            'score': base * factor,
            'log_score': log_base + correction,
            'support': hann > 0,
            'active': active,
        }


class TargetReferenceERCV5(CalibratedSelectiveTargetReferenceERCPrototype):
    DESIGN = 'target_reference_erc_trc_v5_calibrated_selective'

    def __init__(self, feat_sz, hidden=32, alpha_max=0.5, temperature=1.0,
                 d_max=4.0, inference_mode='FULL_ERC', force_off=False):
        super().__init__(hidden, alpha_max, temperature)
        self.design = self.DESIGN
        self.feat_sz = int(feat_sz)
        self.d_max = float(d_max)
        self.force_off = bool(force_off)
        self.set_inference_mode(inference_mode)
        window = torch.hann_window(feat_sz, periodic=False)
        self.register_buffer(
            'hann_window', torch.outer(window, window).reshape(1, -1)
        )

    def set_inference_mode(self, mode):
        mode = {'OFF': 'ERC_OFF', 'FULL': 'FULL_ERC'}.get(
            str(mode).upper(), str(mode).upper())
        if mode not in ('ERC_OFF', 'FULL_ERC'):
            raise ValueError('TRC-v5 supports only ERC_OFF/FULL_ERC')
        self.inference_mode = mode

    def forward(self, c, native_t, *, search, prompts, template, template_mask,
                logit_scale, flag):
        evidence = build_evidence(
            c, search, prompts, template, template_mask, logit_scale, flag,
            self.hann_window.reshape(self.feat_sz, self.feat_sz),
            d_max=self.d_max,
        )
        mode = 'ERC_OFF' if self.force_off else self.inference_mode
        packet = super().forward(evidence, mode)
        packet.update({'evidence': evidence, 'alpha_limit': self.alpha_max})
        batch = c.shape[0]
        c_flat, native_t_flat = c.flatten(1), native_t.flatten(1)
        native_base = c_flat * native_t_flat
        native_windowed = native_base * self.hann_window
        probability = packet['action_probability']
        return {
            'features': evidence['channels'],
            'cls_response': c_flat,
            'cont_response': native_t_flat,
            'windowed_cls_score': c_flat * self.hann_window,
            'windowed_cont_score': native_t_flat * self.hann_window,
            'base_score': native_base,
            'windowed_base_score': native_windowed,
            'calibrated_score': (
                evidence['c'] * evidence['t'] * packet['factor']
            ).flatten(1),
            'windowed_calibrated_score': packet['score'].flatten(1),
            'factor': packet['factor'].flatten(1),
            'log_ratio': evidence['d'].flatten(1),
            'alpha': packet['alpha'].flatten(1),
            'raw_alpha': packet['alpha_scalar'][:, None],
            'z_alpha': (
                packet['direction_logits'][:, 0]
                - packet['direction_logits'][:, 1]
            )[:, None],
            'reliability': packet['permission_probability'][:, None],
            'reliability_cls': probability[:, 1:2],
            'reliability_cont': probability[:, 2:3],
            'reliability_base': probability[:, :1],
            'reliability_gate': packet['permission_probability'][:, None],
            'intervention_gate': packet['permission_probability'][:, None],
            'gate_logit': packet['permission_logit'][:, None],
            'delta': packet['strength_logit'][:, None],
            'active_mask': evidence['active'],
            'trc_output': packet,
        }

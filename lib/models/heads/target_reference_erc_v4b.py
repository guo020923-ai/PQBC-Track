"""TRC-v4b: factorized permission, branch direction and strength.

Corrective actions are rare.  A single three-class head therefore learned the
dominant KEEP prior instead of the conditional evidence pattern.  This module
factorizes the problem into: whether to intervene, which branch to favour,
and the smallest useful strength.  All three predictions share the same
target-reference response encoder and are trained in one joint run.
"""

import torch
from torch import nn
from torch.nn import functional as F

from lib.models.heads.target_reference_erc import CHANNELS, build_evidence


class FactorizedTargetReferenceERCPrototype(nn.Module):
    """Hard permission and direction with straight-through training."""

    def __init__(self, hidden=32, alpha_max=0.5, temperature=1.0):
        super().__init__()
        if hidden <= 0 or alpha_max <= 0 or temperature <= 0:
            raise ValueError('Positive hidden/alpha/temperature required')
        self.alpha_max = float(alpha_max)
        self.temperature = float(temperature)
        self.encoder = nn.Sequential(
            nn.Conv2d(len(CHANNELS), hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        decision_dim = 3 * hidden
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

    @property
    def reachable_alpha_supremum(self):
        return self.alpha_max

    @staticmethod
    def _decision_features(features):
        return torch.cat([
            features.mean((-2, -1)),
            features.amax((-2, -1)),
            features.flatten(2).std(-1, unbiased=False),
        ], -1)

    def forward(self, evidence, mode='FULL_ERC'):
        if mode not in ('FULL_ERC', 'ERC_OFF'):
            raise ValueError('Unsupported mode')
        features = self.encoder(evidence['channels'].detach())
        decision_features = self._decision_features(features)
        permission_logit = self.permission_head(decision_features)[:, 0]
        permission_probability = torch.sigmoid(permission_logit / self.temperature)
        hard_permission = permission_probability.gt(0.5).to(permission_probability)
        permission = (
            hard_permission + permission_probability - permission_probability.detach()
            if self.training else hard_permission
        )
        direction_logits = self.direction_head(decision_features)
        direction_probability = (direction_logits / self.temperature).softmax(-1)
        hard_direction_index = direction_probability.argmax(-1)
        hard_direction_policy = F.one_hot(hard_direction_index, 2).to(direction_probability)
        direction_policy = (
            hard_direction_policy + direction_probability - direction_probability.detach()
            if self.training else hard_direction_policy
        )
        direction = direction_policy[:, 0] - direction_policy[:, 1]
        soft_direction = direction_probability[:, 0] - direction_probability[:, 1]
        strength_logit = self.strength_head(decision_features)[:, 0]
        strength = strength_logit.sigmoid()

        alpha_scalar = self.alpha_max * permission * direction * strength
        soft_alpha_scalar = (
            self.alpha_max * permission_probability * soft_direction * strength
        )
        active = evidence['active']
        alpha_scalar = torch.where(active, alpha_scalar, torch.zeros_like(alpha_scalar))
        soft_alpha_scalar = torch.where(
            active, soft_alpha_scalar, torch.zeros_like(soft_alpha_scalar)
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
        log_base = evidence['log_c'] + evidence['log_t']
        log_base = log_base + hann.clamp_min(1e-30).log()
        return {
            'features': features,
            'decision_features': decision_features,
            'permission_logit': permission_logit,
            'permission_probability': permission_probability,
            'hard_permission': hard_permission,
            'direction_logits': direction_logits,
            'direction_probability': direction_probability,
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


class TargetReferenceERCV4B(FactorizedTargetReferenceERCPrototype):
    DESIGN = 'target_reference_erc_trc_v4b_factorized'

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
            raise ValueError('TRC-v4b supports only ERC_OFF/FULL_ERC')
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
        result = {
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
                packet['direction_logits'][:, 0] - packet['direction_logits'][:, 1]
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
        return result

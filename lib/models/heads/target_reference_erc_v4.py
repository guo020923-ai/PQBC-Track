"""TRC-v4: hard-decision, magnitude-calibrated target-reference ERC.

TRC-v3 learned a three-way frame action and an unconstrained spatial gate.
The gate had no identifiable target and saturated to one, while soft oracle
tie-breaking labelled many candidate-preserving actions as interventions.

This version makes the deployment decision explicit: KEEP, favour C, or
favour T.  A separate bounded magnitude predicts how strongly a selected
branch should be favoured.  Training uses a straight-through hard policy;
evaluation uses the same deterministic hard policy.  There is no redundant
spatial gate.  Inference remains response/reference-only and flag=2-only.
"""

import torch
from torch import nn
from torch.nn import functional as F

from lib.models.heads.target_reference_erc import CHANNELS, build_evidence


class HardDecisionTargetReferenceERCPrototype(nn.Module):
    """Deterministic KEEP/C/T policy with an independently learned strength."""

    ACTIONS = ('keep', 'classification', 'contrastive')

    def __init__(self, hidden=32, alpha_max=0.5, action_temperature=1.0):
        super().__init__()
        if hidden <= 0 or alpha_max <= 0 or action_temperature <= 0:
            raise ValueError('Positive hidden/alpha/temperature required')
        self.alpha_max = float(alpha_max)
        self.temperature = float(action_temperature)
        self.encoder = nn.Sequential(
            nn.Conv2d(len(CHANNELS), hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        # Mean, maximum and standard deviation retain response ambiguity that
        # mean pooling alone discarded in v3.
        decision_dim = 3 * hidden
        self.action_head = nn.Sequential(
            nn.Linear(decision_dim, hidden), nn.GELU(), nn.Linear(hidden, 3),
        )
        self.strength_head = nn.Sequential(
            nn.Linear(decision_dim, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )
        # Zero logits deterministically choose KEEP (index zero), giving exact
        # baseline behaviour before any update.  Strength starts at 0.5 but is
        # irrelevant while KEEP is selected.
        nn.init.zeros_(self.action_head[-1].weight)
        nn.init.zeros_(self.action_head[-1].bias)
        nn.init.zeros_(self.strength_head[-1].weight)
        nn.init.zeros_(self.strength_head[-1].bias)

    @property
    def reachable_alpha_supremum(self):
        return self.alpha_max

    @staticmethod
    def _decision_features(features):
        mean = features.mean((-2, -1))
        maximum = features.amax((-2, -1))
        std = features.flatten(2).std(-1, unbiased=False)
        return torch.cat([mean, maximum, std], -1)

    def forward(self, evidence, mode='FULL_ERC'):
        if mode not in ('FULL_ERC', 'ERC_OFF'):
            raise ValueError('Unsupported mode')
        features = self.encoder(evidence['channels'].detach())
        decision_features = self._decision_features(features)
        action_logits = self.action_head(decision_features)
        action_probability = (action_logits / self.temperature).softmax(-1)
        hard_action = action_probability.argmax(-1)
        hard_policy = F.one_hot(hard_action, 3).to(action_probability.dtype)
        # Forward values match deployment exactly while soft probabilities
        # carry gradients to the action classifier during training.
        policy = (
            hard_policy + action_probability - action_probability.detach()
            if self.training else hard_policy
        )
        direction = policy[:, 1] - policy[:, 2]
        soft_direction = action_probability[:, 1] - action_probability[:, 2]
        strength_logit = self.strength_head(decision_features)[:, 0]
        strength = strength_logit.sigmoid()
        alpha_scalar = self.alpha_max * direction * strength
        soft_alpha_scalar = self.alpha_max * soft_direction * strength

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

        c, t, hann = evidence['c'], evidence['t'], evidence['hann']
        base = (c * hann) * t
        log_base = evidence['log_c'] + evidence['log_t']
        log_base = log_base + hann.clamp_min(1e-30).log()
        return {
            'features': features,
            'decision_features': decision_features,
            'action_logits': action_logits,
            'action_probability': action_probability,
            'hard_action': hard_action,
            'hard_policy': hard_policy,
            'strength_logit': strength_logit,
            'strength': strength,
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


class TargetReferenceERCV4(HardDecisionTargetReferenceERCPrototype):
    """Native UVLTrack adapter for hard-decision TRC-v4."""

    DESIGN = 'target_reference_erc_trc_v4_hard_decision'

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
            raise ValueError('TRC-v4 supports only ERC_OFF/FULL_ERC')
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
        packet.update({
            'evidence': evidence,
            'alpha_limit': self.reachable_alpha_supremum,
        })
        batch = c.shape[0]
        c_flat, native_t_flat = c.flatten(1), native_t.flatten(1)
        native_base = c_flat * native_t_flat
        native_windowed = native_base * self.hann_window
        probability = packet['action_probability']
        non_keep_probability = 1.0 - probability[:, :1]
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
                packet['action_logits'][:, 1] - packet['action_logits'][:, 2]
            )[:, None],
            'reliability': non_keep_probability,
            'reliability_cls': probability[:, 1:2],
            'reliability_cont': probability[:, 2:3],
            'reliability_base': probability[:, :1],
            'reliability_gate': non_keep_probability,
            'intervention_gate': non_keep_probability,
            'gate_logit': (
                torch.logsumexp(packet['action_logits'][:, 1:], -1)
                - packet['action_logits'][:, 0]
            )[:, None],
            'delta': packet['strength_logit'][:, None],
            'active_mask': evidence['active'],
            'trc_output': packet,
        }
        return result

"""TRC-v3: decision-selective target-reference ERC.

The estimator makes one coherent frame-level choice among preserving the
native fusion, favouring classification evidence, and favouring contrastive
evidence.  A non-negative spatial gate controls where that choice is applied;
it cannot reverse the frame-level C/T direction.  Inference remains
response-only and is active only for the official flag=2 VL-tracking path.
"""

import torch
from torch import nn

from lib.models.heads.target_reference_erc import CHANNELS, build_evidence


class DecisionSelectiveTargetReferenceERCPrototype(nn.Module):
    """Three-action global policy with a direction-preserving spatial gate."""

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
        self.action_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2 * hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        # Exact identity at initialization: equal C/T action probabilities
        # imply beta=0.  The gate can be 0.5 without changing the native score.
        nn.init.zeros_(self.action_head[-1].weight)
        nn.init.zeros_(self.action_head[-1].bias)
        nn.init.zeros_(self.spatial_gate[-1].weight)
        nn.init.zeros_(self.spatial_gate[-1].bias)

    @property
    def reachable_alpha_supremum(self):
        return self.alpha_max

    def forward(self, evidence, mode='FULL_ERC'):
        if mode not in ('FULL_ERC', 'ERC_OFF'):
            raise ValueError('Unsupported mode')
        features = self.encoder(evidence['channels'].detach())
        pooled = features.mean((-2, -1))
        action_logits = self.action_head(pooled)
        action_probability = (action_logits / self.temperature).softmax(-1)
        beta = action_probability[:, 1] - action_probability[:, 2]

        global_features = pooled[:, :, None, None].expand_as(features)
        gate_logit = self.spatial_gate(
            torch.cat([features, global_features], dim=1)
        )[:, 0]
        gate = gate_logit.sigmoid()
        alpha = self.alpha_max * beta[:, None, None] * gate
        active = evidence['active'][:, None, None]
        alpha = torch.where(active, alpha, torch.zeros_like(alpha))
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
            'action_logits': action_logits,
            'action_probability': action_probability,
            'beta': beta,
            'gate_logit': gate_logit,
            'gate': gate,
            'alpha': alpha,
            'factor': factor,
            'base': base,
            'log_base': log_base,
            'score': base * factor,
            'log_score': log_base + correction,
            'support': hann > 0,
            'active': evidence['active'],
        }


class TargetReferenceERCV3(DecisionSelectiveTargetReferenceERCPrototype):
    """Native UVLTrack Head adapter for decision-selective TRC-v3."""

    DESIGN = 'target_reference_erc_trc_v3_decision_selective'

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
            raise ValueError('TRC-v3 supports only ERC_OFF/FULL_ERC')
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
            'raw_alpha': packet['alpha'].flatten(1),
            'z_alpha': packet['beta'][:, None],
            'reliability': (1.0 - probability[:, :1]),
            'reliability_cls': probability[:, 1:2],
            'reliability_cont': probability[:, 2:3],
            'reliability_base': probability[:, :1],
            'reliability_gate': packet['gate'].flatten(1),
            'intervention_gate': packet['gate'].flatten(1),
            'delta': c.new_zeros(batch, 1),
            'active_mask': evidence['active'],
            'trc_output': packet,
        }
        return result

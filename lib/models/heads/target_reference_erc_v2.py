"""TRC-v2: relative, decision-aligned target-reference ERC.

Inference remains response-only and keeps the TRC-v1 evidence/fusion formula.
The change is deliberately narrow: the two branch reliabilities form one
relative distribution instead of two independently saturating sigmoid scores.
"""

import math

import torch
from torch import nn

from lib.models.heads.target_reference_erc import CHANNELS, build_evidence


class RelativeTargetReferenceERCPrototype(nn.Module):
    """Predict mutually comparable C/T reliabilities at every response cell."""

    def __init__(self, hidden=32, alpha_max=0.5, reliability_temperature=1.0):
        super().__init__()
        if alpha_max <= 0 or reliability_temperature <= 0:
            raise ValueError('Positive coefficient/temperature required')
        self.alpha_max = float(alpha_max)
        self.temperature = float(reliability_temperature)
        self.encoder = nn.Sequential(
            nn.Conv2d(len(CHANNELS), hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.reliability = nn.Conv2d(2 * hidden, 2, 1)
        nn.init.zeros_(self.reliability.weight)
        nn.init.zeros_(self.reliability.bias)

    @property
    def reachable_alpha_supremum(self):
        return self.alpha_max * math.tanh(1.0 / self.temperature)

    def forward(self, evidence, mode='FULL_ERC'):
        if mode not in ('FULL_ERC', 'ERC_OFF'):
            raise ValueError('Unsupported mode')
        features = self.encoder(evidence['channels'].detach())
        global_features = features.mean((-2, -1), keepdim=True).expand_as(features)
        logits = self.reliability(torch.cat([features, global_features], 1))
        # Unlike v1's independent sigmoids, the two values are explicitly
        # comparable and cannot both collapse to one.
        reliabilities = logits.softmax(1)
        preference = reliabilities[:, 0] - reliabilities[:, 1]
        alpha = self.alpha_max * torch.tanh(preference / self.temperature)
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
            'reliability_logits': logits,
            'reliability': reliabilities,
            'preference': preference,
            'alpha': alpha,
            'factor': factor,
            'base': base,
            'log_base': log_base,
            'score': base * factor,
            'log_score': log_base + correction,
            'support': hann > 0,
            'active': evidence['active'],
        }


class TargetReferenceERCV2(RelativeTargetReferenceERCPrototype):
    """Native Head adapter for decision-aligned TRC-v2."""

    DESIGN = 'target_reference_erc_trc_v2_decision_aligned'

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
            raise ValueError('TRC-v2 supports only ERC_OFF/FULL_ERC')
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
        zero = c.new_zeros(batch, 1)
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
            'z_alpha': (
                packet['reliability_logits'][:, 0]
                - packet['reliability_logits'][:, 1]
            ).flatten(1),
            'reliability': packet['reliability'].mean(1).flatten(1),
            'reliability_cls': packet['reliability'][:, 0].flatten(1),
            'reliability_cont': packet['reliability'][:, 1].flatten(1),
            'reliability_base': zero,
            'reliability_gate': zero,
            'intervention_gate': zero,
            'delta': zero,
            'active_mask': evidence['active'],
            'trc_output': packet,
        }
        return result

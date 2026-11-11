"""Decision-aligned peak-quality evidence reliability calibration (PQ-ERC v2).

PQ-v1 learned the quality of the individual C/T peaks, but deployment selects
the maximum of the complete Hann-aware fused response.  PQ-v2 keeps the same
response-only boundary while adding a learned intervention-strength gate and
training the resulting scalar correction against actions that are reachable by
the exact deployed fusion formula.

No bounding-box geometry, candidate IoU or dataset identity is consumed by the
module.  At initialization the two branch qualities are equal, hence alpha is
exactly zero and the official UVLTrack fusion is recovered.
"""

import math

import torch
from torch import nn

from .peak_quality_calibrated_erc_pq1 import PeakQualityCalibratedERCPQ1


class PeakQualityDecisionAlignedERCPQ2(PeakQualityCalibratedERCPQ1):
    """Reliability-directed, decision-aligned scalar C/T calibration."""

    DESIGN = 'peak_quality_decision_aligned_erc_pq2'

    def __init__(
        self, feat_sz, dim, hidden=64, magnitudes=None,
        max_alpha=0.40, d_max=4.0, eps=1e-6,
        action_temperature=1.0, response_temperature=1.0,
        detach_features=True, apply_vl_tracking_only=True,
        inference_mode='FULL_ERC', force_off=False,
    ):
        super().__init__(
            feat_sz=feat_sz, dim=dim, hidden=hidden,
            magnitudes=magnitudes, max_alpha=max_alpha, d_max=d_max,
            eps=eps, action_temperature=action_temperature,
            response_temperature=response_temperature,
            detach_features=detach_features,
            apply_vl_tracking_only=apply_vl_tracking_only,
            inference_mode=inference_mode, force_off=force_off,
        )
        decision_dim = 5 * hidden + 18
        self.gate_head = nn.Sequential(
            nn.Linear(decision_dim + 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.gate_head[-1].weight)
        # A conservative prior does not break exact identity because the
        # reliability gap, and therefore the correction direction, starts at 0.
        self.gate_head[-1].bias.data.fill_(math.log(0.10 / 0.90))
        self.direction_temperature = 0.50
        self.design = self.DESIGN

    def forward(
        self, cls_response, cont_response, search,
        visual_reference, language_reference, flag,
    ):
        output = super().forward(
            cls_response, cont_response, search,
            visual_reference, language_reference, flag,
        )

        quality_logits = output['pq_reliability_logits']
        reliability = output['pq_reliability']
        reliability_gap = quality_logits[:, 0] - quality_logits[:, 1]
        gate_input = torch.cat([
            output['features'], reliability,
            reliability_gap.abs()[:, None],
        ], dim=-1)
        gate_logit = self.gate_head(gate_input).squeeze(-1)
        gate = torch.sigmoid(gate_logit)
        direction = torch.tanh(
            reliability_gap / max(self.direction_temperature, self.eps))
        raw_alpha = self.max_alpha * gate * direction

        active = output['active_mask']
        runtime_enabled = (
            not self.force_off and self.inference_mode == 'FULL_ERC')
        enabled = active if runtime_enabled else torch.zeros_like(active)
        effective_alpha = torch.where(
            enabled, raw_alpha, torch.zeros_like(raw_alpha))
        difference = output['pq_difference']
        raw_correction = raw_alpha[:, None, None] * difference
        correction = torch.where(
            enabled[:, None, None], raw_correction,
            torch.zeros_like(raw_correction))
        factor = correction.exp()
        base = output['base_score']
        windowed_base = output['windowed_base_score']
        calibrated = base * factor.flatten(1)
        windowed_calibrated = windowed_base * factor.flatten(1)
        intervention_strength = (
            effective_alpha.abs() / max(self.max_alpha, self.eps))

        output.update({
            'alpha': correction.flatten(1),
            'raw_alpha': effective_alpha[:, None],
            'z_alpha': reliability_gap[:, None],
            'reliability': reliability.max(-1, keepdim=True).values,
            'reliability_gate': intervention_strength[:, None],
            'reliability_base': 1.0 - intervention_strength[:, None],
            'intervention_gate': torch.where(
                enabled, gate, torch.zeros_like(gate))[:, None],
            'gate_logit': gate_logit[:, None],
            'regional_intervention_gate': intervention_strength[:, None].expand(
                -1, self.feat_sz * self.feat_sz),
            'regional_gate_logit': gate_logit[:, None].expand(
                -1, self.feat_sz * self.feat_sz),
            'delta': correction.flatten(1),
            'factor': factor.flatten(1),
            'calibrated_score': calibrated,
            'windowed_calibrated_score': windowed_calibrated,
            'global_calibrated_score': calibrated,
            'windowed_global_calibrated_score': windowed_calibrated,
            'alpha_global': effective_alpha[:, None],
            'alpha_region': correction,
            'alpha_final': correction,
            # Keep the generic PQ aliases pointed at the actually deployed v2
            # action so existing diagnostics remain meaningful.
            'pq_raw_alpha': raw_alpha,
            'pq_alpha': effective_alpha,
            'pq_raw_correction': raw_correction,
            'pq_correction': correction,
            'pq2_gate_logit': gate_logit,
            'pq2_gate': gate,
            'pq2_direction': direction,
            'pq2_raw_alpha': raw_alpha,
            'pq2_alpha': effective_alpha,
        })
        return output

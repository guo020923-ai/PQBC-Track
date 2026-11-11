"""Independent-branch response calibration for PQ-ERC v3.

PQ3 removes the coupled ``gate * direction`` scalar used by PQ2.  It predicts
two bounded, zero-initialized response-sharpness residuals instead::

    S = Hann * C * T * exp(delta_c * centre(log C)
                           + delta_t * centre(log T))

The module remains response-level: bbox geometry and candidate IoU are not
inputs.  Zero residuals exactly recover the native UVLTrack ranking.
"""

import torch
from torch import nn

from .peak_quality_calibrated_erc_pq1 import PeakQualityCalibratedERCPQ1


class PeakQualityIndependentBranchERCPQ3(PeakQualityCalibratedERCPQ1):
    """Calibrate C and T independently without a second multiplicative gate."""

    DESIGN = 'peak_quality_independent_branch_erc_pq3'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # PQ1's quality head is retained only so old initialization remains
        # structurally compatible.  It is not optimized by PQ3.
        self.quality_head.requires_grad_(False)
        decision_dim = self.quality_head[0].in_features
        hidden = self.quality_head[0].out_features
        self.control_head = nn.Sequential(
            nn.Linear(decision_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.control_head[-1].weight)
        nn.init.zeros_(self.control_head[-1].bias)
        self.design = self.DESIGN

    @staticmethod
    def _masked_center(values, support):
        weights = support.to(values.dtype)
        mean = (values * weights).flatten(1).sum(-1)
        mean = mean / weights.flatten(1).sum(-1).clamp_min(1.0)
        return (values - mean[:, None, None]) * weights

    def forward(
        self, cls_response, cont_response, search,
        visual_reference, language_reference, flag,
    ):
        output = super().forward(
            cls_response, cont_response, search,
            visual_reference, language_reference, flag,
        )
        features = output['features']
        control_logits = self.control_head(features)
        raw_delta = self.max_alpha * torch.tanh(control_logits)

        support = output['pq_support_mask']
        log_c = cls_response.clamp_min(self.eps).log()
        log_t = cont_response.clamp_min(self.eps).log()
        centred_log_c = self._masked_center(log_c, support)
        centred_log_t = self._masked_center(log_t, support)
        if self.detach_features:
            centred_log_c = centred_log_c.detach()
            centred_log_t = centred_log_t.detach()

        raw_correction = (
            raw_delta[:, 0, None, None] * centred_log_c
            + raw_delta[:, 1, None, None] * centred_log_t
        )
        active = output['active_mask']
        runtime_enabled = (
            not self.force_off and self.inference_mode == 'FULL_ERC')
        enabled = active if runtime_enabled else torch.zeros_like(active)
        effective_delta = torch.where(
            enabled[:, None], raw_delta, torch.zeros_like(raw_delta))
        correction = torch.where(
            enabled[:, None, None], raw_correction,
            torch.zeros_like(raw_correction))
        factor = correction.exp()
        base = output['base_score']
        windowed_base = output['windowed_base_score']
        calibrated = base * factor.flatten(1)
        windowed_calibrated = windowed_base * factor.flatten(1)
        intervention = (
            effective_delta.abs().amax(-1) / max(self.max_alpha, self.eps))

        # Override every generic response-control field consumed by the head
        # and tracker.  PQ1-only diagnostic keys are deliberately left unused.
        output.update({
            'alpha': correction.flatten(1),
            'raw_alpha': effective_delta[:, :1],
            'z_alpha': effective_delta[:, 1:2],
            'reliability': 1.0 - intervention[:, None],
            'reliability_gate': intervention[:, None],
            'reliability_cls': effective_delta[:, :1],
            'reliability_cont': effective_delta[:, 1:2],
            'reliability_base': 1.0 - intervention[:, None],
            'intervention_gate': intervention[:, None],
            'gate_logit': control_logits.abs().amax(-1, keepdim=True),
            'regional_intervention_gate': intervention[:, None].expand(
                -1, cls_response.shape[-1] * cls_response.shape[-2]),
            'delta': correction.flatten(1),
            'log_ratio': (centred_log_c - centred_log_t).flatten(1),
            'factor': factor.flatten(1),
            'calibrated_score': calibrated,
            'windowed_calibrated_score': windowed_calibrated,
            'global_calibrated_score': calibrated,
            'windowed_global_calibrated_score': windowed_calibrated,
            'alpha_global': effective_delta[:, :1],
            'alpha_region': correction,
            'alpha_final': correction,
            'pq3_control_logits': control_logits,
            'pq3_raw_delta': raw_delta,
            'pq3_delta': effective_delta,
            'pq3_centered_log_c': centred_log_c,
            'pq3_centered_log_t': centred_log_t,
            'pq3_raw_correction': raw_correction,
            'pq3_correction': correction,
            'inference_mode': self.inference_mode,
        })
        return output

"""Hann-aware decision Gate for response-level ERC-v8c.

The evidence calibration formula and response-only scope are unchanged.  This
version only gives the intervention Gate normalized statistics from the exact
Hann*C*T decision surface used by the tracker.  No boxes, geometry, IoU, or
future-frame state enter the module.
"""

import math

import torch

from .response_hann_hardpeak_erc_v8 import ResponseHannHardPeakERCV8


class ResponseHannGateERCV8C(ResponseHannHardPeakERCV8):
    """ERC-v8 calibration with a decision-aligned intervention Gate."""

    DESIGN = 'response_hann_gate_v8c'
    NUM_GATE_DECISION_FEATURES = 10

    def _gate_decision_features(
        self, cls_response, cont_response, height, width
    ):
        cls_safe = cls_response.flatten(1).clamp_min(self.eps)
        cont_safe = cont_response.flatten(1).clamp_min(self.eps)
        window = self.hann_window.to(dtype=cls_safe.dtype)
        windowed_base = cls_safe * cont_safe * window
        probability = self._distribution(windowed_base)
        entropy = -(
            probability * probability.clamp_min(self.eps).log()
        ).sum(dim=-1) / math.log(windowed_base.shape[-1])

        top_values, top_indices = torch.topk(
            windowed_base, k=2, dim=-1
        )
        peak_index = top_indices[:, 0]
        peak = top_values[:, 0]
        peak_margin = (top_values[:, 0] - top_values[:, 1]) / (
            peak.abs() + self.eps
        )
        concentration = (peak - windowed_base.mean(dim=-1)) / (
            peak.abs() + self.eps
        )

        def gather(values, index):
            return values.gather(1, index[:, None]).squeeze(1)

        hann_at_peak = gather(window.expand_as(cls_safe), peak_index)
        cls_support = gather(cls_safe, peak_index) / (
            cls_safe.max(dim=-1).values + self.eps
        )
        cont_support = gather(cont_safe, peak_index) / (
            cont_safe.max(dim=-1).values + self.eps
        )
        log_ratio = (cls_safe.log() - cont_safe.log()).clamp(
            -self.log_ratio_clip, self.log_ratio_clip
        )
        log_ratio_at_peak = gather(log_ratio, peak_index) / max(
            self.log_ratio_clip, self.eps
        )

        top_row = torch.div(
            top_indices, width, rounding_mode='floor'
        ).float()
        top_col = (top_indices % width).float()
        top2_distance = torch.sqrt(
            (top_row[:, 0] - top_row[:, 1]).square()
            + (top_col[:, 0] - top_col[:, 1]).square()
        ) / (math.sqrt(2.0) * max(max(height, width) - 1, 1))
        local_mass = self._local_peak_mass(
            probability, peak_index, height, width
        )

        return torch.stack(
            [
                entropy,
                peak_margin,
                concentration,
                hann_at_peak,
                cls_support,
                cont_support,
                log_ratio_at_peak,
                log_ratio_at_peak.abs(),
                top2_distance,
                local_mass,
            ],
            dim=-1,
        )

"""Continuous reliability-aware local response calibration (ERC-v12).

ERC-v12 keeps the response-only boundary of ERC-v11 but separates two
questions that v11 mixed together:

1. Is the current Hann-aware response ranking reliably correctable?
2. If it is, which bounded local residual should change that ranking?

The module predicts a continuous reliability strength from the same 5x5
response neighbourhood used by the residual branch.  The strength scales the
residual continuously at inference time; there is no hand-written test
threshold.  Candidate boxes, box geometry, IoU and search-region features are
never consumed here, leaving candidate-quality estimation to RQC.
"""

import math

import torch
from torch import nn

from .response_local_spatial_erc_v11 import ResponseLocalSpatialERCV11


class ResponseLocalReliabilityERCV12(ResponseLocalSpatialERCV11):
    """Gate a bounded local response residual with learned reliability."""

    DESIGN = 'response_local_reliability_v12'

    def __init__(self, *args, reliability_prior=0.25, **kwargs):
        super().__init__(*args, **kwargs)
        reliability_prior = float(reliability_prior)
        if not 0.0 < reliability_prior < 1.0:
            raise ValueError('ERC-v12 reliability_prior must be in (0, 1)')
        self.reliability_prior = reliability_prior
        hidden_dim = self.local_residual_head.in_features
        self.local_reliability_head = nn.Linear(hidden_dim, 1)

        # The residual head is already zero initialized by ERC-v11, so ON is
        # exactly the official baseline at initialization.  A finite prior
        # still lets correction gradients reach the residual head immediately.
        nn.init.zeros_(self.local_reliability_head.weight)
        nn.init.constant_(
            self.local_reliability_head.bias,
            math.log(reliability_prior / (1.0 - reliability_prior)),
        )

    def forward(self, cls_response, cont_response, flag=None):
        output = super().forward(cls_response, cont_response, flag=flag)
        local_latent = output['features']
        reliability_logit = self.local_reliability_head(local_latent)
        reliability = torch.sigmoid(reliability_logit)

        active_mask = output['active_mask']
        enabled_mask = active_mask & (not self.force_off) & (
            self.inference_mode == 'FULL_ERC'
        )
        effective_reliability = torch.where(
            enabled_mask[:, None], reliability, torch.zeros_like(reliability)
        )
        valid_float = output['local_valid_mask'].to(
            dtype=output['local_predicted_delta_patch'].dtype
        )
        local_delta_patch = (
            effective_reliability
            * output['local_predicted_delta_patch']
            * valid_float
        )

        base_score = output['base_score']
        delta_flat = base_score.new_zeros(base_score.shape)
        delta_flat = delta_flat.scatter_add(
            1, output['local_indices'], local_delta_patch
        )
        factor = torch.exp(delta_flat)
        calibrated_score = base_score * factor
        windowed_calibrated_score = output['windowed_base_score'] * factor
        batch_size = base_score.shape[0]
        height = cls_response.shape[-2]
        width = cls_response.shape[-1]

        output.update({
            'alpha': delta_flat,
            'raw_alpha': output['local_predicted_delta_patch'],
            'reliability': reliability,
            'reliability_gate': effective_reliability,
            'intervention_gate': effective_reliability,
            'gate_logit': reliability_logit,
            'regional_intervention_gate': effective_reliability,
            'regional_gate_logit': reliability_logit,
            'delta': delta_flat,
            'factor': factor,
            'calibrated_score': calibrated_score,
            'windowed_calibrated_score': windowed_calibrated_score,
            'alpha_region': delta_flat.reshape(batch_size, height, width),
            'alpha_final': delta_flat.reshape(batch_size, height, width),
            'local_reliability': reliability,
            'local_reliability_logit': reliability_logit,
            'local_effective_reliability': effective_reliability,
            'local_delta_patch': local_delta_patch,
            'local_delta_map': delta_flat.reshape(batch_size, height, width),
        })
        return output

"""Precision-gated global and incremental-regional response ERC-v9.

Global evidence calibration and Regional residual calibration use independent
continuous Gates.  The Global Gate decides whether a shared scalar C/T
preference is worth applying.  The Regional Gate decides whether an additional
spatial residual is useful after the Global result.  The module remains purely
response-level: it consumes no boxes, geometry, candidate features, or IoU.
"""

import math

import torch
from torch import nn

from .response_hann_gate_erc_v8c import ResponseHannGateERCV8C


class ResponsePrecisionIncrementalERCV9(ResponseHannGateERCV8C):
    """Global scalar calibration plus independently gated regional residual."""

    DESIGN = 'response_precision_incremental_v9'

    def __init__(
        self,
        feat_sz,
        hidden_dim=64,
        regional_hidden_dim=32,
        drop_rate=0.0,
        max_alpha=0.20,
        global_alpha_scale=0.10,
        region_alpha_scale=0.05,
        eps=1e-6,
        log_ratio_clip=8.0,
        detach_features=False,
        apply_vl_tracking_only=True,
        gate_prior=0.10,
        regional_gate_prior=0.02,
        detach_reliability_for_alpha=True,
        force_off=False,
        inference_mode='FULL_ERC',
        response_topk=10,
    ):
        super().__init__(
            feat_sz=feat_sz,
            hidden_dim=hidden_dim,
            regional_hidden_dim=regional_hidden_dim,
            drop_rate=drop_rate,
            max_alpha=max_alpha,
            global_alpha_scale=global_alpha_scale,
            region_alpha_scale=region_alpha_scale,
            eps=eps,
            log_ratio_clip=log_ratio_clip,
            detach_features=detach_features,
            apply_vl_tracking_only=apply_vl_tracking_only,
            gate_prior=gate_prior,
            detach_reliability_for_alpha=detach_reliability_for_alpha,
            force_off=force_off,
            inference_mode=inference_mode,
            response_topk=response_topk,
        )
        if not 0.0 < regional_gate_prior < 1.0:
            raise ValueError('regional_gate_prior must be in (0, 1)')
        regional_hidden_dim = max(int(regional_hidden_dim), 8)
        conditioned_dim = int(hidden_dim) + 5 + self.NUM_GATE_DECISION_FEATURES
        self.regional_gate_head = nn.Sequential(
            nn.Linear(conditioned_dim + regional_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.regional_gate_head[-1].weight)
        self.regional_gate_head[-1].bias.data.fill_(
            math.log(regional_gate_prior / (1.0 - regional_gate_prior))
        )

    def _regional_intervention_gate(
        self,
        gate_conditioned,
        intervention_gate,
        gate_logit,
        regional_latent,
        alpha_region,
    ):
        regional_state = regional_latent.mean(dim=(-2, -1))
        if self.detach_features:
            regional_state = regional_state.detach()
        regional_gate_logit = self.regional_gate_head(
            torch.cat([gate_conditioned, regional_state], dim=-1)
        )
        return torch.sigmoid(regional_gate_logit), regional_gate_logit

    def _compose_alpha(
        self,
        alpha_global_map,
        alpha_region,
        intervention_gate,
        regional_intervention_gate,
    ):
        if self.inference_mode == 'ERC_OFF':
            alpha_raw = torch.zeros_like(alpha_region)
            alpha_final = torch.zeros_like(alpha_region)
        elif self.inference_mode == 'GLOBAL_ONLY':
            alpha_raw = alpha_global_map
            alpha_final = intervention_gate[:, :, None] * alpha_global_map
        else:
            alpha_raw = (alpha_global_map + alpha_region).clamp(
                -self.max_alpha, self.max_alpha
            )
            alpha_final = (
                intervention_gate[:, :, None] * alpha_global_map
                + regional_intervention_gate[:, :, None] * alpha_region
            ).clamp(-self.max_alpha, self.max_alpha)
        return alpha_raw, alpha_final

"""Competition-pair response Evidence Reliability Calibration (ERC-v10).

Global ERC keeps the shared C/T reliability direction.  Regional ERC no
longer spreads an unconstrained dense residual over the response map.  It
selects the Hann-aware baseline peak and its strongest spatially distinct
competitor, predicts a desired relative log-score correction, and converts
that correction into the minimum-norm pair of C/T fusion coefficients.

The module remains response-only: it consumes C, T and their response-space
statistics, but never consumes boxes, box geometry, candidate features or IoU.
"""

import math

import torch
from torch import nn

from .response_precision_incremental_erc_v9 import (
    ResponsePrecisionIncrementalERCV9,
)


class ResponseCompetitionPairERCV10(ResponsePrecisionIncrementalERCV9):
    """Global reliability direction plus focused top1/competitor calibration."""

    DESIGN = 'response_competition_pair_v10'
    NUM_PAIR_FEATURES = 12

    def __init__(
        self,
        feat_sz,
        hidden_dim=64,
        regional_hidden_dim=32,
        drop_rate=0.0,
        max_alpha=0.20,
        global_alpha_scale=0.10,
        region_alpha_scale=0.15,
        eps=1e-6,
        log_ratio_clip=8.0,
        detach_features=False,
        apply_vl_tracking_only=True,
        gate_prior=0.10,
        regional_gate_prior=0.05,
        detach_reliability_for_alpha=True,
        force_off=False,
        inference_mode='FULL_ERC',
        response_topk=10,
        pair_beta_scale=0.30,
        pair_exclusion_radius=2.0,
        pair_mask_sigma=1.25,
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
            regional_gate_prior=regional_gate_prior,
            detach_reliability_for_alpha=detach_reliability_for_alpha,
            force_off=force_off,
            inference_mode=inference_mode,
            response_topk=response_topk,
        )
        if pair_beta_scale < 0.0:
            raise ValueError('pair_beta_scale must be non-negative')
        if pair_exclusion_radius < 0.0:
            raise ValueError('pair_exclusion_radius must be non-negative')
        if pair_mask_sigma <= 0.0:
            raise ValueError('pair_mask_sigma must be positive')
        self.pair_beta_scale = float(pair_beta_scale)
        self.pair_exclusion_radius = float(pair_exclusion_radius)
        self.pair_mask_sigma = float(pair_mask_sigma)

        gate_conditioned_dim = (
            int(hidden_dim) + 5 + self.NUM_GATE_DECISION_FEATURES
        )
        pair_input_dim = gate_conditioned_dim + self.NUM_PAIR_FEATURES
        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
        )
        self.pair_beta_head = nn.Linear(hidden_dim, 1)
        self.pair_gate_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.pair_beta_head.weight)
        nn.init.zeros_(self.pair_beta_head.bias)
        nn.init.zeros_(self.pair_gate_head.weight)
        self.pair_gate_head.bias.data.fill_(
            math.log(regional_gate_prior / (1.0 - regional_gate_prior))
        )

        rows, cols = torch.meshgrid(
            torch.arange(self.feat_sz, dtype=torch.float32),
            torch.arange(self.feat_sz, dtype=torch.float32),
            indexing='ij',
        )
        self.register_buffer(
            'pair_coordinates',
            torch.stack([rows.flatten(), cols.flatten()], dim=-1),
        )

    @staticmethod
    def _gather(flat, index):
        return flat.gather(1, index[:, None]).squeeze(1)

    def _build_regional_calibration(
        self,
        regional_latent,
        regional_features,
        cls_response,
        cont_response,
        gate_conditioned,
        intervention_gate,
        gate_logit,
        alpha_global,
        height,
        width,
    ):
        del regional_latent, regional_features, gate_logit
        batch_size = cls_response.shape[0]
        cls_flat = cls_response.flatten(1).clamp_min(self.eps)
        cont_flat = cont_response.flatten(1).clamp_min(self.eps)
        window = self.hann_window.to(dtype=cls_flat.dtype)
        windowed_base = cls_flat * cont_flat * window
        log_base = windowed_base.clamp_min(self.eps).log()
        log_ratio = (cls_flat.log() - cont_flat.log()).clamp(
            -self.log_ratio_clip, self.log_ratio_clip
        )

        # The hard indices define the two response regions and are deliberately
        # detached.  Gradients flow through their gathered C/T evidence and the
        # learned pair heads, not through argmax.
        top_index = windowed_base.argmax(dim=-1).detach()
        coordinates = self.pair_coordinates.to(dtype=cls_flat.dtype)
        top_coordinates = coordinates[top_index]
        squared_distance = (
            coordinates[None, :, :] - top_coordinates[:, None, :]
        ).square().sum(dim=-1)
        valid_competitor = squared_distance.gt(
            self.pair_exclusion_radius ** 2
        )
        competitor_index = windowed_base.masked_fill(
            ~valid_competitor, float('-inf')
        ).argmax(dim=-1).detach()
        competitor_coordinates = coordinates[competitor_index]

        top_log_ratio = self._gather(log_ratio, top_index)
        competitor_log_ratio = self._gather(log_ratio, competitor_index)
        top_log_base = self._gather(log_base, top_index)
        competitor_log_base = self._gather(log_base, competitor_index)
        cls_peak = cls_flat.max(dim=-1).values.clamp_min(self.eps)
        cont_peak = cont_flat.max(dim=-1).values.clamp_min(self.eps)
        top_base = self._gather(windowed_base, top_index).clamp_min(self.eps)
        competitor_base = self._gather(
            windowed_base, competitor_index
        ).clamp_min(self.eps)
        pair_distance = (
            top_coordinates - competitor_coordinates
        ).square().sum(dim=-1).sqrt()
        distance_scale = math.sqrt(2.0) * max(self.feat_sz - 1, 1)
        alpha_scale = max(self.global_alpha_scale, self.eps)

        pair_features = torch.stack(
            [
                top_log_ratio / self.log_ratio_clip,
                competitor_log_ratio / self.log_ratio_clip,
                (competitor_log_ratio - top_log_ratio)
                / (2.0 * self.log_ratio_clip),
                (top_log_base - competitor_log_base).clamp(0.0, 8.0) / 8.0,
                self._gather(cls_flat, top_index) / cls_peak,
                self._gather(cont_flat, top_index) / cont_peak,
                self._gather(cls_flat, competitor_index) / cls_peak,
                self._gather(cont_flat, competitor_index) / cont_peak,
                pair_distance / distance_scale,
                alpha_global.squeeze(1) / alpha_scale,
                intervention_gate.squeeze(1),
                competitor_base / top_base,
            ],
            dim=-1,
        )
        if self.detach_features:
            pair_features = pair_features.detach()
        pair_state = self.pair_encoder(
            torch.cat([gate_conditioned, pair_features], dim=-1)
        )
        pair_beta_logit = self.pair_beta_head(pair_state)
        pair_beta = self.pair_beta_scale * torch.tanh(pair_beta_logit)
        regional_gate_logit = self.pair_gate_head(pair_state)
        regional_intervention_gate = torch.sigmoid(regional_gate_logit)

        # Minimum-norm alpha pair satisfying
        #   correction_competitor - correction_top1 = pair_beta.
        # Each correction is alpha_i * (log C_i - log T_i), so ERC remains a
        # dynamic C/T evidence calibration rather than an additive score head.
        denominator = (
            top_log_ratio.square()
            + competitor_log_ratio.square()
        ).clamp_min(self.eps)
        top_alpha = (
            -pair_beta.squeeze(1) * top_log_ratio / denominator
        ).clamp(-self.region_alpha_scale, self.region_alpha_scale)
        competitor_alpha = (
            pair_beta.squeeze(1) * competitor_log_ratio / denominator
        ).clamp(-self.region_alpha_scale, self.region_alpha_scale)

        top_distance_sq = (
            coordinates[None, :, :] - top_coordinates[:, None, :]
        ).square().sum(dim=-1)
        competitor_distance_sq = (
            coordinates[None, :, :] - competitor_coordinates[:, None, :]
        ).square().sum(dim=-1)
        gaussian_denominator = 2.0 * self.pair_mask_sigma ** 2
        top_mask = torch.exp(-top_distance_sq / gaussian_denominator)
        competitor_mask = torch.exp(
            -competitor_distance_sq / gaussian_denominator
        )
        alpha_region_flat = (
            top_alpha[:, None] * top_mask
            + competitor_alpha[:, None] * competitor_mask
        ).clamp(-self.region_alpha_scale, self.region_alpha_scale)
        alpha_region = alpha_region_flat.reshape(batch_size, height, width)

        achieved_pair_delta = (
            self._gather(alpha_region_flat * log_ratio, competitor_index)
            - self._gather(alpha_region_flat * log_ratio, top_index)
        )
        diagnostics = {
            'pair_top_index': top_index,
            'pair_competitor_index': competitor_index,
            'pair_beta': pair_beta,
            'pair_beta_logit': pair_beta_logit,
            'pair_achieved_delta': achieved_pair_delta[:, None],
            'pair_base_log_margin': (
                top_log_base - competitor_log_base
            )[:, None],
            'pair_distance': pair_distance[:, None],
        }
        return (
            alpha_region,
            regional_intervention_gate,
            regional_gate_logit,
            diagnostics,
        )

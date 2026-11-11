"""Local spatial Evidence Reliability Calibration (ERC-v11).

ERC-v11 is deliberately response-only.  It observes a 5x5 neighbourhood of
the Hann-aware UVLTrack response ``S0 = Hann * C * T`` around the baseline
top-1 location and predicts bounded relative log-score residuals inside that
same neighbourhood.  It never consumes candidate boxes, box geometry, IoU or
search-region features; candidate-quality reasoning remains the responsibility
of RQC.
"""

import torch
import torch.nn.functional as F
from torch import nn


class ResponseLocalSpatialERCV11(nn.Module):
    """Apply a bounded, zero-initialized residual to a local response patch."""

    DESIGN = 'response_local_spatial_v11'
    NUM_LOCAL_CHANNELS = 9

    def __init__(
        self,
        feat_sz,
        hidden_dim=64,
        local_hidden_dim=32,
        local_window=5,
        delta_max=0.65,
        drop_rate=0.0,
        eps=1e-6,
        log_ratio_clip=8.0,
        detach_features=True,
        apply_vl_tracking_only=True,
        force_off=False,
        inference_mode='FULL_ERC',
    ):
        super().__init__()
        if int(local_window) < 3 or int(local_window) % 2 == 0:
            raise ValueError('ERC-v11 local_window must be an odd integer >= 3')
        if float(delta_max) < 0.0:
            raise ValueError('ERC-v11 delta_max must be non-negative')

        self.feat_sz = int(feat_sz)
        self.local_window = int(local_window)
        self.local_radius = self.local_window // 2
        self.delta_max = float(delta_max)
        self.eps = float(eps)
        self.log_ratio_clip = float(log_ratio_clip)
        self.detach_features = bool(detach_features)
        self.apply_vl_tracking_only = bool(apply_vl_tracking_only)
        self.force_off = bool(force_off)
        self.design = self.DESIGN
        self.inference_mode = self._normalize_inference_mode(inference_mode)

        window_1d = torch.hann_window(self.feat_sz, periodic=False)
        hann_2d = torch.outer(window_1d, window_1d)
        self.register_buffer('hann_window', hann_2d.flatten().unsqueeze(0))

        offsets = torch.arange(-self.local_radius, self.local_radius + 1)
        offset_rows, offset_cols = torch.meshgrid(
            offsets, offsets, indexing='ij'
        )
        self.register_buffer('local_offset_rows', offset_rows.flatten())
        self.register_buffer('local_offset_cols', offset_cols.flatten())

        local_hidden_dim = max(int(local_hidden_dim), 8)
        hidden_dim = max(int(hidden_dim), 16)
        self.local_encoder = nn.Sequential(
            nn.Conv2d(
                self.NUM_LOCAL_CHANNELS,
                local_hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                local_hidden_dim,
                local_hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
        )
        self.local_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                local_hidden_dim * self.local_window * self.local_window,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
        )
        self.local_residual_head = nn.Linear(
            hidden_dim, self.local_window * self.local_window
        )

        # Exact identity at initialization: exp(delta)==1 and ERC ON is
        # element-wise equal to the official Hann*C*T ranking.
        nn.init.zeros_(self.local_residual_head.weight)
        nn.init.zeros_(self.local_residual_head.bias)

    @staticmethod
    def _normalize_inference_mode(mode):
        aliases = {
            'OFF': 'ERC_OFF',
            'ERC_OFF': 'ERC_OFF',
            'FULL': 'FULL_ERC',
            'FULL_ERC': 'FULL_ERC',
        }
        normalized = str(mode).upper()
        if normalized not in aliases:
            raise ValueError(
                'ERC-v11 supports only ERC_OFF or FULL_ERC, got %s' % mode
            )
        return aliases[normalized]

    def set_inference_mode(self, mode):
        self.inference_mode = self._normalize_inference_mode(mode)

    def _local_indices(self, top_index, height, width):
        top_row = torch.div(top_index, width, rounding_mode='floor')
        top_col = top_index % width
        rows = top_row[:, None] + self.local_offset_rows[None, :]
        cols = top_col[:, None] + self.local_offset_cols[None, :]
        valid = rows.ge(0) & rows.lt(height) & cols.ge(0) & cols.lt(width)
        safe_rows = rows.clamp(0, height - 1)
        safe_cols = cols.clamp(0, width - 1)
        return safe_rows * width + safe_cols, valid

    def _extract_local_features(
        self, cls_response, cont_response, windowed_base, top_index
    ):
        batch_size, height, width = cls_response.shape
        cls_safe = cls_response.clamp_min(self.eps)
        cont_safe = cont_response.clamp_min(self.eps)
        window_2d = self.hann_window.reshape(1, height, width).to(
            dtype=cls_response.dtype
        )
        base_safe = windowed_base.reshape(batch_size, height, width).clamp_min(
            self.eps
        )

        cls_flat = cls_safe.flatten(1)
        cont_flat = cont_safe.flatten(1)
        base_flat = base_safe.flatten(1)
        cls_peak = cls_flat.max(dim=-1).values[:, None, None]
        cont_peak = cont_flat.max(dim=-1).values[:, None, None]
        base_peak = base_flat.max(dim=-1).values[:, None, None]
        valid_map = torch.ones_like(cls_safe)

        channels = torch.stack(
            [
                cls_safe.log().clamp(-self.log_ratio_clip, 0.0)
                / self.log_ratio_clip,
                cont_safe.log().clamp(-self.log_ratio_clip, 0.0)
                / self.log_ratio_clip,
                base_safe.log().clamp(-2.0 * self.log_ratio_clip, 0.0)
                / (2.0 * self.log_ratio_clip),
                (
                    cls_safe.log() - cont_safe.log()
                ).clamp(-self.log_ratio_clip, self.log_ratio_clip)
                / self.log_ratio_clip,
                cls_safe / cls_peak.clamp_min(self.eps),
                cont_safe / cont_peak.clamp_min(self.eps),
                base_safe / base_peak.clamp_min(self.eps),
                window_2d.expand(batch_size, -1, -1),
                valid_map,
            ],
            dim=1,
        )
        patches = F.unfold(
            channels,
            kernel_size=self.local_window,
            padding=self.local_radius,
        )
        gather_index = top_index[:, None, None].expand(
            -1, patches.shape[1], 1
        )
        local = patches.gather(2, gather_index).squeeze(2)
        return local.reshape(
            batch_size,
            self.NUM_LOCAL_CHANNELS,
            self.local_window,
            self.local_window,
        )

    def forward(self, cls_response, cont_response, flag=None):
        if cls_response.ndim != 3 or cont_response.ndim != 3:
            raise ValueError('ERC-v11 expects [B,H,W] response maps')
        if cls_response.shape != cont_response.shape:
            raise ValueError('ERC-v11 C/T response shapes must be identical')
        batch_size, height, width = cls_response.shape
        if height != self.feat_sz or width != self.feat_sz:
            raise ValueError(
                'Configured feat_sz=%d but responses are %dx%d'
                % (self.feat_sz, height, width)
            )

        cls_flat = cls_response.flatten(1)
        cont_flat = cont_response.flatten(1)
        base_score = cls_flat * cont_flat
        window = self.hann_window.to(dtype=base_score.dtype)
        windowed_base_score = base_score * window

        # The baseline top-1 defines only where the local response patch is
        # extracted.  It is a detached discrete routing decision.
        top_index = windowed_base_score.argmax(dim=-1).detach()
        local_indices, local_valid_mask = self._local_indices(
            top_index, height, width
        )
        local_features = self._extract_local_features(
            cls_response, cont_response, windowed_base_score, top_index
        )
        encoder_input = (
            local_features.detach() if self.detach_features else local_features
        )
        local_latent_map = self.local_encoder(encoder_input)
        local_latent = self.local_projection(local_latent_map)
        local_logits = self.local_residual_head(local_latent)

        valid_float = local_valid_mask.to(dtype=local_logits.dtype)
        valid_count = valid_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
        local_mean = (local_logits * valid_float).sum(
            dim=-1, keepdim=True
        ) / valid_count
        centered_logits = (local_logits - local_mean) * valid_float
        predicted_delta_patch = (
            self.delta_max * torch.tanh(centered_logits) * valid_float
        )

        if self.apply_vl_tracking_only:
            if flag is None:
                raise ValueError('flag is required for VL-tracking-only ERC-v11')
            active_mask = flag.reshape(-1).eq(2)
        else:
            active_mask = torch.ones(
                batch_size, dtype=torch.bool, device=cls_response.device
            )
        enabled_mask = active_mask & (not self.force_off) & (
            self.inference_mode == 'FULL_ERC'
        )
        local_delta_patch = torch.where(
            enabled_mask[:, None],
            predicted_delta_patch,
            torch.zeros_like(predicted_delta_patch),
        )

        delta_flat = base_score.new_zeros(base_score.shape)
        delta_flat = delta_flat.scatter_add(
            1,
            local_indices,
            local_delta_patch * valid_float,
        )
        factor = torch.exp(delta_flat)
        calibrated_score = base_score * factor
        windowed_calibrated_score = windowed_base_score * factor

        zeros_scalar = base_score.new_zeros(batch_size, 1)
        ones_scalar = base_score.new_ones(batch_size, 1)
        output = {
            'features': local_latent,
            'cls_response': cls_flat,
            'cont_response': cont_flat,
            'windowed_cls_score': cls_flat * window,
            'windowed_cont_score': cont_flat * window,
            'alpha': delta_flat,
            'raw_alpha': predicted_delta_patch,
            'z_alpha': local_logits,
            'reliability': ones_scalar,
            'reliability_gate': active_mask.to(base_score.dtype)[:, None],
            'reliability_cls': ones_scalar,
            'reliability_cont': ones_scalar,
            'reliability_base': ones_scalar,
            'intervention_gate': active_mask.to(base_score.dtype)[:, None],
            'gate_logit': zeros_scalar,
            'regional_intervention_gate': active_mask.to(base_score.dtype)[:, None],
            'regional_gate_logit': zeros_scalar,
            'gate_decision_features': base_score.new_zeros(batch_size, 0),
            'delta': delta_flat,
            'log_ratio': (
                cls_flat.clamp_min(self.eps).log()
                - cont_flat.clamp_min(self.eps).log()
            ).clamp(-self.log_ratio_clip, self.log_ratio_clip),
            'factor': factor,
            'base_score': base_score,
            'windowed_base_score': windowed_base_score,
            'calibrated_score': calibrated_score,
            'windowed_calibrated_score': windowed_calibrated_score,
            'global_calibrated_score': base_score,
            'windowed_global_calibrated_score': windowed_base_score,
            'alpha_global': zeros_scalar,
            'alpha_region': delta_flat.reshape(batch_size, height, width),
            'alpha_final': delta_flat.reshape(batch_size, height, width),
            'regional_latent': local_latent_map,
            'active_mask': active_mask,
            'inference_mode': self.inference_mode,
            'local_top_index': top_index,
            'local_indices': local_indices.detach(),
            'local_valid_mask': local_valid_mask.detach(),
            'local_features': local_features,
            'local_logits': local_logits,
            'local_centered_logits': centered_logits,
            'local_predicted_delta_patch': predicted_delta_patch,
            'local_delta_patch': local_delta_patch,
            'local_delta_map': delta_flat.reshape(batch_size, height, width),
        }
        return output

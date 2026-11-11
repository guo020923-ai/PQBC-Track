"""Branch-constrained spatial evidence calibration (ERC-v13).

ERC-v13 observes only C, T, the official Hann-aware fusion, their stabilized
log difference, and the Hann prior.  It predicts a signed branch coefficient
and one smooth non-negative spatial gate:

    S_ERC(x) = S0(x) * exp(beta * g(x) * D(x)).

The global sign of beta decides whether the frame leans toward classification
or contrastive evidence.  The gate can only localize that same branch choice;
it cannot create an unconstrained response residual.  No box, geometry, IoU,
or search feature is consumed by this module.
"""

import torch
from torch import nn


class ResponseBranchSpatialGateERCV13(nn.Module):
    DESIGN = 'response_branch_spatial_gate_v13'
    NUM_INPUT_CHANNELS = 5

    def __init__(
        self,
        feat_sz,
        hidden_dim=64,
        spatial_hidden_dim=32,
        drop_rate=0.0,
        beta_max=0.5,
        d_max=4.0,
        sigmas=(0.75, 1.25, 2.0, 3.0),
        eps=1e-6,
        log_c_min=-8.0,
        log_t_min=-16.0,
        log_s0_min=-16.0,
        detach_features=True,
        apply_vl_tracking_only=True,
        force_off=False,
        inference_mode='FULL_ERC',
    ):
        super().__init__()
        if float(beta_max) < 0.0:
            raise ValueError('ERC-v13 beta_max must be non-negative')
        if float(d_max) <= 0.0:
            raise ValueError('ERC-v13 d_max must be positive')
        sigma_values = sorted({float(value) for value in sigmas})
        if not sigma_values or any(value <= 0.0 for value in sigma_values):
            raise ValueError('ERC-v13 sigmas must be positive')

        self.feat_sz = int(feat_sz)
        self.beta_max = float(beta_max)
        self.d_max = float(d_max)
        self.eps = float(eps)
        self.log_c_min = float(log_c_min)
        self.log_t_min = float(log_t_min)
        self.log_s0_min = float(log_s0_min)
        self.detach_features = bool(detach_features)
        self.apply_vl_tracking_only = bool(apply_vl_tracking_only)
        self.force_off = bool(force_off)
        self.design = self.DESIGN
        self.inference_mode = self._normalize_inference_mode(inference_mode)

        window_1d = torch.hann_window(self.feat_sz, periodic=False)
        hann_2d = torch.outer(window_1d, window_1d)
        self.register_buffer('hann_window', hann_2d.flatten().unsqueeze(0))
        self.register_buffer(
            'sigma_values', torch.tensor(sigma_values, dtype=torch.float32)
        )
        coordinates = torch.arange(self.feat_sz, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(
            coordinates, coordinates, indexing='ij'
        )
        # torch.meshgrid may return expanded views with overlapping storage.
        # Checkpoint loading copies buffers in-place, which PyTorch rejects for
        # such views even when the shapes match.  Store independent contiguous
        # tensors; this changes no coordinate value or model computation.
        self.register_buffer('grid_y', grid_y.contiguous())
        self.register_buffer('grid_x', grid_x.contiguous())

        spatial_hidden_dim = max(int(spatial_hidden_dim), 8)
        hidden_dim = max(int(hidden_dim), 16)
        self.encoder = nn.Sequential(
            nn.Conv2d(
                self.NUM_INPUT_CHANNELS, spatial_hidden_dim,
                kernel_size=3, padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                spatial_hidden_dim, spatial_hidden_dim,
                kernel_size=3, padding=1,
            ),
            nn.GELU(),
        )
        self.global_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(spatial_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
        )
        self.center_head = nn.Conv2d(spatial_hidden_dim, 1, kernel_size=1)
        # The branch decision must see the evidence at the predicted gate,
        # not only a spatial average.  Otherwise centered D is largely erased
        # by global pooling and beta can collapse to one dataset-wide bias.
        self.decision_projection = nn.Sequential(
            nn.Linear(
                hidden_dim + spatial_hidden_dim + self.NUM_INPUT_CHANNELS,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
        )
        self.beta_head = nn.Linear(hidden_dim, 1)
        self.sigma_head = nn.Linear(hidden_dim, len(sigma_values))

        # beta==0 makes factor exactly one regardless of the initial center or
        # sigma.  Auxiliary response-only targets still train those two heads.
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)
        nn.init.zeros_(self.center_head.weight)
        nn.init.zeros_(self.center_head.bias)
        nn.init.zeros_(self.sigma_head.weight)
        nn.init.zeros_(self.sigma_head.bias)

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
                'ERC-v13 supports only ERC_OFF or FULL_ERC, got %s' % mode
            )
        return aliases[normalized]

    def set_inference_mode(self, mode):
        self.inference_mode = self._normalize_inference_mode(mode)

    def _stable_channels(self, cls_response, cont_response):
        batch_size, height, width = cls_response.shape
        cls_safe = cls_response.clamp_min(self.eps)
        cont_safe = cont_response.clamp_min(self.eps)
        window = self.hann_window.reshape(1, height, width).to(
            dtype=cls_response.dtype
        )
        s0 = cls_safe * cont_safe * window
        log_c = cls_safe.log()
        log_t = cont_safe.log()
        log_s0 = (s0 + self.eps).log()
        d_raw = log_c - log_t
        d_centered = d_raw - d_raw.flatten(1).mean(dim=-1)[:, None, None]
        d = d_centered.clamp(-self.d_max, self.d_max)

        def normalize_log(value, lower):
            scale = max(abs(float(lower)), self.eps)
            return value.clamp(float(lower), 0.0) / scale

        channels = torch.stack([
            normalize_log(log_c, self.log_c_min),
            normalize_log(log_t, self.log_t_min),
            normalize_log(log_s0, self.log_s0_min),
            d / self.d_max,
            window.expand(batch_size, -1, -1),
        ], dim=1)
        return channels, d, s0, window

    def decision_from_center_probability(
        self, spatial_features, encoder_input, center_probability
    ):
        """Predict branch direction/width at a supplied response center.

        Training can supply a detached Oracle-anchor distribution during the
        early curriculum.  Inference always supplies the center predicted by
        ``center_head``.  Both paths share every decision parameter.
        """
        global_features = self.global_projection(spatial_features)
        local_spatial_features = (
            spatial_features.flatten(2)
            * center_probability[:, None, :]
        ).sum(dim=-1)
        local_evidence_features = (
            encoder_input.flatten(2)
            * center_probability[:, None, :]
        ).sum(dim=-1)
        decision_features = self.decision_projection(torch.cat([
            global_features,
            local_spatial_features,
            local_evidence_features,
        ], dim=-1))
        beta_logit = self.beta_head(decision_features)
        sigma_logits = self.sigma_head(decision_features)
        return {
            'global_features': global_features,
            'local_spatial_features': local_spatial_features,
            'local_evidence_features': local_evidence_features,
            'decision_features': decision_features,
            'beta_logit': beta_logit,
            'sigma_logits': sigma_logits,
        }

    def forward(self, cls_response, cont_response, flag=None):
        if cls_response.ndim != 3 or cont_response.ndim != 3:
            raise ValueError('ERC-v13 expects [B,H,W] response maps')
        if cls_response.shape != cont_response.shape:
            raise ValueError('ERC-v13 C/T response shapes must be identical')
        batch_size, height, width = cls_response.shape
        if height != self.feat_sz or width != self.feat_sz:
            raise ValueError(
                'Configured feat_sz=%d but responses are %dx%d'
                % (self.feat_sz, height, width)
            )

        channels, d_map, s0_map, window_2d = self._stable_channels(
            cls_response, cont_response
        )
        encoder_input = channels.detach() if self.detach_features else channels
        spatial_features = self.encoder(encoder_input)
        center_logits = self.center_head(spatial_features).flatten(1)
        center_probability = center_logits.softmax(dim=-1)
        decision = self.decision_from_center_probability(
            spatial_features, encoder_input, center_probability
        )
        global_features = decision['global_features']
        local_spatial_features = decision['local_spatial_features']
        local_evidence_features = decision['local_evidence_features']
        decision_features = decision['decision_features']
        beta_logit = decision['beta_logit']
        predicted_beta = self.beta_max * torch.tanh(beta_logit)
        sigma_logits = decision['sigma_logits']
        sigma_probability = sigma_logits.softmax(dim=-1)
        sigma_values = self.sigma_values.to(dtype=cls_response.dtype)
        predicted_sigma = (
            sigma_probability * sigma_values[None, :]
        ).sum(dim=-1, keepdim=True)

        flat_grid_y = self.grid_y.flatten().to(dtype=cls_response.dtype)
        flat_grid_x = self.grid_x.flatten().to(dtype=cls_response.dtype)
        center_y = (center_probability * flat_grid_y[None, :]).sum(
            dim=-1, keepdim=True
        )
        center_x = (center_probability * flat_grid_x[None, :]).sum(
            dim=-1, keepdim=True
        )
        distance_sq = (
            (self.grid_y[None].to(dtype=cls_response.dtype)
             - center_y[:, :, None]).square()
            + (self.grid_x[None].to(dtype=cls_response.dtype)
               - center_x[:, :, None]).square()
        )
        spatial_gate = torch.exp(
            -distance_sq / (2.0 * predicted_sigma[:, :, None].square())
        )

        if self.apply_vl_tracking_only:
            if flag is None:
                raise ValueError('flag is required for VL-tracking-only ERC-v13')
            active_mask = flag.reshape(-1).eq(2)
        else:
            active_mask = torch.ones(
                batch_size, dtype=torch.bool, device=cls_response.device
            )
        enabled_mask = active_mask & (not self.force_off) & (
            self.inference_mode == 'FULL_ERC'
        )
        effective_beta = torch.where(
            enabled_mask[:, None], predicted_beta,
            torch.zeros_like(predicted_beta),
        )
        correction_map = effective_beta[:, :, None] * spatial_gate * d_map
        factor = correction_map.exp().flatten(1)

        cls_flat = cls_response.flatten(1)
        cont_flat = cont_response.flatten(1)
        base_score = cls_flat * cont_flat
        window = self.hann_window.to(dtype=base_score.dtype)
        windowed_base_score = base_score * window
        calibrated_score = base_score * factor
        windowed_calibrated_score = windowed_base_score * factor
        zeros = base_score.new_zeros(batch_size, 1)
        ones = base_score.new_ones(batch_size, 1)
        return {
            'features': decision_features,
            'cls_response': cls_flat,
            'cont_response': cont_flat,
            'windowed_cls_score': cls_flat * window,
            'windowed_cont_score': cont_flat * window,
            'alpha': correction_map.flatten(1),
            'raw_alpha': predicted_beta,
            'z_alpha': beta_logit,
            'reliability': effective_beta.abs() / max(self.beta_max, self.eps),
            'reliability_gate': effective_beta.abs() / max(
                self.beta_max, self.eps
            ),
            'reliability_cls': ones,
            'reliability_cont': ones,
            'reliability_base': ones,
            'intervention_gate': effective_beta.abs() / max(
                self.beta_max, self.eps
            ),
            'gate_logit': beta_logit,
            'regional_intervention_gate': spatial_gate.flatten(1),
            'regional_gate_logit': center_logits,
            'gate_decision_features': decision_features,
            'delta': correction_map.flatten(1),
            'log_ratio': d_map.flatten(1),
            'factor': factor,
            'base_score': base_score,
            'windowed_base_score': windowed_base_score,
            'calibrated_score': calibrated_score,
            'windowed_calibrated_score': windowed_calibrated_score,
            'global_calibrated_score': base_score,
            'windowed_global_calibrated_score': windowed_base_score,
            'alpha_global': effective_beta,
            'alpha_region': correction_map,
            'alpha_final': correction_map,
            'regional_latent': spatial_features,
            'active_mask': active_mask,
            'inference_mode': self.inference_mode,
            'v13_input_channels': channels,
            'v13_encoder_input': encoder_input,
            'v13_spatial_features': spatial_features,
            'v13_global_features': global_features,
            'v13_local_spatial_features': local_spatial_features,
            'v13_local_evidence_features': local_evidence_features,
            'v13_decision_features': decision_features,
            'v13_d_map': d_map,
            'v13_s0_map': s0_map,
            'v13_hann_map': window_2d,
            'v13_beta': predicted_beta,
            'v13_effective_beta': effective_beta,
            'v13_beta_logit': beta_logit,
            'v13_center_logits': center_logits,
            'v13_center_probability': center_probability,
            'v13_center_x': center_x,
            'v13_center_y': center_y,
            'v13_sigma_logits': sigma_logits,
            'v13_sigma_probability': sigma_probability,
            'v13_sigma': predicted_sigma,
            'v13_spatial_gate': spatial_gate,
            'v13_correction_map': correction_map,
            'v13_zero_scalar': zeros,
        }

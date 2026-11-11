"""Response-level global-regional Evidence Reliability Calibration (ERC-v7).

This module only calibrates the relationship between UVLTrack's classification
response ``C`` and contrastive response ``T``.  It deliberately has no access
to candidate boxes, candidate features, or candidate IoU.  Candidate-quality
reasoning belongs to the future RQC module.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class ResponseGlobalRegionalERCV7(nn.Module):
    """Global preference plus a continuous regional response residual."""

    DESIGN = 'response_global_regional_v7'
    NUM_GLOBAL_FEATURES = 17
    NUM_REGIONAL_CHANNELS = 8
    NUM_GATE_DECISION_FEATURES = 0

    def __init__(
        self,
        feat_sz,
        hidden_dim=64,
        regional_hidden_dim=32,
        drop_rate=0.0,
        max_alpha=0.20,
        global_alpha_scale=0.10,
        region_alpha_scale=0.10,
        eps=1e-6,
        log_ratio_clip=8.0,
        detach_features=False,
        apply_vl_tracking_only=True,
        gate_prior=0.05,
        detach_reliability_for_alpha=True,
        force_off=False,
        inference_mode='FULL_ERC',
        response_topk=10,
    ):
        super().__init__()
        if not 0.0 < gate_prior < 1.0:
            raise ValueError('gate_prior must be in (0, 1)')
        if min(max_alpha, global_alpha_scale, region_alpha_scale) < 0.0:
            raise ValueError('alpha limits must be non-negative')

        self.feat_sz = int(feat_sz)
        self.max_alpha = float(max_alpha)
        self.global_alpha_scale = float(global_alpha_scale)
        self.region_alpha_scale = float(region_alpha_scale)
        self.eps = float(eps)
        self.log_ratio_clip = float(log_ratio_clip)
        self.detach_features = bool(detach_features)
        self.apply_vl_tracking_only = bool(apply_vl_tracking_only)
        self.detach_reliability_for_alpha = bool(
            detach_reliability_for_alpha
        )
        self.force_off = bool(force_off)
        self.response_topk = max(int(response_topk), 1)
        self.design = self.DESIGN
        self.inference_mode = self._normalize_inference_mode(inference_mode)

        window_1d = torch.hann_window(self.feat_sz, periodic=False)
        self.register_buffer(
            'hann_window',
            torch.outer(window_1d, window_1d).flatten().unsqueeze(0),
        )

        self.global_encoder = nn.Sequential(
            nn.Linear(self.NUM_GLOBAL_FEATURES, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
        )
        self.reliability_head = nn.Linear(hidden_dim, 3)
        conditioned_dim = hidden_dim + 5
        self.global_alpha_head = nn.Sequential(
            nn.Linear(conditioned_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, 1),
        )
        gate_conditioned_dim = (
            conditioned_dim + self.NUM_GATE_DECISION_FEATURES
        )
        self.gate_head = nn.Sequential(
            nn.Linear(gate_conditioned_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, 1),
        )

        regional_hidden_dim = max(int(regional_hidden_dim), 8)
        self.regional_encoder = nn.Sequential(
            nn.Conv2d(
                self.NUM_REGIONAL_CHANNELS,
                regional_hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                regional_hidden_dim,
                regional_hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
        )
        self.regional_output = nn.Conv2d(
            regional_hidden_dim, 1, kernel_size=1
        )

        # Reliability starts from a neutral 0.5 prediction.  Both calibration
        # branches are exactly zero so an untrained ERC-v7 recovers C*T.
        nn.init.zeros_(self.reliability_head.weight)
        nn.init.zeros_(self.reliability_head.bias)
        nn.init.zeros_(self.global_alpha_head[-1].weight)
        nn.init.zeros_(self.global_alpha_head[-1].bias)
        nn.init.zeros_(self.regional_output.weight)
        nn.init.zeros_(self.regional_output.bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        self.gate_head[-1].bias.data.fill_(
            math.log(gate_prior / (1.0 - gate_prior))
        )

    @staticmethod
    def _normalize_inference_mode(mode):
        aliases = {
            'OFF': 'ERC_OFF',
            'ERC_OFF': 'ERC_OFF',
            'GLOBAL': 'GLOBAL_ONLY',
            'GLOBAL_ONLY': 'GLOBAL_ONLY',
            'FULL': 'FULL_ERC',
            'FULL_ERC': 'FULL_ERC',
        }
        normalized = str(mode).upper()
        if normalized not in aliases:
            raise ValueError('Unsupported ERC-v7 inference mode: %s' % mode)
        return aliases[normalized]

    def set_inference_mode(self, mode):
        self.inference_mode = self._normalize_inference_mode(mode)

    def _regional_intervention_gate(
        self,
        gate_conditioned,
        intervention_gate,
        gate_logit,
        regional_latent,
        alpha_region,
    ):
        """Return the legacy shared Gate for v7/v8 compatibility."""
        return intervention_gate, gate_logit

    def _compose_alpha(
        self,
        alpha_global_map,
        alpha_region,
        intervention_gate,
        regional_intervention_gate,
    ):
        """Compose calibration while preserving all pre-v9 behavior."""
        if self.inference_mode == 'ERC_OFF':
            alpha_raw = torch.zeros_like(alpha_region)
        elif self.inference_mode == 'GLOBAL_ONLY':
            alpha_raw = alpha_global_map
        else:
            alpha_raw = alpha_global_map + alpha_region
        alpha_raw = alpha_raw.clamp(-self.max_alpha, self.max_alpha)
        alpha_final = intervention_gate[:, :, None] * alpha_raw
        return alpha_raw, alpha_final

    def _distribution(self, response):
        safe = response.clamp_min(self.eps)
        return safe / safe.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def _statistics(self, response):
        probability = self._distribution(response)
        entropy = -(probability * probability.clamp_min(self.eps).log()).sum(
            dim=-1
        ) / math.log(response.shape[-1])
        top_values, top_indices = torch.topk(response, k=2, dim=-1)
        peak = top_values[:, 0]
        margin = (top_values[:, 0] - top_values[:, 1]) / (
            top_values[:, 0].abs() + self.eps
        )
        return probability, entropy, peak, margin, top_indices[:, 0]

    def _local_peak_mass(self, probability, peak_index, height, width):
        probability_2d = probability.reshape(-1, 1, height, width)
        local_mass = F.avg_pool2d(
            probability_2d, kernel_size=3, stride=1, padding=1
        ) * 9.0
        return local_mass.flatten(1).gather(1, peak_index[:, None]).squeeze(1)

    def _extract_global_features(self, cls_response, cont_response, height, width):
        # Hann is intentionally absent: ERC-v7 sees evidence only.
        base_response = cls_response * cont_response
        cls_prob, cls_entropy, cls_peak, cls_margin, cls_index = \
            self._statistics(cls_response)
        cont_prob, cont_entropy, cont_peak, cont_margin, cont_index = \
            self._statistics(cont_response)
        _, base_entropy, base_peak, base_margin, _ = \
            self._statistics(base_response)

        cls_row = torch.div(cls_index, width, rounding_mode='floor')
        cls_col = cls_index % width
        cont_row = torch.div(cont_index, width, rounding_mode='floor')
        cont_col = cont_index % width
        peak_distance = torch.sqrt(
            (cls_row - cont_row).float().square()
            + (cls_col - cont_col).float().square()
        ) / (math.sqrt(2.0) * max(max(height, width) - 1, 1))

        midpoint = 0.5 * (cls_prob + cont_prob)
        js_divergence = 0.5 * (
            cls_prob
            * (
                cls_prob.clamp_min(self.eps).log()
                - midpoint.clamp_min(self.eps).log()
            )
        ).sum(dim=-1)
        js_divergence += 0.5 * (
            cont_prob
            * (
                cont_prob.clamp_min(self.eps).log()
                - midpoint.clamp_min(self.eps).log()
            )
        ).sum(dim=-1)
        js_divergence = js_divergence / math.log(2.0)
        total_variation = 0.5 * (cls_prob - cont_prob).abs().sum(dim=-1)
        base_concentration = (base_peak - base_response.mean(dim=-1)) / (
            base_peak.abs() + self.eps
        )

        cls_local = self._local_peak_mass(
            cls_prob, cls_index, height, width
        )
        cont_local = self._local_peak_mass(
            cont_prob, cont_index, height, width
        )
        topk = min(self.response_topk, cls_response.shape[-1])
        cls_topk = cls_response.topk(topk, dim=-1).indices
        cont_topk = cont_response.topk(topk, dim=-1).indices
        overlap = (
            cls_topk[:, :, None] == cont_topk[:, None, :]
        ).any(dim=-1).float().mean(dim=-1)
        peak_difference = (cls_peak - cont_peak) / (
            cls_peak.abs() + cont_peak.abs() + self.eps
        )

        return torch.stack(
            [
                cls_entropy,
                cont_entropy,
                base_entropy,
                cls_peak,
                cont_peak,
                base_peak,
                cls_margin,
                cont_margin,
                base_margin,
                peak_distance,
                js_divergence,
                total_variation,
                base_concentration,
                cls_local,
                cont_local,
                overlap,
                peak_difference,
            ],
            dim=-1,
        )

    def _regional_features(self, cls_map, cont_map):
        cls_safe = cls_map.clamp_min(self.eps)
        cont_safe = cont_map.clamp_min(self.eps)
        cls_flat = cls_safe.flatten(1)
        cont_flat = cont_safe.flatten(1)
        cls_prob = self._distribution(cls_flat).reshape_as(cls_map)
        cont_prob = self._distribution(cont_flat).reshape_as(cont_map)
        cls_peak = cls_flat.max(dim=-1).values[:, None, None]
        cont_peak = cont_flat.max(dim=-1).values[:, None, None]
        log_cls = cls_safe.log()
        log_cont = cont_safe.log()
        log_ratio = (log_cls - log_cont).clamp(
            min=-self.log_ratio_clip, max=self.log_ratio_clip
        )
        return torch.stack(
            [
                log_cls,
                log_cont,
                log_ratio,
                cls_prob,
                cont_prob,
                cls_map * cont_map,
                cls_safe / cls_peak.clamp_min(self.eps),
                cont_safe / cont_peak.clamp_min(self.eps),
            ],
            dim=1,
        )

    def _gate_decision_features(
        self, cls_response, cont_response, height, width
    ):
        """Optional inference-available decision features for newer Gates."""
        return cls_response.new_zeros(
            cls_response.shape[0], self.NUM_GATE_DECISION_FEATURES
        )

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
        """Build the legacy dense Regional residual.

        New response-only ERC variants can override this hook without copying
        the complete C/T fusion forward pass.  Returning an empty diagnostic
        dictionary keeps every v7-v9 output and checkpoint path unchanged.
        """
        delta_region = self.regional_output(regional_latent)
        delta_region = F.interpolate(
            delta_region,
            size=(height, width),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)
        delta_centered = delta_region - delta_region.mean(
            dim=(-2, -1), keepdim=True
        )
        alpha_region = self.region_alpha_scale * torch.tanh(delta_centered)
        regional_intervention_gate, regional_gate_logit = (
            self._regional_intervention_gate(
                gate_conditioned,
                intervention_gate,
                gate_logit,
                regional_latent,
                alpha_region,
            )
        )
        return (
            alpha_region,
            regional_intervention_gate,
            regional_gate_logit,
            {},
        )

    def forward(self, cls_response, cont_response, flag=None):
        if cls_response.ndim != 3 or cont_response.ndim != 3:
            raise ValueError(
                'ERC-v7 expects [B,H,W] response maps, got %s and %s'
                % (tuple(cls_response.shape), tuple(cont_response.shape))
            )
        if cls_response.shape != cont_response.shape:
            raise ValueError('C/T response shapes must be identical')
        batch_size, height, width = cls_response.shape
        if height != self.feat_sz or width != self.feat_sz:
            raise ValueError(
                'Configured feat_sz=%d but responses are %dx%d'
                % (self.feat_sz, height, width)
            )

        cls_flat = cls_response.flatten(1)
        cont_flat = cont_response.flatten(1)
        cls_safe = cls_flat.clamp_min(self.eps)
        cont_safe = cont_flat.clamp_min(self.eps)
        global_features = self._extract_global_features(
            cls_safe, cont_safe, height, width
        )
        encoder_input = (
            global_features.detach()
            if self.detach_features
            else global_features
        )
        encoded = self.global_encoder(encoder_input)
        reliability_logits = self.reliability_head(encoded)
        reliability_cls = torch.sigmoid(reliability_logits[:, 0:1])
        reliability_cont = torch.sigmoid(reliability_logits[:, 1:2])
        reliability_base = torch.sigmoid(reliability_logits[:, 2:3])
        reliability_state = torch.cat(
            [
                reliability_cls,
                reliability_cont,
                reliability_base,
                reliability_cls - reliability_cont,
                (reliability_cls - reliability_cont).abs(),
            ],
            dim=-1,
        )
        if self.detach_reliability_for_alpha:
            reliability_state = reliability_state.detach()
        conditioned = torch.cat([encoded, reliability_state], dim=-1)

        z_global = self.global_alpha_head(conditioned)
        alpha_global = self.global_alpha_scale * torch.tanh(z_global)
        gate_decision_features = self._gate_decision_features(
            cls_response, cont_response, height, width
        )
        if gate_decision_features.shape != (
            batch_size, self.NUM_GATE_DECISION_FEATURES
        ):
            raise ValueError(
                'Gate decision feature shape mismatch: %s'
                % (tuple(gate_decision_features.shape),)
            )
        if self.detach_features:
            gate_decision_features = gate_decision_features.detach()
        gate_conditioned = torch.cat(
            [conditioned, gate_decision_features], dim=-1
        )
        gate_logit = self.gate_head(gate_conditioned)
        intervention_gate = torch.sigmoid(gate_logit)

        regional_cls = (
            cls_response.detach() if self.detach_features else cls_response
        )
        regional_cont = (
            cont_response.detach() if self.detach_features else cont_response
        )
        regional_features = self._regional_features(
            regional_cls, regional_cont
        )
        regional_latent = self.regional_encoder(regional_features)
        (
            alpha_region,
            regional_intervention_gate,
            regional_gate_logit,
            regional_diagnostics,
        ) = self._build_regional_calibration(
            regional_latent,
            regional_features,
            regional_cls,
            regional_cont,
            gate_conditioned,
            intervention_gate,
            gate_logit,
            alpha_global,
            height,
            width,
        )
        alpha_global_map = alpha_global[:, :, None].expand(
            -1, height, width
        )
        alpha_raw, alpha_final = self._compose_alpha(
            alpha_global_map,
            alpha_region,
            intervention_gate,
            regional_intervention_gate,
        )

        if self.apply_vl_tracking_only:
            if flag is None:
                raise ValueError('flag is required for VL-tracking-only ERC-v7')
            active_mask = flag.reshape(-1).eq(2)
        else:
            active_mask = torch.ones(
                batch_size, dtype=torch.bool, device=cls_response.device
            )
        alpha_final = torch.where(
            active_mask[:, None, None],
            alpha_final,
            torch.zeros_like(alpha_final),
        )
        if self.force_off:
            alpha_final = torch.zeros_like(alpha_final)

        # Global-only diagnostics use the same learned gate and active mask.
        alpha_global_effective = intervention_gate * alpha_global
        alpha_global_effective = torch.where(
            active_mask[:, None],
            alpha_global_effective,
            torch.zeros_like(alpha_global_effective),
        )
        if self.force_off:
            alpha_global_effective = torch.zeros_like(alpha_global_effective)

        log_ratio = (cls_safe.log() - cont_safe.log()).clamp(
            min=-self.log_ratio_clip, max=self.log_ratio_clip
        )
        alpha_flat = alpha_final.flatten(1)
        factor = torch.exp(alpha_flat * log_ratio)
        global_factor = torch.exp(alpha_global_effective * log_ratio)

        # Use the original, unclamped responses here.  This makes alpha=0
        # exactly equal to UVLTrack's official C*T fusion element-by-element.
        base_score = cls_flat * cont_flat
        calibrated_score = base_score * factor
        global_calibrated_score = base_score * global_factor
        windowed_base_score = base_score * self.hann_window

        output = {
            'features': global_features,
            'cls_response': cls_flat,
            'cont_response': cont_flat,
            'windowed_cls_score': cls_flat * self.hann_window,
            'windowed_cont_score': cont_flat * self.hann_window,
            'alpha': alpha_flat,
            'raw_alpha': alpha_raw.flatten(1),
            'z_alpha': z_global,
            'reliability': reliability_base,
            'reliability_gate': intervention_gate,
            'reliability_cls': reliability_cls,
            'reliability_cont': reliability_cont,
            'reliability_base': reliability_base,
            'intervention_gate': intervention_gate,
            'gate_logit': gate_logit,
            'regional_intervention_gate': regional_intervention_gate,
            'regional_gate_logit': regional_gate_logit,
            'gate_decision_features': gate_decision_features,
            'delta': alpha_region.flatten(1),
            'log_ratio': log_ratio,
            'factor': factor,
            'base_score': base_score,
            'windowed_base_score': windowed_base_score,
            'calibrated_score': calibrated_score,
            'windowed_calibrated_score': windowed_base_score * factor,
            'global_calibrated_score': global_calibrated_score,
            'windowed_global_calibrated_score': (
                windowed_base_score * global_factor
            ),
            'alpha_global': alpha_global,
            'alpha_region': alpha_region,
            'alpha_final': alpha_final,
            'regional_latent': regional_latent,
            'active_mask': active_mask,
            'inference_mode': self.inference_mode,
        }
        output.update(regional_diagnostics)
        return output

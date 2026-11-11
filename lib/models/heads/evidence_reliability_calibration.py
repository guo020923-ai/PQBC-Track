import math

import torch
from torch import nn


class EvidenceReliabilityCalibration(nn.Module):
    """Frame-level reliability estimation and dual-response calibration.

    The branch-gate design predicts classification, contrastive, and baseline
    reliability together with an intervention gate and a bounded direction
    residual.  Its output layer is zero-initialized, so the initial calibration
    factor is exactly one and UVLTrack's original fusion is recovered.
    """

    NUM_FEATURES = 13

    def __init__(self, feat_sz, hidden_dim=64, drop_rate=0.0, max_alpha=0.5,
                 eps=1e-6, log_ratio_clip=8.0, detach_features=True,
                 reliability_gated_alpha=False,
                 detach_reliability_gate=True,
                 reliability_gate_floor=0.0,
                 apply_vl_tracking_only=False,
                 design='legacy', direction_scale=3.0, delta_scale=0.5,
                 gate_prior=0.05,
                 detach_reliability_for_alpha=True,
                 force_off=False):
        super().__init__()
        self.feat_sz = feat_sz
        self.max_alpha = max_alpha
        self.eps = eps
        self.log_ratio_clip = log_ratio_clip
        self.detach_features = detach_features
        self.reliability_gated_alpha = reliability_gated_alpha
        self.detach_reliability_gate = detach_reliability_gate
        if not 0.0 <= reliability_gate_floor <= 1.0:
            raise ValueError("reliability_gate_floor must be in [0, 1]")
        self.reliability_gate_floor = reliability_gate_floor
        self.apply_vl_tracking_only = apply_vl_tracking_only
        self.design = design.lower()
        if self.design not in ('legacy', 'branch_gate', 'free_gate'):
            raise ValueError("Unsupported ERC design: %s" % design)
        self.direction_scale = direction_scale
        self.delta_scale = delta_scale
        if not 0.0 < gate_prior < 1.0:
            raise ValueError("gate_prior must be in (0, 1)")
        self.gate_prior = gate_prior
        self.detach_reliability_for_alpha = detach_reliability_for_alpha
        self.force_off = force_off
        window_1d = torch.hann_window(feat_sz, periodic=False)
        self.register_buffer(
            'hann_window',
            torch.outer(window_1d, window_1d).flatten().unsqueeze(0),
        )

        if self.design == 'free_gate':
            # v6 uses a two-stage predictor.  Reliability remains an explicit,
            # interpretable auxiliary prediction, while gate/alpha can use the
            # reliability state without hard-coding alpha's direction.
            self.encoder = nn.Sequential(
                nn.Linear(self.NUM_FEATURES, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(drop_rate),
            )
            self.reliability_head = nn.Linear(hidden_dim, 3)
            alpha_input_dim = hidden_dim + 5
            self.gate_alpha_predictor = nn.Sequential(
                nn.Linear(alpha_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(hidden_dim, 2),
            )
            nn.init.zeros_(self.reliability_head.weight)
            nn.init.zeros_(self.reliability_head.bias)
            nn.init.zeros_(self.gate_alpha_predictor[-1].weight)
            nn.init.zeros_(self.gate_alpha_predictor[-1].bias)
            self.gate_alpha_predictor[-1].bias.data[0] = math.log(
                gate_prior / (1.0 - gate_prior)
            )
        else:
            output_dim = 5 if self.design == 'branch_gate' else 2
            self.predictor = nn.Sequential(
                nn.Linear(self.NUM_FEATURES, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(hidden_dim, output_dim),
            )
            nn.init.zeros_(self.predictor[-1].weight)
            nn.init.zeros_(self.predictor[-1].bias)

    def _distribution(self, response):
        response = response.clamp_min(self.eps)
        return response / response.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def _response_statistics(self, response):
        probability = self._distribution(response)
        entropy = -(probability * probability.log()).sum(dim=-1)
        entropy = entropy / math.log(response.shape[-1])

        top_values, top_indices = torch.topk(response, k=2, dim=-1)
        peak = top_values[:, 0]
        relative_margin = (top_values[:, 0] - top_values[:, 1]) / (
            top_values[:, 0].abs() + self.eps
        )
        return probability, entropy, peak, relative_margin, top_indices[:, 0]

    def _extract_features(self, cls_response, cont_response):
        # UVLTrack's official test-time fusion includes the Hann spatial prior.
        fused_response = cls_response * cont_response * self.hann_window

        cls_prob, cls_entropy, cls_peak, cls_margin, cls_index = \
            self._response_statistics(cls_response)
        cont_prob, cont_entropy, cont_peak, cont_margin, cont_index = \
            self._response_statistics(cont_response)
        _, fused_entropy, fused_peak, fused_margin, _ = \
            self._response_statistics(fused_response)

        cls_row = torch.div(cls_index, self.feat_sz, rounding_mode='floor')
        cls_col = cls_index % self.feat_sz
        cont_row = torch.div(cont_index, self.feat_sz, rounding_mode='floor')
        cont_col = cont_index % self.feat_sz
        peak_distance = torch.sqrt(
            (cls_row - cont_row).float().square()
            + (cls_col - cont_col).float().square()
        )
        peak_distance = peak_distance / (
            math.sqrt(2.0) * max(self.feat_sz - 1, 1)
        )

        midpoint = 0.5 * (cls_prob + cont_prob)
        js_divergence = 0.5 * (
            cls_prob * (cls_prob.log() - midpoint.log())
        ).sum(dim=-1)
        js_divergence += 0.5 * (
            cont_prob * (cont_prob.log() - midpoint.log())
        ).sum(dim=-1)
        js_divergence = js_divergence / math.log(2.0)

        total_variation = 0.5 * (cls_prob - cont_prob).abs().sum(dim=-1)
        fused_mean = fused_response.mean(dim=-1)
        fused_concentration = (fused_peak - fused_mean) / (
            fused_peak.abs() + self.eps
        )

        features = torch.stack([
            cls_entropy,
            cont_entropy,
            fused_entropy,
            cls_peak,
            cont_peak,
            fused_peak,
            cls_margin,
            cont_margin,
            fused_margin,
            peak_distance,
            js_divergence,
            total_variation,
            fused_concentration,
        ], dim=-1)
        return features

    def forward(self, cls_response, cont_response, flag=None):
        cls_response = cls_response.flatten(1).clamp_min(self.eps)
        cont_response = cont_response.flatten(1).clamp_min(self.eps)

        features = self._extract_features(cls_response, cont_response)
        predictor_input = features.detach() if self.detach_features else features
        if self.design == 'free_gate':
            encoded = self.encoder(predictor_input)
            reliability_logits = self.reliability_head(encoded)
            reliability_cls = torch.sigmoid(reliability_logits[:, 0:1])
            reliability_cont = torch.sigmoid(reliability_logits[:, 1:2])
            reliability_base = torch.sigmoid(reliability_logits[:, 2:3])
            reliability_state = torch.cat([
                reliability_cls,
                reliability_cont,
                reliability_base,
                reliability_cls - reliability_cont,
                (reliability_cls - reliability_cont).abs(),
            ], dim=-1)
            if self.detach_reliability_for_alpha:
                reliability_state = reliability_state.detach()
            gate_alpha_prediction = self.gate_alpha_predictor(
                torch.cat([encoded, reliability_state], dim=-1)
            )
            intervention_gate = torch.sigmoid(
                gate_alpha_prediction[:, 0:1]
            )
            z_alpha = gate_alpha_prediction[:, 1:2]
            raw_alpha = self.max_alpha * torch.tanh(z_alpha)
            alpha = intervention_gate * raw_alpha
            reliability = reliability_base
            reliability_gate = intervention_gate
            delta = torch.zeros_like(alpha)
        else:
            prediction = self.predictor(predictor_input)
            z_alpha = torch.zeros_like(prediction[:, :1])

        if self.design == 'branch_gate':
            reliability_cls = torch.sigmoid(prediction[:, 0:1])
            reliability_cont = torch.sigmoid(prediction[:, 1:2])
            reliability_base = torch.sigmoid(prediction[:, 2:3])
            intervention_gate = torch.sigmoid(prediction[:, 3:4])
            delta = self.delta_scale * torch.tanh(prediction[:, 4:5])
            direction = (
                self.direction_scale * (reliability_cls - reliability_cont)
                + delta
            )
            raw_alpha = self.max_alpha * torch.tanh(direction)
            alpha = intervention_gate * raw_alpha
            # Backward-compatible aliases used by existing diagnostics.
            reliability = reliability_base
            reliability_gate = intervention_gate
        elif self.design == 'legacy':
            raw_alpha = self.max_alpha * torch.tanh(prediction[:, :1])
            z_alpha = prediction[:, :1]
            reliability = torch.sigmoid(prediction[:, 1:2])
            reliability_gate = self.reliability_gate_floor + (
                1.0 - self.reliability_gate_floor
            ) * (1.0 - reliability)
            if self.detach_reliability_gate:
                reliability_gate = reliability_gate.detach()
            alpha = (
                reliability_gate * raw_alpha
                if self.reliability_gated_alpha
                else raw_alpha
            )
            reliability_cls = reliability
            reliability_cont = reliability
            reliability_base = reliability
            intervention_gate = reliability_gate
            delta = torch.zeros_like(alpha)
        if self.apply_vl_tracking_only:
            if flag is None:
                raise ValueError("flag is required when ERC is VL-tracking-only")
            # Official UVLTrack flag semantics:
            # 0 = BBOX tracking, 1 = NL grounding initialization,
            # 2 = vision-language tracking (used after NL initialization and
            #     throughout the NLBBOX protocol).
            active_mask = flag.reshape(-1).eq(2)
            alpha = torch.where(active_mask[:, None], alpha, torch.zeros_like(alpha))
        else:
            active_mask = torch.ones(
                alpha.shape[0], dtype=torch.bool, device=alpha.device
            )
        if self.force_off:
            # Strict ablation path: factor becomes exactly one and the official
            # Hann-windowed C*T score is used without ERC calibration.
            alpha = torch.zeros_like(alpha)
        log_ratio = (
            cls_response.log() - cont_response.log()
        ).clamp(min=-self.log_ratio_clip, max=self.log_ratio_clip)
        factor = torch.exp(alpha * log_ratio)
        base_score = cls_response * cont_response
        windowed_base_score = base_score * self.hann_window
        calibrated_score = base_score * factor
        windowed_calibrated_score = windowed_base_score * factor

        return {
            'features': features,
            'cls_response': cls_response,
            'cont_response': cont_response,
            'windowed_cls_score': cls_response * self.hann_window,
            'windowed_cont_score': cont_response * self.hann_window,
            'alpha': alpha,
            'raw_alpha': raw_alpha,
            'z_alpha': z_alpha,
            'reliability': reliability,
            'reliability_gate': reliability_gate,
            'reliability_cls': reliability_cls,
            'reliability_cont': reliability_cont,
            'reliability_base': reliability_base,
            'intervention_gate': intervention_gate,
            'delta': delta,
            'log_ratio': log_ratio,
            'factor': factor,
            'base_score': base_score,
            'windowed_base_score': windowed_base_score,
            'calibrated_score': calibrated_score,
            'windowed_calibrated_score': windowed_calibrated_score,
            'active_mask': active_mask,
        }

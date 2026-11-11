"""Branch-aligned dense evidence reliability calibration (FER-v4).

FER-v3 estimated visual-reference versus language-reference reliability but
used the result to calibrate UVLTrack's native classification (C) and
contrastive (T) responses.  Those pairs are not interchangeable.  FER-v4
removes that semantic/action mismatch:

    p_C, p_T = softmax(branch_reliability_logits)
    direction = p_C - p_T
    g(x) = sigmoid(spatial_strength_logits(x))
    alpha(x) = alpha_max * direction * g(x)
    S_erc(x) = C(x) * T(x) * exp(alpha(x) * D(x))
    D(x) = center(log(C(x)) - log(T(x))).

The reliability difference is the only source of the correction sign.  The
spatial branch is non-negative, so it can decide where and how strongly to
intervene but cannot silently reverse which native response is trusted.  The
zero-initialized branch logits give p_C == p_T and therefore reproduce the
native UVLTrack fusion exactly before training.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class BranchAlignedDenseResponseERCV4(nn.Module):
    DESIGN = 'branch_aligned_dense_response_erc_v4'
    NUM_INPUT_CHANNELS = 8

    def __init__(
        self,
        feat_sz,
        dim,
        hidden=64,
        max_alpha=0.20,
        d_max=4.0,
        eps=1e-6,
        reliability_temperature=1.0,
        response_temperature=1.0,
        spatial_prior=0.5,
        detach_features=True,
        apply_vl_tracking_only=True,
        inference_mode='FULL_ERC',
        force_off=False,
    ):
        super().__init__()
        if min(int(feat_sz), int(dim), int(hidden)) <= 0:
            raise ValueError('FER-v4 dimensions must be positive')
        if float(max_alpha) <= 0.0 or float(d_max) <= 0.0:
            raise ValueError('FER-v4 bounds must be positive')
        if min(float(reliability_temperature), float(response_temperature)) <= 0.0:
            raise ValueError('FER-v4 temperatures must be positive')
        if not 0.0 < float(spatial_prior) < 1.0:
            raise ValueError('FER-v4 spatial_prior must be in (0, 1)')

        self.feat_sz = int(feat_sz)
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.max_alpha = float(max_alpha)
        self.d_max = float(d_max)
        self.eps = float(eps)
        self.reliability_temperature = float(reliability_temperature)
        self.response_temperature = float(response_temperature)
        self.spatial_prior = float(spatial_prior)
        self.detach_features = bool(detach_features)
        self.apply_vl_tracking_only = bool(apply_vl_tracking_only)
        self.force_off = bool(force_off)
        self.design = self.DESIGN

        window = torch.hann_window(self.feat_sz, periodic=False)
        self.register_buffer(
            'hann_window', torch.outer(window, window).flatten().unsqueeze(0))

        # These logits have a literal C/T meaning.  Zero initialization makes
        # their difference zero and hence makes the whole calibration an exact
        # identity, independently of the spatial-strength prior.
        self.reliability_head = nn.Sequential(
            nn.Linear(8, hidden), nn.GELU(), nn.Linear(hidden, 2))
        spatial_hidden = max(hidden // 2, 16)
        self.encoder = nn.Sequential(
            nn.Conv2d(
                self.NUM_INPUT_CHANNELS, spatial_hidden,
                kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(
                spatial_hidden, spatial_hidden,
                kernel_size=3, padding=1),
            nn.GELU(),
        )
        # This branch is deliberately unsigned.  It cannot reverse the
        # reliability direction chosen above.
        self.spatial_strength_head = nn.Conv2d(
            spatial_hidden, 1, kernel_size=1)

        nn.init.zeros_(self.reliability_head[-1].weight)
        nn.init.zeros_(self.reliability_head[-1].bias)
        nn.init.zeros_(self.spatial_strength_head.weight)
        nn.init.constant_(
            self.spatial_strength_head.bias,
            math.log(self.spatial_prior / (1.0 - self.spatial_prior)),
        )
        self.set_inference_mode(inference_mode)

    @staticmethod
    def _normalize_mode(mode):
        aliases = {
            'OFF': 'ERC_OFF', 'ERC_OFF': 'ERC_OFF',
            'FULL': 'FULL_ERC', 'FULL_ERC': 'FULL_ERC',
        }
        normalized = str(mode).upper()
        if normalized not in aliases:
            raise ValueError('FER-v4 supports ERC_OFF/FULL_ERC only')
        return aliases[normalized]

    def set_inference_mode(self, mode):
        self.inference_mode = self._normalize_mode(mode)

    @staticmethod
    def _one_reference(reference, batch, dim, name):
        if reference.ndim == 3:
            if reference.shape[1] != 1:
                raise ValueError('Expected one ' + name + ' reference token')
            reference = reference[:, 0]
        if reference.shape != (batch, dim):
            raise ValueError(name + ' reference shape mismatch')
        return reference

    @staticmethod
    def _standardize(response):
        centered = response - response.mean(-1, keepdim=True)
        scale = response.std(
            -1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return centered / scale

    @staticmethod
    def _entropy(probability):
        scale = math.log(max(probability.shape[-1], 2))
        return -(
            probability * probability.clamp_min(1e-12).log()
        ).sum(-1) / scale

    @staticmethod
    def _peak_margin(response):
        top = response.topk(min(2, response.shape[-1]), dim=-1).values
        return top[:, 0], top[:, 0] - top[:, -1]

    def _stable_response_channels(self, cls_response, cont_response):
        batch, height, width = cls_response.shape
        cls_safe = cls_response.clamp_min(self.eps)
        cont_safe = cont_response.clamp_min(self.eps)
        window = self.hann_window.reshape(1, height, width).to(
            cls_response.dtype)
        base = cls_safe * cont_safe
        windowed_cls = cls_safe * window
        windowed_cont = cont_safe * window
        windowed_base = base * window
        raw_log_c = cls_safe.log()
        raw_log_t = cont_safe.log()
        raw_log_s = (windowed_base + self.eps).log()
        difference = raw_log_c - raw_log_t
        difference = difference - difference.flatten(1).mean(-1)[:, None, None]
        difference = difference.clamp(-self.d_max, self.d_max)

        def bounded_log(value, lower):
            return value.clamp(lower, 0.0) / abs(lower)

        return {
            'log_c': bounded_log(raw_log_c, -8.0),
            'log_t': bounded_log(raw_log_t, -16.0),
            'log_s': bounded_log(raw_log_s, -16.0),
            'difference': difference,
            'base': base,
            'windowed_cls': windowed_cls,
            'windowed_cont': windowed_cont,
            'windowed_base': windowed_base,
            'window': window,
        }

    def _branch_statistics(self, stable):
        # C and T have different native scales.  Standardizing their log maps
        # makes confidence-shape statistics comparable without relabelling
        # visual/language references as native response branches.
        cls_log = self._standardize(
            (stable['windowed_cls'] + self.eps).log().flatten(1))
        cont_log = self._standardize(
            (stable['windowed_cont'] + self.eps).log().flatten(1))
        cls_probability = F.softmax(
            cls_log / self.response_temperature, dim=-1)
        cont_probability = F.softmax(
            cont_log / self.response_temperature, dim=-1)
        cls_peak, cls_margin = self._peak_margin(cls_log)
        cont_peak, cont_margin = self._peak_margin(cont_log)
        agreement = F.cosine_similarity(
            cls_probability, cont_probability, dim=-1)
        disagreement = (
            cls_probability - cont_probability).abs().mean(-1)
        return torch.stack([
            cls_peak, cls_margin, self._entropy(cls_probability),
            cont_peak, cont_margin, self._entropy(cont_probability),
            agreement, agreement - disagreement,
        ], dim=-1)

    def forward(
        self, cls_response, cont_response, search,
        visual_reference, language_reference, flag,
    ):
        if cls_response.ndim != 3 or cls_response.shape != cont_response.shape:
            raise ValueError('FER-v4 expects matching C/T maps [B,H,W]')
        batch, height, width = cls_response.shape
        if (height, width) != (self.feat_sz, self.feat_sz):
            raise ValueError('FER-v4 response size mismatch')
        if search.ndim != 3 or search.shape[:2] != (
            batch, height * width
        ) or search.shape[-1] != self.dim:
            raise ValueError('FER-v4 search token shape mismatch')

        visual_reference = self._one_reference(
            visual_reference, batch, self.dim, 'visual')
        language_reference = self._one_reference(
            language_reference, batch, self.dim, 'language')
        search_evidence = search.detach() if self.detach_features else search
        visual_evidence = (
            visual_reference.detach() if self.detach_features
            else visual_reference)
        language_evidence = (
            language_reference.detach() if self.detach_features
            else language_reference)

        search_unit = F.normalize(search_evidence, dim=-1)
        visual_response = self._standardize((
            search_unit * F.normalize(visual_evidence, dim=-1)[:, None]
        ).sum(-1))
        language_response = self._standardize((
            search_unit * F.normalize(language_evidence, dim=-1)[:, None]
        ).sum(-1))

        stable = self._stable_response_channels(cls_response, cont_response)
        branch_statistics = self._branch_statistics(stable)
        reliability_input = (
            branch_statistics.detach()
            if self.detach_features else branch_statistics)
        reliability_logits = self.reliability_head(reliability_input)
        reliability = F.softmax(
            reliability_logits / self.reliability_temperature, dim=-1)
        reliability_direction = reliability[:, :1] - reliability[:, 1:]

        # Visual/language maps remain useful VL context, but no longer define
        # or bypass the C/T reliability direction.
        reference_context = 0.5 * (visual_response + language_response)
        spatial_input = torch.stack([
            stable['log_c'], stable['log_t'], stable['log_s'],
            stable['difference'] / self.d_max,
            visual_response.reshape(batch, height, width).clamp(-4.0, 4.0) / 4.0,
            language_response.reshape(batch, height, width).clamp(-4.0, 4.0) / 4.0,
            reference_context.reshape(batch, height, width).clamp(-4.0, 4.0) / 4.0,
            stable['window'].expand(batch, -1, -1),
        ], dim=1)
        encoder_input = (
            spatial_input.detach() if self.detach_features else spatial_input)
        spatial_features = self.encoder(encoder_input)
        spatial_strength_logits = self.spatial_strength_head(
            spatial_features).squeeze(1)
        spatial_strength = torch.sigmoid(spatial_strength_logits)
        predicted_alpha = (
            self.max_alpha
            * reliability_direction[:, :, None]
            * spatial_strength
        )

        if self.apply_vl_tracking_only:
            active = flag.reshape(batch).eq(2)
        else:
            active = torch.ones(
                batch, dtype=torch.bool, device=cls_response.device)
        enabled = active & (not self.force_off) & (
            self.inference_mode == 'FULL_ERC')
        effective_alpha = torch.where(
            enabled[:, None, None], predicted_alpha,
            torch.zeros_like(predicted_alpha))
        correction_basis = (
            stable['difference'].detach()
            if self.detach_features else stable['difference'])
        correction = effective_alpha * correction_basis
        factor = correction.exp().flatten(1)
        base_flat = stable['base'].flatten(1)
        windowed_base_flat = stable['windowed_base'].flatten(1)
        calibrated = base_flat * factor
        windowed_calibrated = windowed_base_flat * factor
        alpha_abs = effective_alpha.abs().flatten(1).mean(-1, keepdim=True)
        pooled_features = spatial_features.mean(dim=(-2, -1))

        return {
            'features': pooled_features,
            'cls_response': cls_response.flatten(1),
            'cont_response': cont_response.flatten(1),
            'windowed_cls_score': stable['windowed_cls'].flatten(1),
            'windowed_cont_score': stable['windowed_cont'].flatten(1),
            'alpha': effective_alpha.flatten(1),
            'raw_alpha': predicted_alpha.flatten(1),
            'z_alpha': reliability_logits[:, :1] - reliability_logits[:, 1:],
            'reliability': reliability.max(-1, keepdim=True).values,
            'reliability_gate': reliability_direction.abs(),
            'reliability_cls': reliability[:, :1],
            'reliability_cont': reliability[:, 1:],
            'reliability_base': 1.0 - reliability_direction.abs(),
            'intervention_gate': reliability_direction.abs(),
            'gate_logit': (
                reliability_logits[:, :1] - reliability_logits[:, 1:]).abs(),
            'regional_intervention_gate': (
                effective_alpha.abs().flatten(1) / self.max_alpha),
            'regional_gate_logit': spatial_strength_logits.flatten(1),
            'delta': correction.flatten(1),
            'log_ratio': stable['difference'].flatten(1),
            'factor': factor,
            'base_score': base_flat,
            'windowed_base_score': windowed_base_flat,
            'calibrated_score': calibrated,
            'windowed_calibrated_score': windowed_calibrated,
            'global_calibrated_score': base_flat,
            'windowed_global_calibrated_score': windowed_base_flat,
            'alpha_global': (
                self.max_alpha * reliability_direction),
            'alpha_region': spatial_strength,
            'alpha_final': effective_alpha,
            'active_mask': active,
            'v4_visual_response': visual_response,
            'v4_language_response': language_response,
            'v4_reference_context': reference_context,
            'v4_branch_statistics': branch_statistics,
            'v4_reliability_input': reliability_input,
            'v4_reliability_logits': reliability_logits,
            'v4_reliability': reliability,
            'v4_reliability_direction': reliability_direction,
            'v4_input_channels': spatial_input,
            'v4_encoder_input': encoder_input,
            'v4_spatial_features': spatial_features,
            'v4_spatial_strength_logits': spatial_strength_logits,
            'v4_spatial_strength': spatial_strength,
            'v4_predicted_alpha': predicted_alpha,
            'v4_effective_alpha': effective_alpha,
            'v4_difference_map': stable['difference'],
            'v4_correction_basis': correction_basis,
            'v4_correction_map': correction,
            'v4_alpha_abs': alpha_abs,
            'inference_mode': self.inference_mode,
        }

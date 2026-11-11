"""Dense response-level evidence reliability calibration (FER-v3).

FER-v3 keeps UVLTrack's candidate geometry completely unchanged.  It observes
the native classification/contrast responses together with aligned visual and
language reference responses, then predicts a bounded dense field that can
only rebalance the disagreement between C and T:

    S_erc = C * T * exp(alpha(x) * D(x)),
    D = center(log(C) - log(T)).

The visual/language reliability weights condition the dense field but do not
directly move boxes.  A zero-initialized output projection makes FULL_ERC
exactly identical to the official fusion before training.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ReferenceReliabilityDenseResponseERCV3(nn.Module):
    DESIGN = 'reference_reliability_dense_response_erc_v3'
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
        detach_features=True,
        apply_vl_tracking_only=True,
        inference_mode='FULL_ERC',
        force_off=False,
    ):
        super().__init__()
        if min(int(feat_sz), int(dim), int(hidden)) <= 0:
            raise ValueError('FER-v3 dimensions must be positive')
        if float(max_alpha) <= 0.0 or float(d_max) <= 0.0:
            raise ValueError('FER-v3 bounds must be positive')
        if min(float(reliability_temperature), float(response_temperature)) <= 0.0:
            raise ValueError('FER-v3 temperatures must be positive')

        self.feat_sz = int(feat_sz)
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.max_alpha = float(max_alpha)
        self.d_max = float(d_max)
        self.eps = float(eps)
        self.reliability_temperature = float(reliability_temperature)
        self.response_temperature = float(response_temperature)
        self.detach_features = bool(detach_features)
        self.apply_vl_tracking_only = bool(apply_vl_tracking_only)
        self.force_off = bool(force_off)
        self.design = self.DESIGN

        window = torch.hann_window(self.feat_sz, periodic=False)
        self.register_buffer(
            'hann_window', torch.outer(window, window).flatten().unsqueeze(0))

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
        self.alpha_head = nn.Conv2d(spatial_hidden, 1, kernel_size=1)

        nn.init.zeros_(self.reliability_head[-1].weight)
        nn.init.zeros_(self.reliability_head[-1].bias)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.alpha_head.bias)
        self.set_inference_mode(inference_mode)

    @staticmethod
    def _normalize_mode(mode):
        aliases = {
            'OFF': 'ERC_OFF', 'ERC_OFF': 'ERC_OFF',
            'FULL': 'FULL_ERC', 'FULL_ERC': 'FULL_ERC',
        }
        normalized = str(mode).upper()
        if normalized not in aliases:
            raise ValueError('FER-v3 supports ERC_OFF/FULL_ERC only')
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

    def _reference_statistics(
        self, visual_response, language_response,
        visual_reference, language_reference,
    ):
        visual_probability = F.softmax(
            visual_response / self.response_temperature, dim=-1)
        language_probability = F.softmax(
            language_response / self.response_temperature, dim=-1)
        visual_peak, visual_margin = self._peak_margin(visual_response)
        language_peak, language_margin = self._peak_margin(language_response)
        reference_agreement = (
            F.normalize(visual_reference, dim=-1)
            * F.normalize(language_reference, dim=-1)
        ).sum(-1)
        response_agreement = F.cosine_similarity(
            visual_probability, language_probability, dim=-1)
        response_disagreement = (
            visual_probability - language_probability).abs().mean(-1)
        return torch.stack([
            visual_peak, visual_margin, self._entropy(visual_probability),
            language_peak, language_margin,
            self._entropy(language_probability), reference_agreement,
            response_agreement - response_disagreement,
        ], dim=-1)

    def _stable_response_channels(self, cls_response, cont_response):
        batch, height, width = cls_response.shape
        cls_safe = cls_response.clamp_min(self.eps)
        cont_safe = cont_response.clamp_min(self.eps)
        window = self.hann_window.reshape(1, height, width).to(
            cls_response.dtype)
        base = cls_safe * cont_safe
        windowed_base = base * window
        log_c = cls_safe.log()
        log_t = cont_safe.log()
        log_s = (windowed_base + self.eps).log()
        difference = log_c - log_t
        difference = difference - difference.flatten(1).mean(-1)[:, None, None]
        difference = difference.clamp(-self.d_max, self.d_max)

        def bounded_log(value, lower):
            return value.clamp(lower, 0.0) / abs(lower)

        return {
            'log_c': bounded_log(log_c, -8.0),
            'log_t': bounded_log(log_t, -16.0),
            'log_s': bounded_log(log_s, -16.0),
            'difference': difference,
            'base': base,
            'windowed_base': windowed_base,
            'window': window,
        }

    def forward(
        self, cls_response, cont_response, search,
        visual_reference, language_reference, flag,
    ):
        if cls_response.ndim != 3 or cls_response.shape != cont_response.shape:
            raise ValueError('FER-v3 expects matching C/T maps [B,H,W]')
        batch, height, width = cls_response.shape
        if (height, width) != (self.feat_sz, self.feat_sz):
            raise ValueError('FER-v3 response size mismatch')
        if search.ndim != 3 or search.shape[:2] != (
            batch, height * width
        ) or search.shape[-1] != self.dim:
            raise ValueError('FER-v3 search token shape mismatch')

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
        statistics = self._reference_statistics(
            visual_response, language_response,
            visual_evidence, language_evidence)
        reliability_logits = self.reliability_head(statistics)
        reliability = F.softmax(
            reliability_logits / self.reliability_temperature, dim=-1)
        reference_response = (
            reliability[:, :1] * visual_response
            + reliability[:, 1:] * language_response)

        stable = self._stable_response_channels(cls_response, cont_response)
        spatial_input = torch.stack([
            stable['log_c'], stable['log_t'], stable['log_s'],
            stable['difference'] / self.d_max,
            visual_response.reshape(batch, height, width).clamp(-4.0, 4.0) / 4.0,
            language_response.reshape(batch, height, width).clamp(-4.0, 4.0) / 4.0,
            reference_response.reshape(batch, height, width).clamp(-4.0, 4.0) / 4.0,
            stable['window'].expand(batch, -1, -1),
        ], dim=1)
        encoder_input = spatial_input.detach() if self.detach_features else spatial_input
        spatial_features = self.encoder(encoder_input)
        alpha_logits = self.alpha_head(spatial_features).squeeze(1)
        predicted_alpha = self.max_alpha * torch.tanh(alpha_logits)

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
        reliability_strength = reliability.max(-1, keepdim=True).values
        pooled_features = spatial_features.mean(dim=(-2, -1))

        return {
            'features': pooled_features,
            'cls_response': cls_response.flatten(1),
            'cont_response': cont_response.flatten(1),
            'windowed_cls_score': (
                cls_response * stable['window']).flatten(1),
            'windowed_cont_score': (
                cont_response * stable['window']).flatten(1),
            'alpha': effective_alpha.flatten(1),
            'raw_alpha': predicted_alpha.flatten(1),
            'z_alpha': alpha_logits.flatten(1),
            'reliability': reliability_strength,
            'reliability_gate': alpha_abs / self.max_alpha,
            'reliability_cls': reliability[:, :1],
            'reliability_cont': reliability[:, 1:],
            'reliability_base': 1.0 - reliability_strength,
            'intervention_gate': alpha_abs / self.max_alpha,
            'gate_logit': alpha_logits.flatten(1).abs().mean(-1, keepdim=True),
            'regional_intervention_gate': effective_alpha.abs().flatten(1) / self.max_alpha,
            'regional_gate_logit': alpha_logits.flatten(1),
            'delta': correction.flatten(1),
            'log_ratio': stable['difference'].flatten(1),
            'factor': factor,
            'base_score': base_flat,
            'windowed_base_score': windowed_base_flat,
            'calibrated_score': calibrated,
            'windowed_calibrated_score': windowed_calibrated,
            'global_calibrated_score': base_flat,
            'windowed_global_calibrated_score': windowed_base_flat,
            'alpha_global': effective_alpha.flatten(1).mean(-1, keepdim=True),
            'alpha_region': effective_alpha,
            'alpha_final': effective_alpha,
            'active_mask': active,
            'v3_visual_response': visual_response,
            'v3_language_response': language_response,
            'v3_reference_response': reference_response,
            'v3_reference_statistics': statistics,
            'v3_reliability_logits': reliability_logits,
            'v3_reliability': reliability,
            'v3_input_channels': spatial_input,
            'v3_encoder_input': encoder_input,
            'v3_spatial_features': spatial_features,
            'v3_alpha_logits': alpha_logits,
            'v3_predicted_alpha': predicted_alpha,
            'v3_effective_alpha': effective_alpha,
            'v3_difference_map': stable['difference'],
            'v3_correction_basis': correction_basis,
            'v3_correction_map': correction,
            'inference_mode': self.inference_mode,
        }

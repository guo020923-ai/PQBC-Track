"""Branch-aligned tri-state multi-action evidence calibration (FER-v5).

FER-v5 makes the deployable response correction a single, identifiable
decision.  The network chooses one element of a fixed action set whose first
element is KEEP (zero), whose positive elements favour UVLTrack's native
classification response C, and whose negative elements favour its native
contrastive response T::

    L0(x) = log(Hann(x)) + log(C(x)) + log(T(x))
    D(x)  = clip(center(log(C(x)) - log(T(x))), -d_max, d_max)
    L(x)  = L0(x) + alpha * D(x)

There is deliberately no learned spatial gate.  ``D`` already varies over
space, while eliminating a free gate prevents the all-one spatial solution
observed in FER-v4.  During training a straight-through one-hot policy gives
the action logits gradients, but its forward value is the same hard argmax
used for evaluation.  With a zero-initialized action head and KEEP at index
zero, FULL_ERC is exactly the native fusion at initialization.
"""

import math
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class BranchAlignedTriStateMultiActionERCV5(nn.Module):
    """Hard KEEP/C/T response policy with two or more fixed magnitudes."""

    DESIGN = 'branch_aligned_tristate_multiaction_erc_v5'
    NUM_INPUT_CHANNELS = 8

    def __init__(
        self,
        feat_sz: int,
        dim: int,
        hidden: int = 64,
        action_alpha_values: Iterable[float] = (
            0.0, 0.10, 0.20, -0.10, -0.20,
        ),
        max_alpha: float = 0.20,
        d_max: float = 4.0,
        eps: float = 1e-6,
        action_temperature: float = 1.0,
        response_temperature: float = 1.0,
        detach_features: bool = True,
        apply_vl_tracking_only: bool = True,
        inference_mode: str = 'FULL_ERC',
        force_off: bool = False,
    ) -> None:
        super().__init__()
        if min(int(feat_sz), int(dim), int(hidden)) <= 0:
            raise ValueError('FER-v5 dimensions must be positive')
        if min(float(max_alpha), float(d_max), float(eps)) <= 0.0:
            raise ValueError('FER-v5 bounds and eps must be positive')
        if min(float(action_temperature), float(response_temperature)) <= 0:
            raise ValueError('FER-v5 temperatures must be positive')

        action_values = tuple(float(value) for value in action_alpha_values)
        if len(action_values) < 3:
            raise ValueError('FER-v5 needs KEEP and both C/T actions')
        if abs(action_values[0]) > 1e-12:
            raise ValueError('FER-v5 KEEP action must be index zero')
        zero_count = sum(abs(value) <= 1e-12 for value in action_values)
        if zero_count != 1:
            raise ValueError('FER-v5 action set must contain exactly one zero')
        if not any(value > 0.0 for value in action_values):
            raise ValueError('FER-v5 action set needs a C-favouring action')
        if not any(value < 0.0 for value in action_values):
            raise ValueError('FER-v5 action set needs a T-favouring action')
        if max(abs(value) for value in action_values) > float(max_alpha) + 1e-8:
            raise ValueError('FER-v5 action exceeds MAX_ALPHA')

        self.feat_sz = int(feat_sz)
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.max_alpha = float(max_alpha)
        self.d_max = float(d_max)
        self.eps = float(eps)
        self.action_temperature = float(action_temperature)
        self.response_temperature = float(response_temperature)
        self.detach_features = bool(detach_features)
        self.apply_vl_tracking_only = bool(apply_vl_tracking_only)
        self.force_off = bool(force_off)
        self.design = self.DESIGN

        self.register_buffer(
            'action_alpha_values', torch.tensor(action_values, dtype=torch.float32)
        )
        self.register_buffer(
            'action_class_ema',
            torch.full((len(action_values),), 1.0 / len(action_values)),
        )
        window = torch.hann_window(self.feat_sz, periodic=False)
        window_2d = torch.outer(window, window)
        self.register_buffer('hann_window_2d', window_2d[None])
        self.register_buffer('hann_window', window_2d.flatten()[None])

        self.encoder = nn.Sequential(
            nn.Conv2d(self.NUM_INPUT_CHANNELS, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )
        # Learned mean/max/std plus eight explicit branch-shape statistics.
        decision_dim = 3 * hidden + 8
        self.action_head = nn.Sequential(
            nn.Linear(decision_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(action_values)),
        )
        # Equal zero logits make torch.argmax select KEEP at index zero.
        nn.init.zeros_(self.action_head[-1].weight)
        nn.init.zeros_(self.action_head[-1].bias)
        self.set_inference_mode(inference_mode)

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        aliases = {
            'OFF': 'ERC_OFF',
            'ERC_OFF': 'ERC_OFF',
            'FULL': 'FULL_ERC',
            'FULL_ERC': 'FULL_ERC',
        }
        normalized = str(mode).upper()
        if normalized not in aliases:
            raise ValueError('FER-v5 supports ERC_OFF/FULL_ERC only')
        return aliases[normalized]

    def set_inference_mode(self, mode: str) -> None:
        self.inference_mode = self._normalize_mode(mode)

    @staticmethod
    def _one_reference(reference, batch: int, dim: int, name: str):
        if reference.ndim == 3:
            if reference.shape[1] != 1:
                raise ValueError('Expected one ' + name + ' reference token')
            reference = reference[:, 0]
        if reference.shape != (batch, dim):
            raise ValueError(name + ' reference shape mismatch')
        return reference

    @staticmethod
    def _masked_moments(values, support):
        support_float = support.to(values.dtype)
        count = support_float.sum(-1, keepdim=True).clamp_min(1.0)
        mean = (values * support_float).sum(-1, keepdim=True) / count
        variance = (
            (values - mean).square() * support_float
        ).sum(-1, keepdim=True) / count
        return mean, variance.clamp_min(1e-12).sqrt()

    def _standardize(self, response, support=None):
        flat = response.flatten(1)
        if support is None:
            mean = flat.mean(-1, keepdim=True)
            scale = flat.std(-1, keepdim=True, unbiased=False).clamp_min(1e-6)
            return (flat - mean) / scale
        support_flat = support.flatten(1).expand_as(flat)
        mean, scale = self._masked_moments(flat, support_flat)
        standardized = (flat - mean) / scale.clamp_min(1e-6)
        return torch.where(support_flat, standardized, torch.zeros_like(flat))

    @staticmethod
    def _entropy(probability):
        scale = math.log(max(probability.shape[-1], 2))
        return -(
            probability * probability.clamp_min(1e-12).log()
        ).sum(-1) / scale

    @staticmethod
    def _peak_margin(response, support):
        masked = response.masked_fill(~support, torch.finfo(response.dtype).min)
        top = masked.topk(min(2, response.shape[-1]), dim=-1).values
        return top[:, 0], top[:, 0] - top[:, -1]

    def _stable_response_channels(self, cls_response, cont_response):
        batch, height, width = cls_response.shape
        cls_safe = cls_response.clamp_min(self.eps)
        cont_safe = cont_response.clamp_min(self.eps)
        window = self.hann_window_2d.to(
            device=cls_response.device, dtype=cls_response.dtype
        ).expand(batch, height, width)
        support = window.gt(0.0)
        support_float = support.to(cls_response.dtype)
        base = cls_safe * cont_safe
        windowed_cls = cls_safe * window
        windowed_cont = cont_safe * window
        windowed_base = base * window
        raw_log_c = cls_safe.log()
        raw_log_t = cont_safe.log()
        raw_difference = raw_log_c - raw_log_t
        difference_mean, _ = self._masked_moments(
            raw_difference.flatten(1), support.flatten(1)
        )
        difference = raw_difference - difference_mean[:, :, None]
        difference = difference.clamp(-self.d_max, self.d_max)
        difference = difference * support_float

        def bounded_log(value, lower):
            return value.clamp(lower, 0.0) / abs(lower)

        log_s = (windowed_base + self.eps).log()
        return {
            'log_c': bounded_log(raw_log_c, -8.0) * support_float,
            'log_t': bounded_log(raw_log_t, -16.0) * support_float,
            'log_s': bounded_log(log_s, -16.0) * support_float,
            'difference': difference,
            'base': base,
            'windowed_cls': windowed_cls,
            'windowed_cont': windowed_cont,
            'windowed_base': windowed_base,
            'window': window,
            'support': support,
        }

    def _branch_statistics(self, stable):
        support = stable['support'].flatten(1)
        cls_log = self._standardize(
            (stable['windowed_cls'] + self.eps).log(), stable['support']
        )
        cont_log = self._standardize(
            (stable['windowed_cont'] + self.eps).log(), stable['support']
        )
        minimum = torch.finfo(cls_log.dtype).min
        cls_probability = F.softmax(
            (cls_log / self.response_temperature).masked_fill(~support, minimum),
            dim=-1,
        )
        cont_probability = F.softmax(
            (cont_log / self.response_temperature).masked_fill(~support, minimum),
            dim=-1,
        )
        cls_peak, cls_margin = self._peak_margin(cls_log, support)
        cont_peak, cont_margin = self._peak_margin(cont_log, support)
        agreement = F.cosine_similarity(cls_probability, cont_probability, dim=-1)
        disagreement = (
            (cls_probability - cont_probability).abs() * support
        ).sum(-1) / support.sum(-1).clamp_min(1)
        return torch.stack([
            cls_peak,
            cls_margin,
            self._entropy(cls_probability),
            cont_peak,
            cont_margin,
            self._entropy(cont_probability),
            agreement,
            agreement - disagreement,
        ], dim=-1)

    @staticmethod
    def _learned_statistics(features, support):
        support_float = support[:, None].to(features.dtype)
        count = support_float.sum((-2, -1)).clamp_min(1.0)
        mean = (features * support_float).sum((-2, -1)) / count
        centered = (features - mean[:, :, None, None]) * support_float
        std = (
            centered.square().sum((-2, -1)) / count
        ).clamp_min(1e-12).sqrt()
        minimum = torch.finfo(features.dtype).min
        maximum = features.masked_fill(~support[:, None], minimum).amax((-2, -1))
        return torch.cat([mean, maximum, std], dim=-1)

    def _semantic_probabilities(self, action_probability) -> Tuple:
        values = self.action_alpha_values.to(action_probability)
        keep = values.eq(0.0)
        favour_cls = values.gt(0.0)
        favour_cont = values.lt(0.0)
        p_keep = action_probability[:, keep].sum(-1, keepdim=True)
        p_cls = action_probability[:, favour_cls].sum(-1, keepdim=True)
        p_cont = action_probability[:, favour_cont].sum(-1, keepdim=True)
        return p_keep, p_cls, p_cont

    def forward(
        self,
        cls_response,
        cont_response,
        search,
        visual_reference,
        language_reference,
        flag,
    ):
        if cls_response.ndim != 3 or cls_response.shape != cont_response.shape:
            raise ValueError('FER-v5 expects matching C/T maps [B,H,W]')
        batch, height, width = cls_response.shape
        if (height, width) != (self.feat_sz, self.feat_sz):
            raise ValueError('FER-v5 response size mismatch')
        if search.ndim != 3 or search.shape != (
            batch, height * width, self.dim
        ):
            raise ValueError('FER-v5 search token shape mismatch')

        visual_reference = self._one_reference(
            visual_reference, batch, self.dim, 'visual'
        )
        language_reference = self._one_reference(
            language_reference, batch, self.dim, 'language'
        )
        search_evidence = search.detach() if self.detach_features else search
        visual_evidence = (
            visual_reference.detach() if self.detach_features else visual_reference
        )
        language_evidence = (
            language_reference.detach()
            if self.detach_features else language_reference
        )

        search_unit = F.normalize(search_evidence, dim=-1)
        visual_response = self._standardize((
            search_unit * F.normalize(visual_evidence, dim=-1)[:, None]
        ).sum(-1))
        language_response = self._standardize((
            search_unit * F.normalize(language_evidence, dim=-1)[:, None]
        ).sum(-1))

        stable = self._stable_response_channels(cls_response, cont_response)
        support = stable['support']
        support_float = support.to(cls_response.dtype)
        reference_context = 0.5 * (visual_response + language_response)
        spatial_input = torch.stack([
            stable['log_c'],
            stable['log_t'],
            stable['log_s'],
            stable['difference'] / self.d_max,
            visual_response.reshape(batch, height, width).clamp(-4.0, 4.0)
            / 4.0 * support_float,
            language_response.reshape(batch, height, width).clamp(-4.0, 4.0)
            / 4.0 * support_float,
            reference_context.reshape(batch, height, width).clamp(-4.0, 4.0)
            / 4.0 * support_float,
            stable['window'],
        ], dim=1)
        encoder_input = spatial_input.detach() if self.detach_features else spatial_input
        spatial_features = self.encoder(encoder_input)
        spatial_features = spatial_features * support_float[:, None]
        branch_statistics = self._branch_statistics(stable)
        branch_input = (
            branch_statistics.detach() if self.detach_features else branch_statistics
        )
        decision_features = torch.cat([
            self._learned_statistics(spatial_features, support),
            branch_input,
        ], dim=-1)
        action_logits = self.action_head(decision_features)
        action_probability = F.softmax(
            action_logits / self.action_temperature, dim=-1
        )
        raw_hard_action = action_logits.argmax(-1)
        hard_policy = F.one_hot(
            raw_hard_action, action_logits.shape[-1]
        ).to(action_probability.dtype)
        # Numerically hard in both train and eval; only the backward path differs.
        action_policy = (
            hard_policy + action_probability - action_probability.detach()
            if self.training else hard_policy
        )
        values = self.action_alpha_values.to(action_probability)
        predicted_alpha_scalar = (action_policy * values[None]).sum(-1)
        hard_alpha_scalar = (hard_policy * values[None]).sum(-1)
        soft_alpha_scalar = (action_probability * values[None]).sum(-1)

        if self.apply_vl_tracking_only:
            active = flag.reshape(batch).eq(2)
        else:
            active = torch.ones(batch, dtype=torch.bool, device=flag.device)
        enabled = active & (not self.force_off) & (
            self.inference_mode == 'FULL_ERC'
        )
        effective_alpha_scalar = torch.where(
            enabled, predicted_alpha_scalar, torch.zeros_like(predicted_alpha_scalar)
        )
        effective_hard_action = torch.where(
            enabled, raw_hard_action, torch.zeros_like(raw_hard_action)
        )
        effective_alpha = effective_alpha_scalar[:, None, None].expand(
            batch, height, width
        )
        correction_basis = (
            stable['difference'].detach()
            if self.detach_features else stable['difference']
        )
        correction = effective_alpha * correction_basis
        factor = correction.exp()
        base_flat = stable['base'].flatten(1)
        windowed_base_flat = stable['windowed_base'].flatten(1)
        calibrated = base_flat * factor.flatten(1)
        windowed_calibrated = windowed_base_flat * factor.flatten(1)

        p_keep, p_cls, p_cont = self._semantic_probabilities(action_probability)
        non_keep = 1.0 - p_keep
        cls_mask = values.gt(0.0)
        cont_mask = values.lt(0.0)
        nonkeep_mask = values.ne(0.0)
        cls_logit = torch.logsumexp(action_logits[:, cls_mask], dim=-1, keepdim=True)
        cont_logit = torch.logsumexp(
            action_logits[:, cont_mask], dim=-1, keepdim=True
        )
        gate_logit = (
            torch.logsumexp(action_logits[:, nonkeep_mask], dim=-1, keepdim=True)
            - action_logits[:, :1]
        )

        return {
            # Common adapter keys used by ModalityAdaptiveBoxHead.
            'features': decision_features,
            'cls_response': cls_response.flatten(1),
            'cont_response': cont_response.flatten(1),
            'windowed_cls_score': stable['windowed_cls'].flatten(1),
            'windowed_cont_score': stable['windowed_cont'].flatten(1),
            'alpha': effective_alpha.flatten(1),
            'raw_alpha': predicted_alpha_scalar[:, None],
            'z_alpha': cls_logit - cont_logit,
            'reliability': torch.cat([p_keep, p_cls, p_cont], -1).max(
                -1, keepdim=True
            ).values,
            'reliability_gate': non_keep,
            'reliability_cls': p_cls,
            'reliability_cont': p_cont,
            'reliability_base': p_keep,
            'intervention_gate': non_keep,
            'gate_logit': gate_logit,
            'regional_intervention_gate': non_keep.expand(-1, height * width),
            'regional_gate_logit': gate_logit.expand(-1, height * width),
            'delta': correction.flatten(1),
            'log_ratio': correction_basis.flatten(1),
            'factor': factor.flatten(1),
            'base_score': base_flat,
            'windowed_base_score': windowed_base_flat,
            'calibrated_score': calibrated,
            'windowed_calibrated_score': windowed_calibrated,
            'global_calibrated_score': calibrated,
            'windowed_global_calibrated_score': windowed_calibrated,
            'alpha_global': effective_alpha_scalar[:, None],
            'alpha_region': torch.ones_like(stable['difference']),
            'alpha_final': effective_alpha,
            'active_mask': active,
            # FER-v5 diagnostics and loss inputs.
            'v5_action_values': values,
            'v5_action_class_ema': self.action_class_ema,
            'v5_action_logits': action_logits,
            'v5_action_probability': action_probability,
            'v5_raw_hard_action': raw_hard_action,
            'v5_hard_action': effective_hard_action,
            'v5_hard_policy': hard_policy,
            'v5_action_policy': action_policy,
            'v5_predicted_alpha_scalar': predicted_alpha_scalar,
            'v5_hard_alpha_scalar': hard_alpha_scalar,
            'v5_soft_alpha_scalar': soft_alpha_scalar,
            'v5_effective_alpha_scalar': effective_alpha_scalar,
            'v5_effective_alpha': effective_alpha,
            'v5_difference_map': stable['difference'],
            'v5_correction_basis': correction_basis,
            'v5_correction_map': correction,
            'v5_support_mask': support,
            'v5_spatial_input': spatial_input,
            'v5_encoder_input': encoder_input,
            'v5_spatial_features': spatial_features,
            'v5_branch_statistics': branch_statistics,
            'v5_decision_features': decision_features,
            'v5_semantic_probability': torch.cat([p_keep, p_cls, p_cont], -1),
            'v5_permission_logit': gate_logit,
            'v5_direction_logits': torch.cat([cls_logit, cont_logit], -1),
            'v5_log_base': (
                stable['windowed_base'].clamp_min(self.eps).log()
            ).masked_fill(~support, torch.finfo(cls_response.dtype).min),
            'inference_mode': self.inference_mode,
        }

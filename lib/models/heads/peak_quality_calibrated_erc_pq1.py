"""Peak-quality calibrated evidence reliability (PQ-ERC).

PQ-ERC estimates how trustworthy the native classification (C) and
contrastive (T) branch peaks are.  It deliberately has no action catalogue,
no separately learned intervention gate, no candidate-IoU target and no
temporal state.  The predicted peak qualities determine both correction
direction and strength::

    alpha = alpha_max * tanh(logit(r_C) - logit(r_T))
            * disagreement(C, T) * max(r_C, r_T)
    S_erc = Hann * C * T * exp(alpha * D)

where D is the centred, bounded log response ratio.  The last reliability
layer is zero initialized, therefore r_C == r_T == 0.5 and alpha == 0 at
initialization, exactly recovering UVLTrack's native response fusion.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .branch_aligned_tristate_multiaction_erc_v5 import (
    BranchAlignedTriStateMultiActionERCV5,
)


class PeakQualityCalibratedERCPQ1(BranchAlignedTriStateMultiActionERCV5):
    """Response-only C/T peak-quality calibration for VL tracking."""

    DESIGN = 'peak_quality_calibrated_erc_pq1'
    NUM_INPUT_CHANNELS = 8

    def __init__(
        self, feat_sz, dim, hidden=64, magnitudes=None,
        max_alpha=0.40, d_max=4.0, eps=1e-6,
        action_temperature=1.0, response_temperature=1.0, detach_features=True,
        apply_vl_tracking_only=True, inference_mode='FULL_ERC',
        force_off=False,
    ):
        super().__init__(
            feat_sz=feat_sz,
            dim=dim,
            hidden=hidden,
            action_alpha_values=(0.0, max_alpha, -max_alpha),
            max_alpha=max_alpha,
            d_max=d_max,
            eps=eps,
            action_temperature=action_temperature,
            response_temperature=response_temperature,
            detach_features=detach_features,
            apply_vl_tracking_only=apply_vl_tracking_only,
            inference_mode=inference_mode,
            force_off=force_off,
        )
        # The parent supplies stable Hann-aware response processing.  PQ-ERC
        # does not use its discrete action head or action-state buffers.
        del self.action_head
        decision_dim = 5 * hidden + 18
        self.quality_head = nn.Sequential(
            nn.Linear(decision_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.quality_head[-1].weight)
        nn.init.zeros_(self.quality_head[-1].bias)
        self.design = self.DESIGN

    @staticmethod
    def _gather_flat(values, indices):
        return values.gather(1, indices[:, None]).squeeze(1)

    @staticmethod
    def _gather_features(features, indices):
        flat = features.flatten(2).transpose(1, 2)
        gather_index = indices[:, None, None].expand(-1, 1, flat.shape[-1])
        return flat.gather(1, gather_index).squeeze(1)

    def _branch_disagreement(self, stable):
        support = stable['support'].flatten(1)
        minimum = torch.finfo(stable['base'].dtype).min
        cls_log = self._standardize(
            (stable['windowed_cls'] + self.eps).log(), stable['support'])
        cont_log = self._standardize(
            (stable['windowed_cont'] + self.eps).log(), stable['support'])
        cls_probability = F.softmax(
            (cls_log / self.response_temperature).masked_fill(
                ~support, minimum), dim=-1)
        cont_probability = F.softmax(
            (cont_log / self.response_temperature).masked_fill(
                ~support, minimum), dim=-1)
        mixture = 0.5 * (cls_probability + cont_probability)
        js = 0.5 * (
            (cls_probability * (
                cls_probability.clamp_min(self.eps).log()
                - mixture.clamp_min(self.eps).log())).sum(-1)
            + (cont_probability * (
                cont_probability.clamp_min(self.eps).log()
                - mixture.clamp_min(self.eps).log())).sum(-1)
        ) / math.log(2.0)

        cls_index = stable['windowed_cls'].flatten(1).argmax(-1)
        cont_index = stable['windowed_cont'].flatten(1).argmax(-1)
        cls_y = torch.div(cls_index, self.feat_sz, rounding_mode='floor')
        cls_x = cls_index.remainder(self.feat_sz)
        cont_y = torch.div(cont_index, self.feat_sz, rounding_mode='floor')
        cont_x = cont_index.remainder(self.feat_sz)
        normalizer = math.sqrt(2.0) * max(self.feat_sz - 1, 1)
        peak_distance = torch.sqrt(
            (cls_x - cont_x).float().square()
            + (cls_y - cont_y).float().square()) / normalizer
        disagreement = (0.5 * (js + peak_distance)).clamp(0.0, 1.0)
        return {
            'cls_probability': cls_probability,
            'cont_probability': cont_probability,
            'cls_index': cls_index,
            'cont_index': cont_index,
            'cls_xy': torch.stack([cls_x, cls_y], dim=-1),
            'cont_xy': torch.stack([cont_x, cont_y], dim=-1),
            'js': js.clamp(0.0, 1.0),
            'peak_distance': peak_distance.clamp(0.0, 1.0),
            'disagreement': disagreement,
        }

    def forward(
        self, cls_response, cont_response, search,
        visual_reference, language_reference, flag,
    ):
        if cls_response.ndim != 3 or cls_response.shape != cont_response.shape:
            raise ValueError('PQ-ERC expects matching C/T maps [B,H,W]')
        batch, height, width = cls_response.shape
        if (height, width) != (self.feat_sz, self.feat_sz):
            raise ValueError('PQ-ERC response size mismatch')
        if search.shape != (batch, height * width, self.dim):
            raise ValueError('PQ-ERC search token shape mismatch')

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
        support = stable['support']
        support_float = support.to(cls_response.dtype)
        reference_context = 0.5 * (visual_response + language_response)
        spatial_input = torch.stack([
            stable['log_c'],
            stable['log_t'],
            stable['log_s'],
            stable['difference'] / self.d_max,
            visual_response.reshape(batch, height, width).clamp(-4, 4) / 4,
            language_response.reshape(batch, height, width).clamp(-4, 4) / 4,
            reference_context.reshape(batch, height, width).clamp(-4, 4) / 4,
            stable['window'],
        ], dim=1) * support_float[:, None]
        encoder_input = (
            spatial_input.detach() if self.detach_features else spatial_input)
        spatial_features = self.encoder(encoder_input) * support_float[:, None]
        branch = self._branch_disagreement(stable)
        cls_index = branch['cls_index']
        cont_index = branch['cont_index']
        cls_flat = cls_response.flatten(1)
        cont_flat = cont_response.flatten(1)
        cross_peak_scores = torch.stack([
            self._gather_flat(cls_flat, cls_index),
            self._gather_flat(cont_flat, cls_index),
            self._gather_flat(cls_flat, cont_index),
            self._gather_flat(cont_flat, cont_index),
        ], dim=-1)
        coordinate_scale = max(self.feat_sz - 1, 1)
        peak_coordinates = torch.cat([
            branch['cls_xy'].to(cls_response.dtype) / coordinate_scale,
            branch['cont_xy'].to(cls_response.dtype) / coordinate_scale,
        ], dim=-1)
        disagreement_features = torch.stack([
            branch['peak_distance'], branch['js']], dim=-1)
        branch_statistics = self._branch_statistics(stable)
        decision_features = torch.cat([
            self._learned_statistics(spatial_features, support),
            self._gather_features(spatial_features, cls_index),
            self._gather_features(spatial_features, cont_index),
            branch_statistics.detach() if self.detach_features else branch_statistics,
            cross_peak_scores.detach() if self.detach_features else cross_peak_scores,
            peak_coordinates,
            disagreement_features,
        ], dim=-1)

        quality_logits = self.quality_head(decision_features)
        reliability = torch.sigmoid(quality_logits)
        reliability_gap = quality_logits[:, 0] - quality_logits[:, 1]
        confidence = reliability.max(-1).values
        raw_alpha = (
            self.max_alpha * torch.tanh(reliability_gap)
            * branch['disagreement'] * confidence)

        if self.apply_vl_tracking_only:
            active = flag.reshape(batch).eq(2)
        else:
            active = torch.ones(batch, dtype=torch.bool, device=flag.device)
        runtime_enabled = (
            not self.force_off and self.inference_mode == 'FULL_ERC')
        enabled = active if runtime_enabled else torch.zeros_like(active)
        effective_alpha = torch.where(
            enabled, raw_alpha, torch.zeros_like(raw_alpha))
        difference = (
            stable['difference'].detach()
            if self.detach_features else stable['difference'])
        raw_correction = raw_alpha[:, None, None] * difference
        correction = torch.where(
            enabled[:, None, None], raw_correction,
            torch.zeros_like(raw_correction))
        factor = correction.exp()
        base = stable['base'].flatten(1)
        windowed_base = stable['windowed_base'].flatten(1)
        calibrated = base * factor.flatten(1)
        windowed_calibrated = windowed_base * factor.flatten(1)
        intervention_strength = (
            effective_alpha.abs() / max(self.max_alpha, self.eps))

        return {
            'features': decision_features,
            'cls_response': cls_flat,
            'cont_response': cont_flat,
            'windowed_cls_score': stable['windowed_cls'].flatten(1),
            'windowed_cont_score': stable['windowed_cont'].flatten(1),
            'alpha': correction.flatten(1),
            'raw_alpha': effective_alpha[:, None],
            'z_alpha': reliability_gap[:, None],
            'reliability': confidence[:, None],
            'reliability_gate': intervention_strength[:, None],
            'reliability_cls': reliability[:, :1],
            'reliability_cont': reliability[:, 1:],
            'reliability_base': 1.0 - intervention_strength[:, None],
            'intervention_gate': intervention_strength[:, None],
            'gate_logit': reliability_gap.abs()[:, None],
            'regional_intervention_gate': intervention_strength[:, None].expand(
                -1, height * width),
            'regional_gate_logit': reliability_gap.abs()[:, None].expand(
                -1, height * width),
            'delta': correction.flatten(1),
            'log_ratio': difference.flatten(1),
            'factor': factor.flatten(1),
            'base_score': base,
            'windowed_base_score': windowed_base,
            'calibrated_score': calibrated,
            'windowed_calibrated_score': windowed_calibrated,
            'global_calibrated_score': calibrated,
            'windowed_global_calibrated_score': windowed_calibrated,
            'alpha_global': effective_alpha[:, None],
            'alpha_region': correction,
            'alpha_final': correction,
            'active_mask': active,
            'pq_reliability_logits': quality_logits,
            'pq_reliability': reliability,
            'pq_cls_peak_index': cls_index,
            'pq_cont_peak_index': cont_index,
            'pq_cls_peak_xy': branch['cls_xy'],
            'pq_cont_peak_xy': branch['cont_xy'],
            'pq_peak_distance': branch['peak_distance'],
            'pq_distribution_disagreement': branch['js'],
            'pq_disagreement': branch['disagreement'],
            'pq_reliability_gap': reliability_gap,
            'pq_raw_alpha': raw_alpha,
            'pq_alpha': effective_alpha,
            'pq_difference': difference,
            'pq_raw_correction': raw_correction,
            'pq_correction': correction,
            'pq_support_mask': support,
            'pq_spatial_input': spatial_input,
            'inference_mode': self.inference_mode,
        }

"""Aligned-token reference reliability calibration for UVLTrack.

Unlike v1, this version compares UVLTrack's official vis_token and txt_token,
which are produced by the same multimodal backbone and are therefore on a
meaningfully comparable representation scale.  Per-response standardization
removes residual sharpness/scale bias before reliability is estimated.
"""

import torch
from torch.nn import functional as F

from .reference_reliability_feature_erc import (
    ReferenceReliabilityFeatureERC,
)


class ReferenceReliabilityFeatureERCV2(ReferenceReliabilityFeatureERC):
    DESIGN = 'reference_reliability_feature_erc_v2_aligned_tokens'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.design = self.DESIGN

    @staticmethod
    def _one_reference_token(reference, batch, dim, name):
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

    def forward(self, search, visual_reference, language_reference, flag):
        if search.ndim != 3:
            raise ValueError('Expected search token tensor [B,N,C]')
        batch, _, dim = search.shape
        if dim != self.dim:
            raise ValueError('Feature ERC search dimension mismatch')
        visual_reference = self._one_reference_token(
            visual_reference, batch, dim, 'visual')
        language_reference = self._one_reference_token(
            language_reference, batch, dim, 'language')

        search_evidence = search.detach()
        visual_evidence = visual_reference.detach()
        language_evidence = language_reference.detach()
        active = flag.reshape(batch).eq(2)

        search_unit = F.normalize(search_evidence, dim=-1)
        raw_visual_response = (
            search_unit * F.normalize(visual_evidence, dim=-1)[:, None]
        ).sum(-1)
        raw_language_response = (
            search_unit * F.normalize(language_evidence, dim=-1)[:, None]
        ).sum(-1)
        visual_response = self._standardize(raw_visual_response)
        language_response = self._standardize(raw_language_response)
        statistics = self._reference_statistics(
            visual_response, language_response,
            visual_evidence, language_evidence,
        )
        reliability_logits = self.reliability_head(statistics)
        reliability = F.softmax(
            reliability_logits / self.reliability_temperature, dim=-1)
        consensus = (
            reliability[:, :1] * visual_evidence
            + reliability[:, 1:] * language_evidence
        )
        hidden = self.adapter_norm(
            self.search_down(search_evidence)
            + self.reference_down(consensus)[:, None]
        )
        raw_delta = self.adapter_up(F.gelu(hidden))
        token_scale = search_evidence.norm(
            dim=-1, keepdim=True).clamp_min(1e-6)
        raw_norm = raw_delta.norm(dim=-1, keepdim=True)
        delta = (
            self.max_delta * token_scale
            * raw_delta / (raw_norm + token_scale)
        )
        mode = 'ERC_OFF' if self.force_off else self.inference_mode
        effective = active[:, None, None]
        if mode == 'ERC_OFF':
            effective = torch.zeros_like(effective)
        effective_delta = torch.where(
            effective, delta, torch.zeros_like(delta))
        return {
            'adapted_search': search + effective_delta,
            'feature_delta': effective_delta,
            'raw_feature_delta': delta,
            'reliability_logits': reliability_logits,
            'reliability': reliability,
            'visual_response': visual_response,
            'language_response': language_response,
            'raw_visual_response': raw_visual_response,
            'raw_language_response': raw_language_response,
            'visual_reference': visual_evidence,
            'language_reference': language_evidence,
            'reference_statistics': statistics,
            'search_token_norm': search_evidence.norm(dim=-1),
            'active': active,
            'mode': mode,
        }

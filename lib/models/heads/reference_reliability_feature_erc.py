"""Reference-reliability feature calibration for NL+BBOX tracking.

This ERC estimates the relative reliability of the visual template and the
language reference before the native UVLTrack Head creates its C/T response
maps.  It applies a zero-initialized residual adapter to search tokens.  The
module never reads candidate boxes or IoU and is active only for official
flag=2 vision-language tracking samples.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ReferenceReliabilityFeatureERC(nn.Module):
    DESIGN = 'reference_reliability_feature_erc_v1'

    def __init__(self, dim, hidden=64, max_delta=0.10,
                 reliability_temperature=1.0,
                 response_temperature=0.10,
                 inference_mode='FULL_ERC', force_off=False):
        super().__init__()
        if min(dim, hidden) <= 0:
            raise ValueError('Feature dimensions must be positive')
        if max_delta <= 0 or reliability_temperature <= 0:
            raise ValueError('Positive delta/temperature required')
        if response_temperature <= 0:
            raise ValueError('Response temperature must be positive')
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.max_delta = float(max_delta)
        self.reliability_temperature = float(reliability_temperature)
        self.response_temperature = float(response_temperature)
        self.force_off = bool(force_off)
        self.design = self.DESIGN

        # Eight interpretable, scale-stable statistics: branch peak, margin,
        # entropy, cross-reference agreement and response disagreement.
        self.reliability_head = nn.Sequential(
            nn.Linear(8, hidden), nn.GELU(), nn.Linear(hidden, 2),
        )
        self.search_down = nn.Linear(dim, hidden, bias=False)
        self.reference_down = nn.Linear(dim, hidden, bias=False)
        self.adapter_norm = nn.LayerNorm(hidden)
        self.adapter_up = nn.Linear(hidden, dim, bias=False)

        # Equal reliability and an exact identity feature path at creation.
        nn.init.zeros_(self.reliability_head[-1].weight)
        nn.init.zeros_(self.reliability_head[-1].bias)
        nn.init.zeros_(self.adapter_up.weight)
        self.set_inference_mode(inference_mode)

    def set_inference_mode(self, mode):
        mode = {'OFF': 'ERC_OFF', 'FULL': 'FULL_ERC'}.get(
            str(mode).upper(), str(mode).upper())
        if mode not in ('ERC_OFF', 'FULL_ERC'):
            raise ValueError('Feature ERC supports ERC_OFF/FULL_ERC only')
        self.inference_mode = mode

    @staticmethod
    def _entropy(probability):
        count = probability.shape[-1]
        scale = math.log(max(count, 2))
        return -(probability * probability.clamp_min(1e-12).log()).sum(-1) / scale

    @staticmethod
    def _peak_margin(response):
        top = response.topk(k=min(2, response.shape[-1]), dim=-1).values
        peak = top[:, 0]
        margin = peak - top[:, -1]
        return peak, margin

    def _reference_statistics(self, visual_response, language_response,
                              visual_reference, language_reference):
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
            visual_probability - language_probability
        ).abs().mean(-1)
        return torch.stack([
            visual_peak,
            visual_margin,
            self._entropy(visual_probability),
            language_peak,
            language_margin,
            self._entropy(language_probability),
            reference_agreement,
            response_agreement - response_disagreement,
        ], -1)

    def forward(self, search, template, template_mask, language_reference,
                flag):
        if search.ndim != 3 or template.ndim != 3:
            raise ValueError('Expected search/template token tensors [B,N,C]')
        batch, _, dim = search.shape
        if dim != self.dim or template.shape[0] != batch:
            raise ValueError('Feature ERC token shape mismatch')
        if template_mask is None or template_mask.shape != template.shape[:2]:
            raise ValueError('Feature ERC requires template target mask')
        if language_reference.ndim == 3:
            if language_reference.shape[1] != 1:
                raise ValueError('Expected one language reference token')
            language_reference = language_reference[:, 0]
        if language_reference.shape != (batch, dim):
            raise ValueError('Language reference shape mismatch')

        # Backbone tensors are evidence, not trainable ERC targets.  Detaching
        # keeps ERC auxiliary gradients away from the frozen backbone while
        # the native Head loss still reaches the adapter parameters.
        search_evidence = search.detach()
        template_evidence = template.detach()
        language_evidence = language_reference.detach()
        mask = template_mask.bool()
        count = mask.sum(-1)
        active = flag.reshape(batch).eq(2)
        if (active & count.eq(0)).any():
            raise ValueError('Active Feature ERC sample has empty template mask')
        visual_reference = (
            template_evidence * mask[..., None]
        ).sum(1) / count.clamp_min(1)[:, None]

        search_unit = F.normalize(search_evidence, dim=-1)
        visual_response = (
            search_unit * F.normalize(visual_reference, dim=-1)[:, None]
        ).sum(-1)
        language_response = (
            search_unit * F.normalize(language_evidence, dim=-1)[:, None]
        ).sum(-1)
        statistics = self._reference_statistics(
            visual_response, language_response,
            visual_reference, language_evidence,
        )
        reliability_logits = self.reliability_head(statistics)
        reliability = F.softmax(
            reliability_logits / self.reliability_temperature, dim=-1)
        consensus = (
            reliability[:, :1] * visual_reference
            + reliability[:, 1:] * language_evidence
        )
        hidden = self.adapter_norm(
            self.search_down(search_evidence)
            + self.reference_down(consensus)[:, None]
        )
        raw_delta = self.adapter_up(F.gelu(hidden))
        # Bound the residual relative to each token norm.  The zero terminal
        # projection still makes the initial mapping bitwise identity.
        token_scale = search_evidence.norm(dim=-1, keepdim=True).clamp_min(1e-6)
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
            effective, delta, torch.zeros_like(delta)
        )
        adapted_search = search + effective_delta
        return {
            'adapted_search': adapted_search,
            'feature_delta': effective_delta,
            'raw_feature_delta': delta,
            'reliability_logits': reliability_logits,
            'reliability': reliability,
            'visual_response': visual_response,
            'language_response': language_response,
            'visual_reference': visual_reference,
            'language_reference': language_evidence,
            'reference_statistics': statistics,
            'search_token_norm': search_evidence.norm(dim=-1),
            'active': active,
            'mode': mode,
        }

"""Target-reference conditioned direct dense ERC (DR1).

DR1 is the evidence-level fallback selected after the response-peak oracle
showed that a fixed Top-K candidate set was too restrictive.  It predicts a
bounded signed residual directly on the native UVLTrack log fusion score:

    log S_erc(x) = log[H(x) C(x) T(x)] + Delta(x).

Unlike earlier dense ERC versions, Delta is not multiplied by log(C)-log(T),
so the module cannot collapse into choosing only one branch.  Its evidence is
the uncompressed target/distractor/background prompt response plus aligned
visual/language reference response.  It never reads candidate boxes, bbox
geometry or IoU; those remain reserved for RQC.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .reference_reliability_dense_response_erc_v3 import (
    ReferenceReliabilityDenseResponseERCV3,
)


class TargetReferenceDirectDenseERCDR1(ReferenceReliabilityDenseResponseERCV3):
    """Direct, reference-conditioned response residual for flag=2 tracking."""

    DESIGN = 'target_reference_direct_dense_erc_dr1'
    NUM_INPUT_CHANNELS = 12

    def __init__(self, *args, **kwargs):
        # The shared response-ERC factory supplies these action-space
        # arguments for older discrete variants; DR1 has no action catalogue.
        kwargs.pop('magnitudes', None)
        kwargs.pop('action_temperature', None)
        super().__init__(*args, **kwargs)
        # The v3 global reliability head and alpha-on-D projection are not
        # part of DR1.  Keeping their parameters frozen preserves checkpoint
        # compatibility without adding unused optimizer state.
        self.reliability_head.requires_grad_(False)
        self.alpha_head.requires_grad_(False)
        spatial_hidden = max(self.hidden // 2, 16)
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
        self.residual_head = nn.Conv2d(spatial_hidden, 1, kernel_size=1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.design = self.DESIGN

    def forward(
        self, cls_response, cont_response, search,
        visual_reference, language_reference, prompts, flag,
    ):
        if cls_response.ndim != 3 or cls_response.shape != cont_response.shape:
            raise ValueError('DR1 expects matching C/T maps [B,H,W]')
        batch, height, width = cls_response.shape
        if (height, width) != (self.feat_sz, self.feat_sz):
            raise ValueError('DR1 response size mismatch')
        if search.shape != (batch, height * width, self.dim):
            raise ValueError('DR1 search token shape mismatch')
        if prompts.ndim != 3 or prompts.shape[0] != batch or (
                prompts.shape[1] < 3 or prompts.shape[2] != self.dim):
            raise ValueError(
                'DR1 requires target/distractor/background prompts [B,>=3,D]')

        visual_reference = self._one_reference(
            visual_reference, batch, self.dim, 'visual')
        language_reference = self._one_reference(
            language_reference, batch, self.dim, 'language')
        if self.detach_features:
            search_evidence = search.detach()
            visual_evidence = visual_reference.detach()
            language_evidence = language_reference.detach()
            prompt_evidence = prompts.detach()
        else:
            search_evidence = search
            visual_evidence = visual_reference
            language_evidence = language_reference
            prompt_evidence = prompts

        search_unit = F.normalize(search_evidence, dim=-1)
        prompt_cosine = search_unit @ F.normalize(
            prompt_evidence[:, :3], dim=-1).transpose(-2, -1)
        target_cosine = prompt_cosine[..., 0]
        distractor_cosine = prompt_cosine[..., 1]
        background_cosine = prompt_cosine[..., 2]
        zero = torch.zeros_like(target_cosine)
        target_margin = target_cosine - torch.maximum(
            torch.maximum(distractor_cosine, background_cosine), zero)

        visual_response = self._standardize((
            search_unit * F.normalize(visual_evidence, dim=-1)[:, None]
        ).sum(-1))
        language_response = self._standardize((
            search_unit * F.normalize(language_evidence, dim=-1)[:, None]
        ).sum(-1))
        reference_response = 0.5 * (visual_response + language_response)

        stable = self._stable_response_channels(cls_response, cont_response)
        window = stable['window'].expand(batch, -1, -1)
        support = window.gt(0.0)
        spatial_input = torch.stack([
            stable['log_c'],
            stable['log_t'],
            stable['log_s'],
            stable['difference'] / self.d_max,
            target_cosine.reshape(batch, height, width),
            distractor_cosine.reshape(batch, height, width),
            background_cosine.reshape(batch, height, width),
            (target_margin / 2.0).clamp(-1.0, 1.0).reshape(
                batch, height, width),
            visual_response.reshape(batch, height, width).clamp(-4.0, 4.0) / 4.0,
            language_response.reshape(batch, height, width).clamp(-4.0, 4.0) / 4.0,
            reference_response.reshape(batch, height, width).clamp(-4.0, 4.0) / 4.0,
            window,
        ], dim=1)
        encoder_input = (
            spatial_input.detach() if self.detach_features else spatial_input)
        spatial_features = self.encoder(encoder_input)
        residual_logits = self.residual_head(spatial_features).squeeze(1)
        bounded = self.max_alpha * torch.tanh(residual_logits)
        # Remove score-invariant global offset.  Clamping preserves the stated
        # +/-MAX_ALPHA bound and gives a 2*MAX_ALPHA pairwise ranking span.
        support_float = support.to(bounded)
        mean = (bounded * support_float).flatten(1).sum(-1) / (
            support_float.flatten(1).sum(-1).clamp_min(1.0))
        raw_correction = (
            bounded - mean[:, None, None]).clamp(
                -self.max_alpha, self.max_alpha)
        raw_correction = raw_correction * support_float

        if self.apply_vl_tracking_only:
            active = flag.reshape(batch).eq(2)
        else:
            active = torch.ones(
                batch, dtype=torch.bool, device=cls_response.device)
        runtime_enabled = (
            not self.force_off and self.inference_mode == 'FULL_ERC')
        enabled = active if runtime_enabled else torch.zeros_like(active)
        correction = torch.where(
            enabled[:, None, None], raw_correction,
            torch.zeros_like(raw_correction))
        factor = correction.exp()
        base = stable['base'].flatten(1)
        windowed_base = stable['windowed_base'].flatten(1)
        calibrated = base * factor.flatten(1)
        windowed_calibrated = windowed_base * factor.flatten(1)
        strength = correction.abs().flatten(1).mean(-1, keepdim=True)

        return {
            'features': spatial_features.mean((-2, -1)),
            'cls_response': cls_response.flatten(1),
            'cont_response': cont_response.flatten(1),
            'windowed_cls_score': (cls_response * window).flatten(1),
            'windowed_cont_score': (cont_response * window).flatten(1),
            'alpha': correction.flatten(1),
            'raw_alpha': raw_correction.flatten(1),
            'z_alpha': residual_logits.flatten(1),
            'reliability': 1.0 - strength / max(self.max_alpha, self.eps),
            'reliability_gate': strength / max(self.max_alpha, self.eps),
            'reliability_cls': cls_response.flatten(1).amax(-1, keepdim=True),
            'reliability_cont': cont_response.flatten(1).amax(-1, keepdim=True),
            'reliability_base': 1.0 - strength / max(self.max_alpha, self.eps),
            'intervention_gate': strength / max(self.max_alpha, self.eps),
            'gate_logit': residual_logits.abs().flatten(1).mean(-1, keepdim=True),
            'regional_intervention_gate': (
                correction.abs().flatten(1) / max(self.max_alpha, self.eps)),
            'regional_gate_logit': residual_logits.flatten(1),
            'delta': correction.flatten(1),
            'log_ratio': stable['difference'].flatten(1),
            'factor': factor.flatten(1),
            'base_score': base,
            'windowed_base_score': windowed_base,
            'calibrated_score': calibrated,
            'windowed_calibrated_score': windowed_calibrated,
            'global_calibrated_score': base,
            'windowed_global_calibrated_score': windowed_base,
            'alpha_global': correction.flatten(1).mean(-1, keepdim=True),
            'alpha_region': correction,
            'alpha_final': correction,
            'active_mask': active,
            'dr1_spatial_input': spatial_input,
            'dr1_spatial_features': spatial_features,
            'dr1_target_cosine': target_cosine.reshape(batch, height, width),
            'dr1_distractor_cosine': distractor_cosine.reshape(
                batch, height, width),
            'dr1_background_cosine': background_cosine.reshape(
                batch, height, width),
            'dr1_target_margin': target_margin.reshape(batch, height, width),
            'dr1_visual_response': visual_response.reshape(
                batch, height, width),
            'dr1_language_response': language_response.reshape(
                batch, height, width),
            'dr1_residual_logits': residual_logits,
            'dr1_raw_correction': raw_correction,
            'dr1_correction': correction,
            'dr1_support_mask': support,
            'inference_mode': self.inference_mode,
        }

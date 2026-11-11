"""Asymmetric peak-verification ERC (PQ4).

PQ3 treated the classification response C and the target-contrast response T
as exchangeable control directions.  In practice its oracle and predictor
collapsed onto C.  PQ4 gives the branches deliberately different jobs:

* The native fusion, C and T expose a small union of spatial alternatives.
* C determines whether the frame is ambiguous.
* T plus the visual/language reference responses verifies the alternatives.

Only existing response peaks are re-ranked.  No bbox geometry or candidate
IoU enters the module, so candidate-box quality estimation remains the job of
RQC.  The final verifier layer is zero initialized and every correction is
anchored to the native UVLTrack peak; consequently PQ4 exactly recovers the
native H*C*T decision at initialization and whenever it abstains.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .peak_quality_calibrated_erc_pq1 import PeakQualityCalibratedERCPQ1


class PeakAmbiguityTargetVerifierERCPQ4(PeakQualityCalibratedERCPQ1):
    """C-proposal/T-verification response calibration."""

    DESIGN = 'peak_ambiguity_target_verifier_erc_pq4'

    def __init__(self, *args, response_topk=10, **kwargs):
        super().__init__(*args, **kwargs)
        if int(response_topk) < 2:
            raise ValueError('PQ4 requires RESPONSE_TOPK >= 2')
        self.response_topk = int(response_topk)

        # PQ1's global reliability head is intentionally not part of PQ4.
        self.quality_head.requires_grad_(False)

        # Candidate evidence is intentionally asymmetric.  These four maps
        # are T/reference-side evidence from PQ1's stable spatial input:
        # log(T), visual-reference similarity, language-reference similarity,
        # and their shared reference context.  Current and baseline-relative
        # values are used together with response-position offsets and the
        # deterministic C ambiguity value.
        evidence_channels = 4
        verifier_input_dim = 2 * evidence_channels + 4
        hidden = self.quality_head[0].out_features
        self.verifier_head = nn.Sequential(
            nn.Linear(verifier_input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.verifier_head[-1].weight)
        nn.init.zeros_(self.verifier_head[-1].bias)
        self.design = self.DESIGN

    @staticmethod
    def _gather_candidates(values, indices):
        """Gather [B,N,D] values at [B,K] indices."""
        return values.gather(
            1, indices[:, :, None].expand(-1, -1, values.shape[-1]))

    def _candidate_indices(
            self, windowed_base, windowed_cls, windowed_cont, support):
        """Return a deduplicated native/C/T peak union."""
        batch = windowed_base.shape[0]
        support_flat = support.flatten(1)
        minimum = torch.finfo(windowed_base.dtype).min
        base_index = windowed_base.flatten(1).masked_fill(
            ~support_flat, minimum).argmax(-1)

        budget = self.response_topk - 1
        counts = [budget // 3 + int(index < budget % 3) for index in range(3)]
        proposal_indices = []
        proposal_valid = []
        for response, count in zip(
                (windowed_base, windowed_cls, windowed_cont), counts):
            if count == 0:
                continue
            local_maximum = response.eq(F.max_pool2d(
                response[:, None], kernel_size=3, stride=1, padding=1)[:, 0])
            proposal_mask = local_maximum & support
            proposal_score = response.flatten(1).masked_fill(
                ~proposal_mask.flatten(1), minimum)
            _, indices = proposal_score.topk(count, dim=-1)
            proposal_indices.append(indices)
            proposal_valid.append(
                proposal_mask.flatten(1).gather(1, indices))

        alternatives = torch.cat(proposal_indices, dim=1)
        alternative_valid = torch.cat(proposal_valid, dim=1)
        indices = torch.cat([base_index[:, None], alternatives], dim=1)
        valid = torch.cat([
            torch.ones(batch, 1, dtype=torch.bool, device=base_index.device),
            alternative_valid,
        ], dim=1)
        # Remove repeated locations while preserving source priority:
        # native fusion, native local peaks, C peaks, then T peaks.
        for slot in range(1, indices.shape[1]):
            duplicate = (
                indices[:, slot, None].eq(indices[:, :slot])
                & valid[:, :slot]).any(-1)
            valid[:, slot] &= ~duplicate
        return indices, valid

    def forward(
        self, cls_response, cont_response, search,
        visual_reference, language_reference, flag,
    ):
        output = super().forward(
            cls_response, cont_response, search,
            visual_reference, language_reference, flag,
        )
        batch, height, width = cls_response.shape
        support = output['pq_support_mask']
        windowed_base = output['windowed_base_score'].reshape(
            batch, height, width)
        windowed_cls = output['windowed_cls_score'].reshape(
            batch, height, width)
        windowed_cont = output['windowed_cont_score'].reshape(
            batch, height, width)
        candidate_index, candidate_valid = self._candidate_indices(
            windowed_base, windowed_cls, windowed_cont, support)

        # C supplies a deterministic ambiguity prior.  A small top-1/top-2
        # gap opens the verifier; a clear C decision suppresses intervention.
        cls_candidate = windowed_cls.flatten(1).gather(1, candidate_index)
        cls_candidate = cls_candidate.masked_fill(~candidate_valid, 0.0)
        top_two = cls_candidate.masked_fill(
            ~candidate_valid, torch.finfo(cls_candidate.dtype).min
        ).topk(2, dim=-1).values
        cls_gap = (
            (top_two[:, 0] - top_two[:, 1]).clamp_min(0.0)
            / top_two[:, 0].abs().clamp_min(self.eps)
        ).clamp(0.0, 1.0)
        ambiguity = (1.0 - cls_gap).clamp(0.0, 1.0)

        # T/reference-only candidate evidence.  The channel selection is
        # deliberate: C is not allowed to silently become the verifier again.
        spatial = output['pq_spatial_input'][:, [1, 4, 5, 6]]
        spatial = spatial.flatten(2).transpose(1, 2)
        candidate_evidence = self._gather_candidates(spatial, candidate_index)
        baseline_evidence = candidate_evidence[:, :1]
        relative_evidence = candidate_evidence - baseline_evidence

        base_y = torch.div(
            candidate_index[:, :1], width, rounding_mode='floor')
        base_x = candidate_index[:, :1].remainder(width)
        candidate_y = torch.div(
            candidate_index, width, rounding_mode='floor')
        candidate_x = candidate_index.remainder(width)
        coordinate_scale = float(max(height - 1, width - 1, 1))
        offsets = torch.stack([
            (candidate_x - base_x).to(cls_response.dtype) / coordinate_scale,
            (candidate_y - base_y).to(cls_response.dtype) / coordinate_scale,
        ], dim=-1)
        ambiguity_feature = ambiguity[:, None, None].expand(
            -1, candidate_index.shape[1], 1)
        valid_feature = candidate_valid[:, :, None].to(cls_response.dtype)
        verifier_input = torch.cat([
            candidate_evidence,
            relative_evidence,
            offsets,
            ambiguity_feature,
            valid_feature,
        ], dim=-1)
        if self.detach_features:
            verifier_input = verifier_input.detach()

        verifier_logits = self.verifier_head(verifier_input).squeeze(-1)
        bounded = self.max_alpha * torch.tanh(verifier_logits)
        correction_ambiguity = (
            ambiguity.detach() if self.detach_features else ambiguity)
        # Relative anchoring makes the native selected peak exactly zero.  The
        # 0.5 factor keeps every pairwise residual within MAX_ALPHA.
        peak_correction = (
            0.5 * (bounded - bounded[:, :1])
            * correction_ambiguity[:, None])
        peak_correction = peak_correction.masked_fill(~candidate_valid, 0.0)

        active = output['active_mask']
        runtime_enabled = (
            not self.force_off and self.inference_mode == 'FULL_ERC')
        enabled = active if runtime_enabled else torch.zeros_like(active)
        effective_peak = torch.where(
            enabled[:, None], peak_correction,
            torch.zeros_like(peak_correction))
        correction_flat = cls_response.new_zeros(
            batch, height * width).scatter(
                1, candidate_index, effective_peak)
        correction = correction_flat.reshape(batch, height, width)
        factor = correction.exp()

        base = output['base_score']
        windowed_base_flat = output['windowed_base_score']
        calibrated = base * factor.flatten(1)
        windowed_calibrated = windowed_base_flat * factor.flatten(1)
        intervention = (
            effective_peak.abs().amax(-1) / max(self.max_alpha, self.eps))

        output.update({
            'alpha': correction.flatten(1),
            'raw_alpha': intervention[:, None],
            'z_alpha': ambiguity[:, None],
            'reliability': 1.0 - intervention[:, None],
            'reliability_gate': intervention[:, None],
            'reliability_cls': (1.0 - ambiguity)[:, None],
            'reliability_cont': ambiguity[:, None],
            'reliability_base': 1.0 - intervention[:, None],
            'intervention_gate': intervention[:, None],
            'gate_logit': ambiguity[:, None],
            'regional_intervention_gate': intervention[:, None].expand(
                -1, height * width),
            'delta': correction.flatten(1),
            'log_ratio': relative_evidence[..., 0],
            'factor': factor.flatten(1),
            'calibrated_score': calibrated,
            'windowed_calibrated_score': windowed_calibrated,
            'global_calibrated_score': calibrated,
            'windowed_global_calibrated_score': windowed_calibrated,
            'alpha_global': intervention[:, None],
            'alpha_region': correction,
            'alpha_final': correction,
            'pq4_candidate_indices': candidate_index,
            'pq4_candidate_valid': candidate_valid,
            'pq4_cls_ambiguity': ambiguity,
            'pq4_verifier_logits': verifier_logits,
            'pq4_raw_peak_correction': peak_correction,
            'pq4_peak_correction': effective_peak,
            'pq4_correction': correction,
            'inference_mode': self.inference_mode,
        })
        return output

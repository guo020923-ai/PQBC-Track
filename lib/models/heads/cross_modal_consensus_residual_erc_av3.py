"""Cross-modal consensus-residual temporal action-value ERC (ERC-AV3).

AV3 keeps AV2's unified KEEP/action utility and causal temporal consensus, but
removes the one-dimensional ``log(C)-log(T)`` action family.  Its 15 non-KEEP
actions instead calibrate the official fused response with three bounded,
response-level consensus residuals:

* global visual-language/branch consensus;
* regional consensus at high-disagreement locations;
* distractor veto where branch evidence is high but references disagree.

Candidate boxes and geometry are never inference inputs.  Detached candidate
IoU is used only by the inherited training oracle to label the utility of the
reachable response-calibration actions.
"""

import torch
from torch.nn import functional as F

from .branch_aligned_temporal_consensus_erc_av2 import (
    BranchAlignedTemporalConsensusERCAV2,
)


class CrossModalConsensusResidualERCAV3(
        BranchAlignedTemporalConsensusERCAV2):
    """Temporal action value over cross-modal spatial consensus residuals."""

    DESIGN = 'cross_modal_consensus_residual_erc_av3'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        number = int(self.reachability_magnitudes.numel())
        # Action order: KEEP, GLOBAL_CONSENSUS, REGIONAL_CONSENSUS,
        # REFERENCE_VETO.  Five magnitudes per non-KEEP family preserve AV2's
        # 16-way utility-head and detached-oracle contracts.
        family = [0] + [1] * number + [2] * number + [3] * number
        strength = [0.0] + self.reachability_magnitudes.tolist() * 3
        self.register_buffer('av3_action_family', torch.tensor(family))
        self.register_buffer('av3_action_strength', torch.tensor(strength))
        self.design = self.DESIGN

    @staticmethod
    def _masked_unit_signed(values, support):
        """Zero-centre and bound a signed residual on the valid Hann support."""
        support_float = support.to(values.dtype)
        count = support_float.sum((-2, -1), keepdim=True).clamp_min(1.0)
        mean = (values * support_float).sum((-2, -1), keepdim=True) / count
        centred = (values - mean) * support_float
        scale = centred.abs().amax((-2, -1), keepdim=True).clamp_min(1e-6)
        return (centred / scale).clamp(-1.0, 1.0) * support_float

    @staticmethod
    def _masked_unit_positive(values, support):
        support_float = support.to(values.dtype)
        values = values.clamp_min(0.0) * support_float
        scale = values.amax((-2, -1), keepdim=True).clamp_min(1e-6)
        return (values / scale).clamp(0.0, 1.0) * support_float

    def _consensus_action_maps(
            self, cls_response, cont_response, search,
            visual_reference, language_reference):
        batch, height, width = cls_response.shape
        stable = self._stable_response_channels(cls_response, cont_response)
        support = stable['support']
        support_float = support.to(cls_response.dtype)

        search_evidence = search.detach() if self.detach_features else search
        visual_evidence = (
            visual_reference.detach() if self.detach_features
            else visual_reference)
        language_evidence = (
            language_reference.detach() if self.detach_features
            else language_reference)
        search_unit = F.normalize(search_evidence, dim=-1)
        visual_map = self._standardize(
            (search_unit * F.normalize(
                visual_evidence, dim=-1)[:, None]).sum(-1), support,
        ).reshape(batch, height, width)
        language_map = self._standardize(
            (search_unit * F.normalize(
                language_evidence, dim=-1)[:, None]).sum(-1), support,
        ).reshape(batch, height, width)

        cls_map = self._standardize(
            cls_response.clamp_min(self.eps).log(), support,
        ).reshape(batch, height, width)
        cont_map = self._standardize(
            cont_response.clamp_min(self.eps).log(), support,
        ).reshape(batch, height, width)
        branch_consensus = 0.5 * (cls_map + cont_map)
        reference_consensus = 0.5 * (visual_map + language_map)

        # Only strong visual-language agreement may overwrite native branch
        # evidence.  This prevents one noisy reference modality from becoming
        # a free spatial classification head.
        reference_agreement = torch.exp(
            -0.5 * (visual_map - language_map).abs()) * support_float
        raw_consensus = (
            reference_consensus - branch_consensus) * reference_agreement
        global_basis = self._masked_unit_signed(raw_consensus, support)

        disagreement = F.avg_pool2d(
            raw_consensus.abs()[:, None], 3, stride=1, padding=1,
        ).squeeze(1)
        regional_mask = self._masked_unit_positive(disagreement, support)
        # Preserve locality: re-centring after applying the regional mask
        # would create a weak correction over the whole valid map.
        regional_basis = global_basis * regional_mask
        regional_scale = regional_basis.abs().amax(
            (-2, -1), keepdim=True).clamp_min(1e-6)
        regional_basis = (
            regional_basis / regional_scale
        ).clamp(-1.0, 1.0) * support_float

        # Suppress only branch-supported locations rejected by mutually
        # consistent visual/language references.  KEEP remains available, so
        # the utility oracle can reject this action when T/C evidence is useful.
        branch_peak = F.relu(branch_consensus)
        reference_deficit = F.relu(branch_consensus - reference_consensus)
        veto_strength = F.avg_pool2d(
            (branch_peak * reference_deficit * reference_agreement)[:, None],
            3, stride=1, padding=1,
        ).squeeze(1)
        veto_basis = -self._masked_unit_positive(veto_strength, support)

        action_maps = [torch.zeros_like(global_basis)]
        magnitudes = self.reachability_magnitudes.to(global_basis)
        action_maps.extend(magnitude * global_basis for magnitude in magnitudes)
        action_maps.extend(magnitude * regional_basis for magnitude in magnitudes)
        action_maps.extend(magnitude * veto_basis for magnitude in magnitudes)
        return {
            'stable': stable,
            'action_maps': torch.stack(action_maps, dim=1),
            'global_basis': global_basis,
            'regional_basis': regional_basis,
            'veto_basis': veto_basis,
            'regional_mask': regional_mask,
            'reference_agreement': reference_agreement,
            'branch_consensus': branch_consensus * support_float,
            'reference_consensus': reference_consensus * support_float,
        }

    def forward(
        self, cls_response, cont_response, search,
        visual_reference, language_reference, flag,
    ):
        batch, height, width = cls_response.shape
        visual_reference = self._one_reference(
            visual_reference, batch, self.dim, 'visual')
        language_reference = self._one_reference(
            language_reference, batch, self.dim, 'language')

        # AV2 computes the shared evidence encoder, utility prediction, causal
        # memory and KEEP/action decision.  We replace only its old C/T-axis
        # correction maps with the consensus-residual action family below.
        output = super().forward(
            cls_response, cont_response, search,
            visual_reference, language_reference, flag,
        )
        consensus = self._consensus_action_maps(
            cls_response, cont_response, search,
            visual_reference, language_reference,
        )
        action_maps = consensus['action_maps']
        raw_action = output['av1_raw_hard_action']
        rows = torch.arange(batch, device=raw_action.device)
        selected_full = action_maps[rows, raw_action]
        utility_gate = output['av1_utility_gate']
        predicted_correction = utility_gate[:, None, None] * selected_full

        active = output['active_mask']
        runtime_enabled = (
            not self.force_off and self.inference_mode == 'FULL_ERC')
        enabled = active if runtime_enabled else torch.zeros_like(active)
        effective_correction = torch.where(
            enabled[:, None, None], predicted_correction,
            torch.zeros_like(predicted_correction))
        effective_action = torch.where(
            enabled, raw_action, torch.zeros_like(raw_action))
        effective_gate = torch.where(
            enabled, utility_gate, torch.zeros_like(utility_gate))

        stable = consensus['stable']
        factor = effective_correction.exp()
        base = stable['base'].flatten(1)
        windowed_base = stable['windowed_base'].flatten(1)
        calibrated = base * factor.flatten(1)
        windowed_calibrated = windowed_base * factor.flatten(1)

        action_probability = output['av1_action_probability']
        family = self.av3_action_family.to(action_probability.device)
        family_probability = torch.stack([
            action_probability[:, family.eq(index)].sum(-1)
            for index in range(4)
        ], dim=-1)
        selected_family = family[raw_action]
        selected_strength = self.av3_action_strength.to(
            utility_gate)[raw_action]
        effective_strength = effective_gate * selected_strength
        regional_selected = selected_family.eq(2) | selected_family.eq(3)

        output.update({
            'alpha': effective_correction.flatten(1),
            'raw_alpha': (utility_gate * selected_strength)[:, None],
            'z_alpha': (
                family_probability[:, 1]
                + family_probability[:, 2]
                - family_probability[:, 3]
            )[:, None],
            'reliability': family_probability.max(-1, keepdim=True).values,
            'reliability_gate': effective_gate[:, None],
            'reliability_cls': (
                effective_gate * (
                    family_probability[:, 1] + family_probability[:, 2])
            )[:, None],
            'reliability_cont': (
                effective_gate * family_probability[:, 3])[:, None],
            'reliability_base': (1.0 - effective_gate)[:, None],
            'intervention_gate': effective_gate[:, None],
            'regional_intervention_gate': (
                effective_gate * regional_selected.to(effective_gate)
            )[:, None].expand(-1, height * width),
            'delta': effective_correction.flatten(1),
            'log_ratio': consensus['global_basis'].flatten(1),
            'factor': factor.flatten(1),
            'base_score': base,
            'windowed_base_score': windowed_base,
            'calibrated_score': calibrated,
            'windowed_calibrated_score': windowed_calibrated,
            'global_calibrated_score': calibrated,
            'windowed_global_calibrated_score': windowed_calibrated,
            'alpha_global': torch.where(
                selected_family.eq(1), effective_strength,
                torch.zeros_like(effective_strength))[:, None],
            'alpha_region': effective_correction,
            'alpha_final': effective_correction,
            'av1_action_correction_maps': action_maps,
            'av1_selected_full_correction': selected_full,
            'av1_predicted_correction_map': predicted_correction,
            'av1_effective_correction_map': effective_correction,
            'av1_support_mask': stable['support'],
            'av1_log_base': stable['windowed_base'].clamp_min(
                self.eps).log().masked_fill(
                    ~stable['support'], torch.finfo(cls_response.dtype).min),
            'av1_action_direction': self.av3_action_family,
            'av1_action_scope': self.av3_action_family,
            'av1_action_signed_alpha': self.av3_action_strength,
            'av1_semantic_probability': family_probability,
            'av3_action_family': self.av3_action_family,
            'av3_action_strength': self.av3_action_strength,
            'av3_family_probability': family_probability,
            'av3_selected_family': selected_family,
            'av3_selected_strength': selected_strength,
            'av3_effective_strength': effective_strength,
            'av3_global_basis': consensus['global_basis'],
            'av3_regional_basis': consensus['regional_basis'],
            'av3_veto_basis': consensus['veto_basis'],
            'av3_regional_mask': consensus['regional_mask'],
            'av3_reference_agreement': consensus['reference_agreement'],
            'av3_branch_consensus': consensus['branch_consensus'],
            'av3_reference_consensus': consensus['reference_consensus'],
        })
        return output

"""Temporal-consensus action-value evidence calibration (ERC-AV2).

ERC-AV2 retains AV1b's response-only 16-action calibration family, but makes
the deployed decision depend on a short history of predicted action utility.
The history is used only for single-sequence inference and is reset at every
tracker initialization.  Batched training/validation remains stateless; its
cross-frame supervision is implemented by the corresponding AV2 loss.
"""

import torch
from torch.nn import functional as F

from .branch_aligned_action_value_erc_av1b import (
    BranchAlignedActionValueERCAV1B,
)


class BranchAlignedTemporalConsensusERCAV2(
        BranchAlignedActionValueERCAV1B):
    """Use current and recent C/T evidence to select a stable action."""

    DESIGN = 'branch_aligned_temporal_consensus_erc_av2'

    def __init__(
        self, *args, temporal_ema=0.60, release_margin=0.02,
        enable_temporal_memory=True, require_current_positive=False,
        require_group_confirmation=False,
        min_current_temporal_gate=False, current_support_margin=0.0,
        **kwargs,
    ):
        if not 0.0 <= float(temporal_ema) < 1.0:
            raise ValueError('ERC-AV2 temporal EMA must be in [0, 1)')
        if float(release_margin) < 0.0:
            raise ValueError('ERC-AV2 release margin must be non-negative')
        if float(current_support_margin) < 0.0:
            raise ValueError('current support margin must be non-negative')
        super().__init__(*args, **kwargs)
        self.temporal_ema = float(temporal_ema)
        self.release_margin = float(release_margin)
        self.enable_temporal_memory = bool(enable_temporal_memory)
        self.require_current_positive = bool(require_current_positive)
        self.require_group_confirmation = bool(require_group_confirmation)
        self.min_current_temporal_gate = bool(min_current_temporal_gate)
        self.current_support_margin = float(current_support_margin)
        self._temporal_utility_memory = None
        self.design = self.DESIGN

    def _temporal_action_group(self):
        """Semantic action grouping used by optional current-frame checks."""
        return self.v6_action_direction

    def reset_temporal_state(self):
        """Forget evidence from the previous sequence."""
        self._temporal_utility_memory = None

    def _temporal_utility(self, current_utility, active):
        """Return causal EMA utility without leaking state across batches."""
        use_memory = (
            self.enable_temporal_memory
            and not self.training
            and current_utility.shape[0] == 1
            and bool(active.reshape(-1)[0].item())
            and not self.force_off
            and self.inference_mode == 'FULL_ERC'
        )
        memory_used = torch.zeros(
            current_utility.shape[0], dtype=torch.bool,
            device=current_utility.device,
        )
        if not use_memory:
            return current_utility, memory_used

        if self._temporal_utility_memory is None:
            temporal_utility = current_utility
        else:
            previous = self._temporal_utility_memory.to(
                device=current_utility.device, dtype=current_utility.dtype)
            temporal_utility = (
                self.temporal_ema * previous
                + (1.0 - self.temporal_ema) * current_utility
            )
            memory_used.fill_(True)
        self._temporal_utility_memory = temporal_utility.detach()
        return temporal_utility, memory_used

    def forward(
        self, cls_response, cont_response, search,
        visual_reference, language_reference, flag,
    ):
        output = super().forward(
            cls_response, cont_response, search,
            visual_reference, language_reference, flag,
        )
        current_utility = output['av1_action_utility']
        active = output['active_mask']
        temporal_utility, memory_used = self._temporal_utility(
            current_utility, active)

        current_nonkeep = current_utility[:, 1:]
        current_best, current_nonkeep_action = current_nonkeep.max(-1)
        current_action = torch.where(
            current_best.gt(0.0), current_nonkeep_action + 1,
            torch.zeros_like(current_nonkeep_action),
        )

        temporal_nonkeep = temporal_utility[:, 1:]
        best_utility, best_nonkeep = temporal_nonkeep.max(-1)
        proposed_action = best_nonkeep + 1
        rows = torch.arange(
            current_utility.shape[0], device=current_utility.device)
        current_support = current_utility[rows, proposed_action]
        current_supported = current_support.gt(self.current_support_margin)
        group_confirmed = torch.ones_like(current_supported)
        if self.require_group_confirmation:
            action_group = self._temporal_action_group().to(
                current_utility.device)
            group_confirmed = action_group[current_action].eq(
                action_group[proposed_action])
        should_intervene = (
            best_utility.gt(0.0)
            & current_support.ge(-self.release_margin)
        )
        if self.require_current_positive:
            should_intervene = should_intervene & current_supported
        if self.require_group_confirmation:
            should_intervene = should_intervene & group_confirmed
        raw_hard_action = torch.where(
            should_intervene, proposed_action,
            torch.zeros_like(proposed_action),
        )
        hard_policy = F.one_hot(
            raw_hard_action, temporal_utility.shape[-1]
        ).to(temporal_utility.dtype)
        gate_utility = best_utility
        if self.min_current_temporal_gate:
            gate_utility = torch.minimum(best_utility, current_support)
        utility_gate = (
            F.relu(gate_utility) / self.utility_gate_scale
        ).clamp(max=1.0)
        utility_gate = torch.where(
            should_intervene, utility_gate,
            torch.zeros_like(utility_gate),
        )

        corrections = output['av1_action_correction_maps']
        selected_correction = corrections[rows, raw_hard_action]
        predicted_correction = (
            utility_gate[:, None, None] * selected_correction
        )
        signed_alpha = output['av1_action_signed_alpha'].to(
            temporal_utility)[raw_hard_action]
        raw_signed_alpha = utility_gate * signed_alpha

        runtime_enabled = (
            not self.force_off and self.inference_mode == 'FULL_ERC'
        )
        enabled = active if runtime_enabled else torch.zeros_like(active)
        effective_correction = torch.where(
            enabled[:, None, None], predicted_correction,
            torch.zeros_like(predicted_correction),
        )
        effective_action = torch.where(
            enabled, raw_hard_action, torch.zeros_like(raw_hard_action))
        effective_gate = torch.where(
            enabled, utility_gate, torch.zeros_like(utility_gate))

        factor = effective_correction.exp()
        base = output['base_score']
        windowed_base = output['windowed_base_score']
        calibrated = base * factor.flatten(1)
        windowed_calibrated = windowed_base * factor.flatten(1)

        action_probability = F.softmax(
            temporal_utility / self.utility_temperature, dim=-1)
        nonkeep_probability = F.softmax(
            temporal_nonkeep / self.utility_temperature, dim=-1)
        direction = output['av1_action_direction'][1:].to(
            temporal_utility.device)
        cls_probability = (
            nonkeep_probability[:, direction.eq(1)].sum(-1) * utility_gate)
        cont_probability = (
            nonkeep_probability[:, direction.eq(2)].sum(-1) * utility_gate)
        keep_probability = 1.0 - utility_gate
        semantic_probability = torch.stack([
            keep_probability, cls_probability, cont_probability,
        ], dim=-1)

        height, width = cls_response.shape[-2:]
        output.update({
            'alpha': effective_correction.flatten(1),
            'raw_alpha': raw_signed_alpha[:, None],
            'z_alpha': (cls_probability - cont_probability)[:, None],
            'reliability': semantic_probability.max(-1, keepdim=True).values,
            'reliability_gate': effective_gate[:, None],
            'reliability_cls': cls_probability[:, None],
            'reliability_cont': cont_probability[:, None],
            'reliability_base': keep_probability[:, None],
            'intervention_gate': effective_gate[:, None],
            'gate_logit': best_utility[:, None],
            'regional_intervention_gate': effective_gate[:, None].expand(
                -1, height * width),
            'regional_gate_logit': best_utility[:, None].expand(
                -1, height * width),
            'delta': effective_correction.flatten(1),
            'factor': factor.flatten(1),
            'calibrated_score': calibrated,
            'windowed_calibrated_score': windowed_calibrated,
            'global_calibrated_score': calibrated,
            'windowed_global_calibrated_score': windowed_calibrated,
            'alpha_global': raw_signed_alpha[:, None],
            'alpha_final': effective_correction,
            'av1_action_probability': action_probability,
            'av1_raw_hard_action': raw_hard_action,
            'av1_hard_action': effective_action,
            'av1_hard_policy': hard_policy,
            'av1_best_utility': best_utility,
            'av1_utility_gate': utility_gate,
            'av1_effective_gate': effective_gate,
            'av1_selected_full_correction': selected_correction,
            'av1_predicted_correction_map': predicted_correction,
            'av1_effective_correction_map': effective_correction,
            'av1_semantic_probability': semantic_probability,
            'av2_current_action_utility': current_utility,
            'av2_temporal_action_utility': temporal_utility,
            'av2_memory_used': memory_used,
            'av2_current_hard_action': current_action,
            'av2_temporal_action_changed': raw_hard_action.ne(current_action),
            'av2_current_support': current_support,
            'av2_current_supported': current_supported,
            'av2_group_confirmed': group_confirmed,
            'av2_policy_confirmed': should_intervene,
            'av2_gate_utility': gate_utility,
        })
        return output

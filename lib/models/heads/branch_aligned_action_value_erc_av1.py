"""Unified action-value evidence calibration (ERC-AV1).

ERC-AV1 keeps FER-v7's deployable response-calibration family, but removes
the independently predicted permission/direction/scope/magnitude decisions.
Instead, one head predicts the utility of every complete non-KEEP action.
KEEP has fixed zero utility.  The best positive utility therefore determines
both whether to intervene and which complete action to apply.

The module remains response-level: bbox geometry is neither an input nor an
output.  Candidate IoU is used only by the detached training oracle.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .branch_aligned_factorized_reachability_erc_v6 import (
    BranchAlignedFactorizedReachabilityERCV6,
)


class BranchAlignedActionValueERCAV1(
        BranchAlignedFactorizedReachabilityERCV6):
    """Predict the signed utility of each complete C/T calibration action."""

    DESIGN = 'branch_aligned_action_value_erc_av1'

    def __init__(
        self, feat_sz, dim, hidden=64,
        magnitudes=(0.10, 0.20, 0.40, 0.60, 0.80),
        max_alpha=0.80, d_max=4.0, eps=1e-6,
        action_temperature=1.0, response_temperature=1.0,
        utility_temperature=0.25, utility_gate_scale=0.50,
        detach_features=True, apply_vl_tracking_only=True,
        inference_mode='FULL_ERC', force_off=False,
    ):
        if float(utility_temperature) <= 0.0:
            raise ValueError('ERC-AV1 utility temperature must be positive')
        if float(utility_gate_scale) <= 0.0:
            raise ValueError('ERC-AV1 utility gate scale must be positive')
        super().__init__(
            feat_sz=feat_sz, dim=dim, hidden=hidden,
            magnitudes=magnitudes, max_alpha=max_alpha, d_max=d_max,
            eps=eps, action_temperature=action_temperature,
            response_temperature=response_temperature,
            detach_features=detach_features,
            apply_vl_tracking_only=apply_vl_tracking_only,
            inference_mode=inference_mode, force_off=force_off,
        )

        # The factorized heads must not remain in the optimizer.  A single
        # utility head predicts all 15 complete non-KEEP actions.
        del self.permission_head
        del self.direction_head
        del self.scope_head
        del self.magnitude_head
        decision_dim = 3 * hidden + 8
        number_nonkeep = int(self.v6_action_direction.numel()) - 1
        self.utility_head = self._head(
            decision_dim, hidden, number_nonkeep)
        nn.init.zeros_(self.utility_head[-1].weight)
        nn.init.zeros_(self.utility_head[-1].bias)
        self.utility_temperature = float(utility_temperature)
        self.utility_gate_scale = float(utility_gate_scale)
        self.design = self.DESIGN

    def _decision_spatial_input(
        self, stable, visual_response, language_response,
        batch, height, width,
    ):
        """Build the legacy AV1 decision tensor.

        Keeping this in a method makes the action-value evidence contract
        replaceable without changing the reachable actions or the detached
        IoU oracle.  Existing AV1/AV2/AV3 designs retain byte-for-byte the
        same feature construction through this default implementation.
        """
        support_float = stable['support'].to(stable['base'].dtype)
        reference_context = 0.5 * (visual_response + language_response)
        return torch.stack([
            stable['log_c'], stable['log_t'], stable['log_s'],
            stable['difference'] / self.d_max,
            visual_response.reshape(batch, height, width).clamp(-4, 4) / 4,
            language_response.reshape(batch, height, width).clamp(-4, 4) / 4,
            reference_context.reshape(batch, height, width).clamp(-4, 4) / 4,
            stable['window'],
        ], dim=1) * support_float[:, None]

    def forward(
        self, cls_response, cont_response, search,
        visual_reference, language_reference, flag,
    ):
        if cls_response.ndim != 3 or cls_response.shape != cont_response.shape:
            raise ValueError('ERC-AV1 expects matching C/T maps [B,H,W]')
        batch, height, width = cls_response.shape
        if (height, width) != (self.feat_sz, self.feat_sz):
            raise ValueError('ERC-AV1 response size mismatch')
        if search.shape != (batch, height * width, self.dim):
            raise ValueError('ERC-AV1 search token shape mismatch')

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
        spatial_input = self._decision_spatial_input(
            stable, visual_response, language_response,
            batch, height, width,
        )
        encoder_input = (
            spatial_input.detach() if self.detach_features else spatial_input)
        spatial_features = self.encoder(encoder_input) * support_float[:, None]
        branch_statistics = self._branch_statistics(stable)
        decision_features = torch.cat([
            self._learned_statistics(spatial_features, support),
            (branch_statistics.detach()
             if self.detach_features else branch_statistics),
        ], dim=-1)

        nonkeep_utility = self.utility_head(decision_features)
        keep_utility = nonkeep_utility.new_zeros(batch, 1)
        action_utility = torch.cat([keep_utility, nonkeep_utility], dim=-1)
        action_probability = F.softmax(
            action_utility / self.utility_temperature, dim=-1)
        best_utility, best_nonkeep = nonkeep_utility.max(-1)
        should_intervene = best_utility.gt(0.0)
        raw_hard_action = torch.where(
            should_intervene, best_nonkeep + 1,
            torch.zeros_like(best_nonkeep))
        hard_policy = F.one_hot(
            raw_hard_action, action_utility.shape[-1]
        ).to(action_utility.dtype)
        utility_gate = (
            F.relu(best_utility) / self.utility_gate_scale
        ).clamp(max=1.0)

        correction_basis = (
            stable['difference'].detach()
            if self.detach_features else stable['difference'])
        action_corrections, regional_t = self._action_correction_maps(
            correction_basis, support)
        rows = torch.arange(batch, device=action_utility.device)
        selected_full_correction = action_corrections[rows, raw_hard_action]
        predicted_correction = utility_gate[:, None, None] * (
            selected_full_correction)
        selected_signed_alpha = self.v6_action_signed_alpha.to(
            action_utility)[raw_hard_action]
        raw_signed_alpha = utility_gate * selected_signed_alpha

        if self.apply_vl_tracking_only:
            active = flag.reshape(batch).eq(2)
        else:
            active = torch.ones(
                batch, dtype=torch.bool, device=flag.device)
        enabled = active & (not self.force_off) & (
            self.inference_mode == 'FULL_ERC')
        effective_correction = torch.where(
            enabled[:, None, None], predicted_correction,
            torch.zeros_like(predicted_correction))
        effective_action = torch.where(
            enabled, raw_hard_action, torch.zeros_like(raw_hard_action))
        effective_gate = torch.where(
            enabled, utility_gate, torch.zeros_like(utility_gate))

        factor = effective_correction.exp()
        base = stable['base'].flatten(1)
        windowed_base = stable['windowed_base'].flatten(1)
        calibrated = base * factor.flatten(1)
        windowed_calibrated = windowed_base * factor.flatten(1)

        nonkeep_probability = F.softmax(
            nonkeep_utility / self.utility_temperature, dim=-1)
        direction = self.v6_action_direction[1:].to(nonkeep_utility.device)
        cls_probability = (
            nonkeep_probability[:, direction.eq(1)].sum(-1) * utility_gate)
        cont_probability = (
            nonkeep_probability[:, direction.eq(2)].sum(-1) * utility_gate)
        keep_probability = 1.0 - utility_gate
        semantic_probability = torch.stack([
            keep_probability, cls_probability, cont_probability,
        ], dim=-1)

        return {
            'features': decision_features,
            'cls_response': cls_response.flatten(1),
            'cont_response': cont_response.flatten(1),
            'windowed_cls_score': stable['windowed_cls'].flatten(1),
            'windowed_cont_score': stable['windowed_cont'].flatten(1),
            'alpha': effective_correction.flatten(1),
            'raw_alpha': raw_signed_alpha[:, None],
            'z_alpha': (
                cls_probability - cont_probability)[:, None],
            'reliability': semantic_probability.max(-1, keepdim=True).values,
            'reliability_gate': effective_gate[:, None],
            'reliability_cls': cls_probability[:, None],
            'reliability_cont': cont_probability[:, None],
            'reliability_base': keep_probability[:, None],
            'intervention_gate': effective_gate[:, None],
            'gate_logit': best_utility[:, None],
            'regional_intervention_gate': (
                effective_gate[:, None].expand(-1, height * width)),
            'regional_gate_logit': (
                best_utility[:, None].expand(-1, height * width)),
            'delta': effective_correction.flatten(1),
            'log_ratio': correction_basis.flatten(1),
            'factor': factor.flatten(1),
            'base_score': base,
            'windowed_base_score': windowed_base,
            'calibrated_score': calibrated,
            'windowed_calibrated_score': windowed_calibrated,
            'global_calibrated_score': calibrated,
            'windowed_global_calibrated_score': windowed_calibrated,
            'alpha_global': raw_signed_alpha[:, None],
            'alpha_region': regional_t,
            'alpha_final': effective_correction,
            'active_mask': active,
            # ERC-AV1 loss inputs and diagnostics.
            'av1_action_utility': action_utility,
            'av1_nonkeep_utility': nonkeep_utility,
            'av1_action_probability': action_probability,
            'av1_raw_hard_action': raw_hard_action,
            'av1_hard_action': effective_action,
            'av1_hard_policy': hard_policy,
            'av1_best_utility': best_utility,
            'av1_utility_gate': utility_gate,
            'av1_effective_gate': effective_gate,
            'av1_action_correction_maps': action_corrections,
            'av1_selected_full_correction': selected_full_correction,
            'av1_predicted_correction_map': predicted_correction,
            'av1_effective_correction_map': effective_correction,
            'av1_support_mask': support,
            'av1_log_base': stable['windowed_base'].clamp_min(
                self.eps).log().masked_fill(
                    ~support, torch.finfo(cls_response.dtype).min),
            'av1_action_direction': self.v6_action_direction,
            'av1_action_scope': self.v6_action_scope,
            'av1_action_magnitude_index': self.v6_action_magnitude_index,
            'av1_action_signed_alpha': self.v6_action_signed_alpha,
            'av1_magnitudes': self.reachability_magnitudes,
            'av1_spatial_features': spatial_features,
            'av1_branch_statistics': branch_statistics,
            'av1_semantic_probability': semantic_probability,
            'inference_mode': self.inference_mode,
        }

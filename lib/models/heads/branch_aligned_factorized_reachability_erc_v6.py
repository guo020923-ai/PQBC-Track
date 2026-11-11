"""Reachability-aligned factorized evidence calibration (FER-v6).

The real-response audit showed useful actions in three families only:
GLOBAL_C, GLOBAL_T and REGIONAL_T.  FER-v6 therefore factorizes permission,
C/T direction, T-scope and the smallest sufficient magnitude.  The regional
mask is deterministic evidence (smoothed negative C/T log-ratio), not a free
learned pixel gate.  No bbox geometry or candidate quality enters ERC.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .branch_aligned_tristate_multiaction_erc_v5 import (
    BranchAlignedTriStateMultiActionERCV5,
)


class BranchAlignedFactorizedReachabilityERCV6(
        BranchAlignedTriStateMultiActionERCV5):
    DESIGN = 'branch_aligned_factorized_reachability_erc_v6'

    def __init__(
        self, feat_sz, dim, hidden=64,
        magnitudes=(0.10, 0.20, 0.40, 0.60, 0.80),
        max_alpha=0.80, d_max=4.0, eps=1e-6,
        action_temperature=1.0, response_temperature=1.0,
        detach_features=True, apply_vl_tracking_only=True,
        inference_mode='FULL_ERC', force_off=False,
    ):
        magnitudes = tuple(float(value) for value in magnitudes)
        if not magnitudes or any(value <= 0 for value in magnitudes):
            raise ValueError('FER-v6 magnitudes must be positive')
        if tuple(sorted(set(magnitudes))) != magnitudes:
            raise ValueError('FER-v6 magnitudes must be unique and ascending')
        if magnitudes[-1] > float(max_alpha) + 1e-8:
            raise ValueError('FER-v6 magnitude exceeds MAX_ALPHA')
        super().__init__(
            feat_sz=feat_sz, dim=dim, hidden=hidden,
            action_alpha_values=(0.0, max_alpha, -max_alpha),
            max_alpha=max_alpha, d_max=d_max, eps=eps,
            action_temperature=action_temperature,
            response_temperature=response_temperature,
            detach_features=detach_features,
            apply_vl_tracking_only=apply_vl_tracking_only,
            inference_mode=inference_mode, force_off=force_off,
        )
        del self.action_head
        self.register_buffer(
            'reachability_magnitudes', torch.tensor(magnitudes))
        number = len(magnitudes)
        # Action order: KEEP, GLOBAL_C, GLOBAL_T, REGIONAL_T.
        direction = [0] + [1] * number + [2] * number + [2] * number
        scope = [0] + [0] * number + [0] * number + [1] * number
        magnitude_index = [-1] + list(range(number)) * 3
        signed_alpha = (
            [0.0] + list(magnitudes)
            + [-value for value in magnitudes]
            + [-value for value in magnitudes])
        self.register_buffer('v6_action_direction', torch.tensor(direction))
        self.register_buffer('v6_action_scope', torch.tensor(scope))
        self.register_buffer(
            'v6_action_magnitude_index', torch.tensor(magnitude_index))
        self.register_buffer('v6_action_signed_alpha', torch.tensor(signed_alpha))
        self.action_class_ema = torch.full(
            (1 + 3 * number,), 1.0 / (1 + 3 * number))

        decision_dim = 3 * hidden + 8
        self.permission_head = self._head(decision_dim, hidden, 1)
        self.direction_head = self._head(decision_dim, hidden, 2)
        self.scope_head = self._head(decision_dim, hidden, 2)
        self.magnitude_head = self._head(decision_dim, hidden, number)
        for head in (
                self.permission_head, self.direction_head,
                self.scope_head, self.magnitude_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    @staticmethod
    def _head(input_dim, hidden, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, output_dim))

    @staticmethod
    def _regional_t_mask(difference, support):
        mask = F.avg_pool2d(
            F.relu(-difference)[:, None], 3, stride=1, padding=1).squeeze(1)
        mask = mask * support.to(mask.dtype)
        return mask / mask.flatten(1).amax(-1).clamp_min(1e-6)[:, None, None]

    def _action_correction_maps(self, difference, support):
        maps = [torch.zeros_like(difference)]
        regional_t = self._regional_t_mask(difference, support)
        for magnitude in self.reachability_magnitudes.to(difference):
            maps.append(magnitude * difference)
        for magnitude in self.reachability_magnitudes.to(difference):
            maps.append(-magnitude * difference)
        for magnitude in self.reachability_magnitudes.to(difference):
            maps.append(-magnitude * regional_t * difference)
        return torch.stack(maps, dim=1), regional_t

    def forward(
        self, cls_response, cont_response, search,
        visual_reference, language_reference, flag,
    ):
        if cls_response.ndim != 3 or cls_response.shape != cont_response.shape:
            raise ValueError('FER-v6 expects matching C/T maps [B,H,W]')
        batch, height, width = cls_response.shape
        if (height, width) != (self.feat_sz, self.feat_sz):
            raise ValueError('FER-v6 response size mismatch')
        if search.shape != (batch, height * width, self.dim):
            raise ValueError('FER-v6 search token shape mismatch')
        visual_reference = self._one_reference(
            visual_reference, batch, self.dim, 'visual')
        language_reference = self._one_reference(
            language_reference, batch, self.dim, 'language')
        search_evidence = search.detach() if self.detach_features else search
        visual_evidence = (
            visual_reference.detach() if self.detach_features else visual_reference)
        language_evidence = (
            language_reference.detach() if self.detach_features else language_reference)
        search_unit = F.normalize(search_evidence, dim=-1)
        visual_response = self._standardize((
            search_unit * F.normalize(visual_evidence, dim=-1)[:, None]).sum(-1))
        language_response = self._standardize((
            search_unit * F.normalize(language_evidence, dim=-1)[:, None]).sum(-1))

        stable = self._stable_response_channels(cls_response, cont_response)
        support = stable['support']
        support_float = support.to(cls_response.dtype)
        reference_context = 0.5 * (visual_response + language_response)
        spatial_input = torch.stack([
            stable['log_c'], stable['log_t'], stable['log_s'],
            stable['difference'] / self.d_max,
            visual_response.reshape(batch, height, width).clamp(-4, 4) / 4,
            language_response.reshape(batch, height, width).clamp(-4, 4) / 4,
            reference_context.reshape(batch, height, width).clamp(-4, 4) / 4,
            stable['window'],
        ], dim=1) * support_float[:, None]
        encoder_input = spatial_input.detach() if self.detach_features else spatial_input
        spatial_features = self.encoder(encoder_input) * support_float[:, None]
        branch_statistics = self._branch_statistics(stable)
        decision_features = torch.cat([
            self._learned_statistics(spatial_features, support),
            branch_statistics.detach() if self.detach_features else branch_statistics,
        ], dim=-1)

        permission_logit = self.permission_head(decision_features).squeeze(-1)
        direction_logits = self.direction_head(decision_features)
        scope_logits = self.scope_head(decision_features)
        magnitude_logits = self.magnitude_head(decision_features)
        permission_probability = torch.sigmoid(
            permission_logit / self.action_temperature)
        direction_probability = F.softmax(
            direction_logits / self.action_temperature, -1)
        scope_probability = F.softmax(scope_logits / self.action_temperature, -1)
        magnitude_probability = F.softmax(
            magnitude_logits / self.action_temperature, -1)

        # Factorized probability over the exact deployable action set.
        keep_probability = 1.0 - permission_probability
        p_cls = permission_probability[:, None] * direction_probability[:, :1]
        p_cont = permission_probability[:, None] * direction_probability[:, 1:]
        joint_probability = torch.cat([
            keep_probability[:, None],
            p_cls * magnitude_probability,
            p_cont * scope_probability[:, :1] * magnitude_probability,
            p_cont * scope_probability[:, 1:] * magnitude_probability,
        ], dim=-1)

        hard_permission = permission_logit.gt(0)
        hard_direction = direction_logits.argmax(-1)
        hard_scope = scope_logits.argmax(-1)
        hard_magnitude = magnitude_logits.argmax(-1)
        number = self.reachability_magnitudes.numel()
        hard_action = torch.zeros_like(hard_magnitude)
        cls_action = 1 + hard_magnitude
        global_t_action = 1 + number + hard_magnitude
        regional_t_action = 1 + 2 * number + hard_magnitude
        t_action = torch.where(hard_scope.eq(0), global_t_action, regional_t_action)
        intervene_action = torch.where(hard_direction.eq(0), cls_action, t_action)
        hard_action = torch.where(hard_permission, intervene_action, hard_action)
        hard_policy = F.one_hot(
            hard_action, joint_probability.shape[-1]).to(joint_probability.dtype)
        action_policy = (
            hard_policy + joint_probability - joint_probability.detach()
            if self.training else hard_policy)

        correction_basis = (
            stable['difference'].detach()
            if self.detach_features else stable['difference'])
        action_corrections, regional_t = self._action_correction_maps(
            correction_basis, support)
        predicted_correction = (
            action_policy[:, :, None, None] * action_corrections).sum(1)
        raw_signed_alpha = (
            action_policy * self.v6_action_signed_alpha.to(action_policy)[None]
        ).sum(-1)
        if self.apply_vl_tracking_only:
            active = flag.reshape(batch).eq(2)
        else:
            active = torch.ones(batch, dtype=torch.bool, device=flag.device)
        enabled = active & (not self.force_off) & (
            self.inference_mode == 'FULL_ERC')
        effective_correction = torch.where(
            enabled[:, None, None], predicted_correction,
            torch.zeros_like(predicted_correction))
        effective_action = torch.where(
            enabled, hard_action, torch.zeros_like(hard_action))
        factor = effective_correction.exp()
        base = stable['base'].flatten(1)
        windowed_base = stable['windowed_base'].flatten(1)
        calibrated = base * factor.flatten(1)
        windowed_calibrated = windowed_base * factor.flatten(1)
        semantic_probability = torch.cat([
            keep_probability[:, None], p_cls, p_cont], -1)

        return {
            'features': decision_features,
            'cls_response': cls_response.flatten(1),
            'cont_response': cont_response.flatten(1),
            'windowed_cls_score': stable['windowed_cls'].flatten(1),
            'windowed_cont_score': stable['windowed_cont'].flatten(1),
            'alpha': effective_correction.flatten(1),
            'raw_alpha': raw_signed_alpha[:, None],
            'z_alpha': direction_logits[:, :1] - direction_logits[:, 1:],
            'reliability': semantic_probability.max(-1, keepdim=True).values,
            'reliability_gate': permission_probability[:, None],
            'reliability_cls': p_cls,
            'reliability_cont': p_cont,
            'reliability_base': keep_probability[:, None],
            'intervention_gate': permission_probability[:, None],
            'gate_logit': permission_logit[:, None],
            'regional_intervention_gate': (
                permission_probability * direction_probability[:, 1]
                * scope_probability[:, 1])[:, None].expand(-1, height * width),
            'regional_gate_logit': scope_logits[:, 1:2].expand(-1, height * width),
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
            'v6_permission_logit': permission_logit,
            'v6_permission_probability': permission_probability,
            'v6_direction_logits': direction_logits,
            'v6_direction_probability': direction_probability,
            'v6_scope_logits': scope_logits,
            'v6_scope_probability': scope_probability,
            'v6_magnitude_logits': magnitude_logits,
            'v6_magnitude_probability': magnitude_probability,
            'v6_joint_probability': joint_probability,
            'v6_hard_action': effective_action,
            'v6_raw_hard_action': hard_action,
            'v6_hard_policy': hard_policy,
            'v6_action_policy': action_policy,
            'v6_action_correction_maps': action_corrections,
            'v6_predicted_correction_map': predicted_correction,
            'v6_effective_correction_map': effective_correction,
            'v6_correction_basis': correction_basis,
            'v6_support_mask': support,
            'v6_log_base': stable['windowed_base'].clamp_min(
                self.eps).log().masked_fill(
                    ~support, torch.finfo(cls_response.dtype).min),
            'v6_regional_t_mask': regional_t,
            'v6_action_direction': self.v6_action_direction,
            'v6_action_scope': self.v6_action_scope,
            'v6_action_magnitude_index': self.v6_action_magnitude_index,
            'v6_action_signed_alpha': self.v6_action_signed_alpha,
            'v6_magnitudes': self.reachability_magnitudes,
            'v6_action_class_ema': self.action_class_ema,
            'v6_semantic_probability': semantic_probability,
            'v6_spatial_features': spatial_features,
            'v6_branch_statistics': branch_statistics,
            'inference_mode': self.inference_mode,
        }

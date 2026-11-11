"""Cross-domain global-regional evidence reliability calibration.

This ERC predicts explicit classification/contrastive reliabilities and uses
their difference to control a continuous, bounded calibration of the original
UVLTrack C*T response.  It remains response-level: candidate boxes and bbox
geometry are not model inputs or outputs.  A causal EMA is applied to the same
control variables during paired training and single-sequence inference.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .branch_aligned_factorized_reachability_erc_v6 import (
    BranchAlignedFactorizedReachabilityERCV6,
)


class CrossDomainGlobalRegionalERCV1(
        BranchAlignedFactorizedReachabilityERCV6):
    """Explicit C/T reliability with continuous global-regional calibration."""

    DESIGN = 'cross_domain_global_regional_reliability_erc_v1'

    def __init__(
        self, feat_sz, dim, hidden=64,
        magnitudes=(0.10, 0.20, 0.40, 0.60, 0.80),
        max_alpha=0.80, d_max=4.0, eps=1e-6,
        action_temperature=1.0, response_temperature=1.0,
        detach_features=True, apply_vl_tracking_only=True,
        inference_mode='FULL_ERC', force_off=False,
        temporal_ema=0.60, enable_temporal_memory=True,
        direction_scale=3.0,
    ):
        if not 0.0 <= float(temporal_ema) < 1.0:
            raise ValueError('CD-ERC temporal EMA must be in [0, 1)')
        if float(direction_scale) <= 0.0:
            raise ValueError('CD-ERC direction scale must be positive')
        super().__init__(
            feat_sz=feat_sz, dim=dim, hidden=hidden,
            magnitudes=magnitudes, max_alpha=max_alpha, d_max=d_max,
            eps=eps, action_temperature=action_temperature,
            response_temperature=response_temperature,
            detach_features=detach_features,
            apply_vl_tracking_only=apply_vl_tracking_only,
            inference_mode=inference_mode, force_off=force_off,
        )

        # Replace the discrete factorized action policy with explicit and
        # interpretable continuous controls.
        del self.permission_head
        del self.direction_head
        del self.scope_head
        del self.magnitude_head
        decision_dim = 3 * hidden + 8
        self.reliability_head = self._head(decision_dim, hidden, 2)
        self.gate_head = self._head(decision_dim, hidden, 1)
        self.strength_head = self._head(decision_dim, hidden, 1)
        self.regional_mix_head = self._head(decision_dim, hidden, 1)
        for head in (
                self.reliability_head, self.gate_head,
                self.strength_head, self.regional_mix_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

        self.temporal_ema = float(temporal_ema)
        self.enable_temporal_memory = bool(enable_temporal_memory)
        self.direction_scale = float(direction_scale)
        self._temporal_control_memory = None
        self.design = self.DESIGN

    def reset_temporal_state(self):
        self._temporal_control_memory = None

    @staticmethod
    def _regional_disagreement_mask(difference, support):
        mask = F.avg_pool2d(
            difference.abs()[:, None], 3, stride=1, padding=1).squeeze(1)
        mask = mask * support.to(mask.dtype)
        maximum = mask.flatten(1).amax(-1).clamp_min(1e-6)
        return mask / maximum[:, None, None]

    def _temporal_controls(self, current, active):
        """Use the same causal control smoothing in paired data and tracking."""
        batch = current.shape[0]
        memory_used = torch.zeros(
            batch, dtype=torch.bool, device=current.device)

        # UVLTrack flattens two search views as [view, batch].  Applying the
        # causal recurrence here makes training/validation match deployment.
        if batch > 1 and batch % 2 == 0:
            half = batch // 2
            first, second = current[:half], current[half:]
            pair_active = active[:half] & active[half:]
            smoothed_second = (
                self.temporal_ema * first
                + (1.0 - self.temporal_ema) * second
            )
            smoothed_second = torch.where(
                pair_active[:, None], smoothed_second, second)
            memory_used[half:] = pair_active
            return torch.cat([first, smoothed_second], dim=0), memory_used

        use_memory = (
            self.enable_temporal_memory
            and not self.training
            and batch == 1
            and bool(active[0].item())
            and not self.force_off
            and self.inference_mode == 'FULL_ERC'
        )
        if not use_memory:
            return current, memory_used
        if self._temporal_control_memory is None:
            smoothed = current
        else:
            previous = self._temporal_control_memory.to(current)
            smoothed = (
                self.temporal_ema * previous
                + (1.0 - self.temporal_ema) * current
            )
            memory_used.fill_(True)
        self._temporal_control_memory = smoothed.detach()
        return smoothed, memory_used

    def forward(
        self, cls_response, cont_response, search,
        visual_reference, language_reference, flag,
    ):
        if cls_response.ndim != 3 or cls_response.shape != cont_response.shape:
            raise ValueError('CD-ERC expects matching C/T maps [B,H,W]')
        batch, height, width = cls_response.shape
        if (height, width) != (self.feat_sz, self.feat_sz):
            raise ValueError('CD-ERC response size mismatch')
        if search.shape != (batch, height * width, self.dim):
            raise ValueError('CD-ERC search token shape mismatch')

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
            search_unit
            * F.normalize(visual_evidence, dim=-1)[:, None]
        ).sum(-1))
        language_response = self._standardize((
            search_unit
            * F.normalize(language_evidence, dim=-1)[:, None]
        ).sum(-1))

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
        encoder_input = (
            spatial_input.detach() if self.detach_features else spatial_input)
        spatial_features = self.encoder(encoder_input) * support_float[:, None]
        branch_statistics = self._branch_statistics(stable)
        decision_features = torch.cat([
            self._learned_statistics(spatial_features, support),
            (branch_statistics.detach()
             if self.detach_features else branch_statistics),
        ], dim=-1)

        reliability_logits = self.reliability_head(decision_features)
        gate_logit = self.gate_head(decision_features).squeeze(-1)
        strength_logit = self.strength_head(decision_features).squeeze(-1)
        regional_mix_logit = self.regional_mix_head(
            decision_features).squeeze(-1)
        current_controls = torch.cat([
            torch.sigmoid(reliability_logits),
            torch.sigmoid(gate_logit)[:, None],
            torch.sigmoid(strength_logit)[:, None],
            torch.sigmoid(regional_mix_logit)[:, None],
        ], dim=-1)

        if self.apply_vl_tracking_only:
            active = flag.reshape(batch).eq(2)
        else:
            active = torch.ones(
                batch, dtype=torch.bool, device=flag.device)
        temporal_controls, memory_used = self._temporal_controls(
            current_controls, active)
        reliability = temporal_controls[:, :2]
        gate = temporal_controls[:, 2]
        strength = temporal_controls[:, 3]
        regional_mix = temporal_controls[:, 4]
        direction = torch.tanh(
            self.direction_scale
            * (reliability[:, 0] - reliability[:, 1]))
        alpha_total = self.max_alpha * gate * strength * direction
        alpha_global = alpha_total * (1.0 - regional_mix)
        alpha_regional = alpha_total * regional_mix

        difference = (
            stable['difference'].detach()
            if self.detach_features else stable['difference'])
        regional_mask = self._regional_disagreement_mask(difference, support)
        raw_correction = (
            alpha_global[:, None, None]
            + alpha_regional[:, None, None] * regional_mask
        ) * difference
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
        effective_gate = torch.where(enabled, gate, torch.zeros_like(gate))
        effective_alpha_total = torch.where(
            enabled, alpha_total, torch.zeros_like(alpha_total))

        return {
            'features': decision_features,
            'cls_response': cls_response.flatten(1),
            'cont_response': cont_response.flatten(1),
            'windowed_cls_score': stable['windowed_cls'].flatten(1),
            'windowed_cont_score': stable['windowed_cont'].flatten(1),
            'alpha': correction.flatten(1),
            'raw_alpha': effective_alpha_total[:, None],
            'z_alpha': direction[:, None],
            'reliability': reliability.max(-1, keepdim=True).values,
            'reliability_gate': effective_gate[:, None],
            'reliability_cls': reliability[:, :1],
            'reliability_cont': reliability[:, 1:],
            'reliability_base': 1.0 - effective_gate[:, None],
            'intervention_gate': effective_gate[:, None],
            'gate_logit': gate_logit[:, None],
            'regional_intervention_gate': (
                effective_gate[:, None] * regional_mask.flatten(1)),
            'regional_gate_logit': regional_mix_logit[:, None].expand(
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
            'alpha_global': alpha_global[:, None],
            'alpha_region': alpha_regional[:, None, None] * regional_mask,
            'alpha_final': correction,
            'active_mask': active,
            'cd1_reliability_logits': reliability_logits,
            'cd1_raw_controls': current_controls,
            'cd1_temporal_controls': temporal_controls,
            'cd1_raw_gate': current_controls[:, 2],
            'cd1_temporal_gate': gate,
            'cd1_strength': strength,
            'cd1_regional_mix': regional_mix,
            'cd1_direction': direction,
            'cd1_alpha_total': alpha_total,
            'cd1_alpha_global': alpha_global,
            'cd1_alpha_regional': alpha_regional,
            'cd1_difference': difference,
            'cd1_regional_mask': regional_mask,
            'cd1_raw_correction': raw_correction,
            'cd1_effective_correction': correction,
            'cd1_support_mask': support,
            'cd1_memory_used': memory_used,
            'inference_mode': self.inference_mode,
        }

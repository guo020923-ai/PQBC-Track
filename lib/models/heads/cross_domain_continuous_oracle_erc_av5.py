"""Cross-domain continuous global-regional evidence calibration (ERC-AV5).

Unlike the earlier CD1 model, branch reliability does not hard-code the sign
of the correction.  The model predicts two signed continuous coefficients and
an intervention gate; training searches both signs in the exact deployable
global/regional response family.  The module remains response-level and never
uses candidate boxes or bbox geometry.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .branch_aligned_factorized_reachability_erc_v6 import (
    BranchAlignedFactorizedReachabilityERCV6,
)


class CrossDomainContinuousOracleERCAV5(
        BranchAlignedFactorizedReachabilityERCV6):
    """Direct signed controls supervised by a bidirectional response oracle."""

    DESIGN = 'cross_domain_continuous_oracle_erc_av5'
    NUM_INPUT_CHANNELS = 10

    def __init__(
        self, feat_sz, dim, hidden=64,
        magnitudes=(0.10, 0.20, 0.40, 0.60, 0.80),
        max_alpha=0.80, d_max=4.0, eps=1e-6,
        action_temperature=1.0, response_temperature=1.0,
        detach_features=True, apply_vl_tracking_only=True,
        inference_mode='FULL_ERC', force_off=False,
        temporal_ema=0.60, enable_temporal_memory=True,
    ):
        if not 0.0 <= float(temporal_ema) < 1.0:
            raise ValueError('ERC-AV5 temporal EMA must be in [0, 1)')
        super().__init__(
            feat_sz=feat_sz, dim=dim, hidden=hidden,
            magnitudes=magnitudes, max_alpha=max_alpha, d_max=d_max,
            eps=eps, action_temperature=action_temperature,
            response_temperature=response_temperature,
            detach_features=detach_features,
            apply_vl_tracking_only=apply_vl_tracking_only,
            inference_mode=inference_mode, force_off=force_off,
        )

        del self.permission_head
        del self.direction_head
        del self.scope_head
        del self.magnitude_head
        # Preserve absolute response scale and relative response shape.  The
        # inherited eight-channel encoder is replaced intentionally.
        self.encoder = nn.Sequential(
            nn.Conv2d(self.NUM_INPUT_CHANNELS, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )
        decision_dim = 3 * hidden + 8
        self.reliability_head = self._head(decision_dim, hidden, 2)
        # Inputs include the two reliability probabilities and their signed
        # difference, but the output sign is not constrained by that difference.
        self.control_head = self._head(decision_dim + 3, hidden, 3)
        for head in (self.reliability_head, self.control_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

        self.temporal_ema = float(temporal_ema)
        self.enable_temporal_memory = bool(enable_temporal_memory)
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
        """Apply the same causal recurrence to train pairs and test frames."""
        batch = current.shape[0]
        memory_used = torch.zeros(
            batch, dtype=torch.bool, device=current.device)
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
            self.enable_temporal_memory and not self.training and batch == 1
            and bool(active[0].item()) and not self.force_off
            and self.inference_mode == 'FULL_ERC'
        )
        if not use_memory:
            return current, memory_used
        if self._temporal_control_memory is None:
            smoothed = current
        else:
            smoothed = (
                self.temporal_ema * self._temporal_control_memory.to(current)
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
            raise ValueError('ERC-AV5 expects matching C/T maps [B,H,W]')
        batch, height, width = cls_response.shape
        if (height, width) != (self.feat_sz, self.feat_sz):
            raise ValueError('ERC-AV5 response size mismatch')
        if search.shape != (batch, height * width, self.dim):
            raise ValueError('ERC-AV5 search token shape mismatch')

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
        cls_relative = self._standardize(
            cls_response.clamp_min(self.eps).log(), support)
        cont_relative = self._standardize(
            cont_response.clamp_min(self.eps).log(), support)
        reference_context = 0.5 * (visual_response + language_response)
        spatial_input = torch.stack([
            stable['log_c'], stable['log_t'], stable['log_s'],
            stable['difference'] / self.d_max,
            cls_relative.reshape(batch, height, width).clamp(-4, 4) / 4,
            cont_relative.reshape(batch, height, width).clamp(-4, 4) / 4,
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
        reliability = F.softmax(reliability_logits, dim=-1)
        control_input = torch.cat([
            decision_features, reliability,
            (reliability[:, :1] - reliability[:, 1:]),
        ], dim=-1)
        raw_control_logits = self.control_head(control_input)
        raw_gate = torch.sigmoid(raw_control_logits[:, 0])
        raw_alpha_global = self.max_alpha * torch.tanh(raw_control_logits[:, 1])
        raw_alpha_regional = self.max_alpha * torch.tanh(raw_control_logits[:, 2])
        current_controls = torch.stack([
            raw_gate,
            raw_alpha_global / self.max_alpha,
            raw_alpha_regional / self.max_alpha,
        ], dim=-1)

        if self.apply_vl_tracking_only:
            active = flag.reshape(batch).eq(2)
        else:
            active = torch.ones(batch, dtype=torch.bool, device=flag.device)
        temporal_controls, memory_used = self._temporal_controls(
            current_controls, active)
        gate = temporal_controls[:, 0].clamp(0.0, 1.0)
        alpha_global = (
            temporal_controls[:, 1].clamp(-1.0, 1.0) * self.max_alpha)
        alpha_regional = (
            temporal_controls[:, 2].clamp(-1.0, 1.0) * self.max_alpha)

        difference = (
            stable['difference'].detach()
            if self.detach_features else stable['difference'])
        regional_mask = self._regional_disagreement_mask(difference, support)
        coefficient = (
            (1.0 - regional_mask) * alpha_global[:, None, None]
            + regional_mask * alpha_regional[:, None, None]
        )
        raw_correction = gate[:, None, None] * coefficient * difference
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
        effective_global = torch.where(
            enabled, gate * alpha_global, torch.zeros_like(alpha_global))
        effective_regional = torch.where(
            enabled, gate * alpha_regional, torch.zeros_like(alpha_regional))

        return {
            'features': decision_features,
            'cls_response': cls_response.flatten(1),
            'cont_response': cont_response.flatten(1),
            'windowed_cls_score': stable['windowed_cls'].flatten(1),
            'windowed_cont_score': stable['windowed_cont'].flatten(1),
            'alpha': correction.flatten(1),
            'raw_alpha': effective_global[:, None],
            'z_alpha': (alpha_global - alpha_regional)[:, None],
            'reliability': reliability.max(-1, keepdim=True).values,
            'reliability_gate': effective_gate[:, None],
            'reliability_cls': reliability[:, :1],
            'reliability_cont': reliability[:, 1:],
            'reliability_base': 1.0 - effective_gate[:, None],
            'intervention_gate': effective_gate[:, None],
            'gate_logit': raw_control_logits[:, :1],
            'regional_intervention_gate': (
                effective_gate[:, None] * regional_mask.flatten(1)),
            'regional_gate_logit': raw_control_logits[:, 2:3].expand(
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
            'alpha_global': effective_global[:, None],
            'alpha_region': (
                effective_regional[:, None, None] * regional_mask),
            'alpha_final': correction,
            'active_mask': active,
            'cd5_reliability_logits': reliability_logits,
            'cd5_reliability': reliability,
            'cd5_raw_control_logits': raw_control_logits,
            'cd5_raw_controls': current_controls,
            'cd5_temporal_controls': temporal_controls,
            'cd5_raw_gate': raw_gate,
            'cd5_gate': gate,
            'cd5_raw_alpha_global': raw_alpha_global,
            'cd5_raw_alpha_regional': raw_alpha_regional,
            'cd5_alpha_global': alpha_global,
            'cd5_alpha_regional': alpha_regional,
            'cd5_effective_alpha_global': gate * alpha_global,
            'cd5_effective_alpha_regional': gate * alpha_regional,
            'cd5_difference': difference,
            'cd5_regional_mask': regional_mask,
            'cd5_coefficient': coefficient,
            'cd5_raw_correction': raw_correction,
            'cd5_effective_correction': correction,
            'cd5_support_mask': support,
            'cd5_memory_used': memory_used,
            'cd5_spatial_input': spatial_input,
            'inference_mode': self.inference_mode,
        }

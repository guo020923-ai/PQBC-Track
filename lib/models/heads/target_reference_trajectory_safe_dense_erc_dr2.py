"""Trajectory-safe target-reference direct dense ERC (DR2).

DR2 keeps DR1's response-level dense residual, but separates *what* spatial
correction is proposed from *whether* it is safe to deploy.  A lightweight
continuous gate is predicted from response uncertainty, cross-modal reference
agreement and the proposed peak displacement:

    log S_dr2(x) = log S0(x) + q_safe * Delta_raw(x).

At single-sequence inference only the scalar permission is accumulated over
time.  The dense map is never averaged across frames because its coordinates
belong to different search crops.  The current frame always retains veto power
through ``min(q_current, q_ema)``.  No candidate box, bbox geometry, dataset
identity or IoU is consumed by the module.
"""

import math

import torch
from torch import nn

from .target_reference_direct_dense_erc_dr1 import (
    TargetReferenceDirectDenseERCDR1,
)


class TargetReferenceTrajectorySafeDenseERCDR2(
        TargetReferenceDirectDenseERCDR1):
    """Direct dense evidence correction with causal safety permission."""

    DESIGN = 'target_reference_trajectory_safe_dense_erc_dr2'
    NUM_GATE_SCALARS = 11

    def __init__(
        self, *args, safety_gate_prior=0.20, temporal_ema=0.60,
        enable_temporal_memory=True, **kwargs,
    ):
        if not 0.0 < float(safety_gate_prior) < 1.0:
            raise ValueError('DR2 safety gate prior must be in (0, 1)')
        if not 0.0 <= float(temporal_ema) < 1.0:
            raise ValueError('DR2 temporal EMA must be in [0, 1)')
        super().__init__(*args, **kwargs)
        spatial_hidden = max(self.hidden // 2, 16)
        gate_hidden = max(spatial_hidden, 16)
        self.safety_head = nn.Sequential(
            nn.Linear(
                2 * spatial_hidden + self.NUM_GATE_SCALARS, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        nn.init.zeros_(self.safety_head[-1].weight)
        nn.init.constant_(
            self.safety_head[-1].bias,
            math.log(float(safety_gate_prior) / (1.0 - float(safety_gate_prior))),
        )
        self.temporal_ema = float(temporal_ema)
        self.enable_temporal_memory = bool(enable_temporal_memory)
        self._temporal_gate_memory = None
        self.design = self.DESIGN

    def reset_temporal_state(self):
        """Never carry permission evidence across independent sequences."""
        self._temporal_gate_memory = None

    @staticmethod
    def _gather(flattened, indices):
        rows = torch.arange(flattened.shape[0], device=flattened.device)
        return flattened[rows, indices]

    def _causal_safe_gate(self, current_gate, active):
        """Causal scalar smoothing with an unconditional current-frame veto."""
        use_memory = (
            self.enable_temporal_memory
            and not self.training
            and current_gate.shape[0] == 1
            and bool(active.reshape(-1)[0].item())
            and not self.force_off
            and self.inference_mode == 'FULL_ERC'
        )
        memory_used = torch.zeros_like(active)
        if not use_memory:
            return current_gate, current_gate, memory_used

        if self._temporal_gate_memory is None:
            # Start every trajectory conservatively.  Persistent current
            # evidence raises the permission smoothly over subsequent frames.
            temporal_gate = (1.0 - self.temporal_ema) * current_gate
        else:
            previous = self._temporal_gate_memory.to(
                device=current_gate.device, dtype=current_gate.dtype)
            temporal_gate = (
                self.temporal_ema * previous
                + (1.0 - self.temporal_ema) * current_gate
            )
            memory_used.fill_(True)
        self._temporal_gate_memory = temporal_gate.detach()
        effective_gate = torch.minimum(current_gate, temporal_gate)
        return effective_gate, temporal_gate, memory_used

    def forward(
        self, cls_response, cont_response, search,
        visual_reference, language_reference, prompts, flag,
    ):
        output = super().forward(
            cls_response, cont_response, search,
            visual_reference, language_reference, prompts, flag,
        )
        batch, height, width = cls_response.shape
        raw_correction = output['dr1_raw_correction']
        support = output['dr1_support_mask'].bool()
        spatial_features = output['dr1_spatial_features']
        windowed_base = output['windowed_base_score'].reshape(
            batch, height, width)
        if self.detach_features:
            windowed_base = windowed_base.detach()

        tiny = torch.finfo(windowed_base.dtype).tiny
        base_log = windowed_base.clamp_min(tiny).log().flatten(1)
        masked_base_log = base_log.masked_fill(
            ~support.flatten(1), torch.finfo(base_log.dtype).min)
        top_values, _ = masked_base_log.topk(k=2, dim=-1)
        baseline_margin = (top_values[:, 0] - top_values[:, 1]).tanh()
        base_probability = torch.softmax(masked_base_log, dim=-1)
        base_entropy = -(
            base_probability * base_probability.clamp_min(tiny).log()
        ).sum(-1) / math.log(float(height * width))

        baseline_index = masked_base_log.argmax(-1)
        proposal_log = masked_base_log + raw_correction.flatten(1)
        proposal_index = proposal_log.argmax(-1)
        rows = torch.arange(batch, device=cls_response.device)
        baseline_y = torch.div(baseline_index, width, rounding_mode='floor')
        baseline_x = baseline_index.remainder(width)
        proposal_y = torch.div(proposal_index, width, rounding_mode='floor')
        proposal_x = proposal_index.remainder(width)
        proposal_distance = torch.sqrt(
            (proposal_x - baseline_x).float().square()
            + (proposal_y - baseline_y).float().square()
        ) / max(math.sqrt(2.0) * float(max(height - 1, 1)), 1.0)

        cls_peak = cls_response.flatten(1).argmax(-1)
        cont_peak = cont_response.flatten(1).argmax(-1)
        cls_y = torch.div(cls_peak, width, rounding_mode='floor')
        cls_x = cls_peak.remainder(width)
        cont_y = torch.div(cont_peak, width, rounding_mode='floor')
        cont_x = cont_peak.remainder(width)
        branch_peak_distance = torch.sqrt(
            (cls_x - cont_x).float().square()
            + (cls_y - cont_y).float().square()
        ) / max(math.sqrt(2.0) * float(max(height - 1, 1)), 1.0)

        target_margin = output['dr1_target_margin'].flatten(1)
        visual_response = output['dr1_visual_response'].flatten(1)
        language_response = output['dr1_language_response'].flatten(1)
        reference_response = 0.5 * (visual_response + language_response)

        def proposal_minus_baseline(values):
            return values[rows, proposal_index] - values[rows, baseline_index]

        scalar_features = torch.stack([
            baseline_margin,
            base_entropy,
            branch_peak_distance,
            proposal_distance,
            raw_correction.abs().flatten(1).mean(-1)
            / max(self.max_alpha, self.eps),
            raw_correction.abs().flatten(1).amax(-1)
            / max(self.max_alpha, self.eps),
            self._gather(target_margin, baseline_index).clamp(-2.0, 2.0) / 2.0,
            proposal_minus_baseline(target_margin).clamp(-2.0, 2.0) / 2.0,
            proposal_minus_baseline(visual_response).clamp(-4.0, 4.0) / 4.0,
            proposal_minus_baseline(language_response).clamp(-4.0, 4.0) / 4.0,
            proposal_minus_baseline(reference_response).clamp(-4.0, 4.0) / 4.0,
        ], dim=-1).to(spatial_features.dtype)
        pooled = torch.cat([
            spatial_features.mean((-2, -1)),
            spatial_features.amax((-2, -1)),
            scalar_features,
        ], dim=-1)
        # Keep the "whether to intervene" learner separate from the dense
        # proposal generator.  Utility/no-harm losses still optimize both via
        # q * Delta, while gate supervision cannot reshape C/T or Delta merely
        # to make its own classification problem easier.
        gate_features = pooled.detach() if self.detach_features else pooled
        gate_logit = self.safety_head(gate_features).squeeze(-1)
        current_gate = torch.sigmoid(gate_logit)
        active = output['active_mask'].reshape(-1).bool()
        effective_gate, temporal_gate, memory_used = self._causal_safe_gate(
            current_gate, active)

        runtime_enabled = (
            not self.force_off and self.inference_mode == 'FULL_ERC')
        enabled = active if runtime_enabled else torch.zeros_like(active)
        gated_correction = raw_correction * effective_gate[:, None, None]
        correction = torch.where(
            enabled[:, None, None], gated_correction,
            torch.zeros_like(gated_correction),
        )
        factor = correction.exp()
        base = output['base_score']
        windowed_base_flat = output['windowed_base_score']
        calibrated = base * factor.flatten(1)
        windowed_calibrated = windowed_base_flat * factor.flatten(1)
        strength = correction.abs().flatten(1).mean(-1, keepdim=True)

        output.update({
            'features': spatial_features.mean((-2, -1)),
            'alpha': correction.flatten(1),
            'raw_alpha': raw_correction.flatten(1),
            'z_alpha': gate_logit[:, None],
            'reliability': 1.0 - effective_gate[:, None],
            'reliability_gate': effective_gate[:, None],
            'reliability_base': 1.0 - effective_gate[:, None],
            'intervention_gate': effective_gate[:, None],
            'gate_logit': gate_logit[:, None],
            'regional_intervention_gate': (
                effective_gate[:, None] * raw_correction.abs().flatten(1)
                / max(self.max_alpha, self.eps)
            ),
            'regional_gate_logit': raw_correction.flatten(1),
            'delta': correction.flatten(1),
            'factor': factor.flatten(1),
            'calibrated_score': calibrated,
            'windowed_calibrated_score': windowed_calibrated,
            'global_calibrated_score': calibrated,
            'windowed_global_calibrated_score': windowed_calibrated,
            'alpha_global': effective_gate[:, None],
            'alpha_region': correction,
            'alpha_final': correction,
            'dr1_correction': correction,
            'dr2_gate_features': pooled,
            'dr2_gate_logit': gate_logit,
            'dr2_current_gate': current_gate,
            'dr2_temporal_gate': temporal_gate,
            'dr2_effective_gate': effective_gate,
            'dr2_memory_used': memory_used,
            'dr2_raw_correction': raw_correction,
            'dr2_gated_correction': gated_correction,
            'dr2_correction': correction,
            'dr2_baseline_margin': baseline_margin,
            'dr2_base_entropy': base_entropy,
            'dr2_branch_peak_distance': branch_peak_distance,
            'dr2_proposal_distance': proposal_distance,
            'dr2_baseline_index': baseline_index,
            'dr2_proposal_index': proposal_index,
            'dr2_scalar_features': scalar_features,
            'dr2_correction_strength': strength.squeeze(-1),
        })
        return output

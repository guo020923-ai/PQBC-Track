"""Cross-domain normalized consensus ERC (ERC-AV4).

AV3 exposed a domain-transfer mismatch: its reachable correction maps were
per-frame normalized, while the utility encoder that decides whether to use
them still consumed fixed-range absolute log responses.  AV4 makes the
decision evidence entirely frame-relative and requires a temporally proposed
action to remain beneficial and semantically consistent on the current frame.

The method stays at the response/evidence level.  It does not use dataset
identity, dataset-specific thresholds, bbox geometry, or candidate quality at
inference.  The action set and detached training oracle are unchanged.
"""

import torch

from .cross_modal_consensus_residual_erc_av3 import (
    CrossModalConsensusResidualERCAV3,
)


class CrossDomainNormalizedConsensusERCAV4(
        CrossModalConsensusResidualERCAV3):
    """Scale-invariant AV3 policy with current-frame temporal confirmation."""

    DESIGN = 'cross_domain_normalized_consensus_erc_av4'

    def __init__(self, *args, decision_clip=4.0, **kwargs):
        if float(decision_clip) <= 0.0:
            raise ValueError('ERC-AV4 decision clip must be positive')
        # A stale EMA proposal is never allowed to survive on merely
        # non-catastrophic current evidence.  The exact proposed action must
        # still have positive utility, agree with the current action family,
        # and its strength is bounded by the weaker of current/temporal value.
        kwargs.update(
            require_current_positive=True,
            require_group_confirmation=True,
            min_current_temporal_gate=True,
            current_support_margin=0.0,
        )
        super().__init__(*args, **kwargs)
        self.decision_clip = float(decision_clip)
        self.domain_normalized_decision = True
        self.design = self.DESIGN

    def _temporal_action_group(self):
        # AV4 actions are KEEP / global consensus / regional consensus /
        # reference veto.  Magnitude changes inside one family are compatible.
        return self.av3_action_family

    def _decision_spatial_input(
        self, stable, visual_response, language_response,
        batch, height, width,
    ):
        support = stable['support']
        support_float = support.to(stable['base'].dtype)
        clip = self.decision_clip

        def relative(values):
            normalized = self._standardize(values, support).reshape(
                batch, height, width)
            return normalized.clamp(-clip, clip) / clip

        # Multiplicative response calibration or dataset-dependent confidence
        # offsets become additive constants in log space and disappear under
        # this per-frame supported standardization.
        cls_relative = relative(
            stable['windowed_cls'].clamp_min(self.eps).log())
        cont_relative = relative(
            stable['windowed_cont'].clamp_min(self.eps).log())
        base_relative = relative(
            stable['windowed_base'].clamp_min(self.eps).log())
        branch_difference = (0.5 * (
            cls_relative - cont_relative)).clamp(-1.0, 1.0)

        visual_relative = relative(
            visual_response.reshape(batch, height, width))
        language_relative = relative(
            language_response.reshape(batch, height, width))
        reference_relative = 0.5 * (
            visual_relative + language_relative)

        return torch.stack([
            cls_relative,
            cont_relative,
            base_relative,
            branch_difference,
            visual_relative,
            language_relative,
            reference_relative,
            stable['window'],
        ], dim=1) * support_float[:, None]

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        output.update({
            'av4_current_support': output['av2_current_support'],
            'av4_current_supported': output['av2_current_supported'],
            'av4_group_confirmed': output['av2_group_confirmed'],
            'av4_policy_confirmed': output['av2_policy_confirmed'],
            'av4_gate_utility': output['av2_gate_utility'],
        })
        return output

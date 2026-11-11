"""ERC-v17 selective abstention worthiness gate.

ERC-v17 changes only the intervention Gate used by ERC-v16.  The branch
direction ``beta``, regional gate ``G`` and response difference ``D`` remain
the ERC-v13 implementation.  Two response-only scalar heads now answer two
different questions:

    g: may ERC intervene on this frame at all?
    w: if intervention is permitted, how worthwhile is it?

The effective correction is

    S_ERC(x) = S0(x) * exp(g * w * beta * G(x) * D(x)).

``g`` is a strict binary gate at inference.  During training a straight-
through estimator retains the same binary forward value while supplying the
sigmoid gradient.  Permission logits are zero initialized and the decision
uses a strict ``p > 0.5`` comparison; the initial module therefore abstains
exactly and reproduces the official response without numerical drift.
"""

import torch
from torch import nn

from .response_intervention_gate_erc_v14 import (
    ResponseInterventionGateERCV14,
)


def selective_permission_target(
    advantage_raw,
    actionable,
    keep_margin=0.05,
    permission_margin=0.10,
):
    """Build detached permission labels and an ambiguity mask.

    A correction is actionable only when the response Oracle selects a
    different location from the baseline.  Non-actionable samples and samples
    at or below ``keep_margin`` are explicit negatives.  Actionable samples at
    or above ``permission_margin`` are positives; the narrow interval between
    the two margins is ignored by the permission classifier.
    """
    keep_margin = float(keep_margin)
    permission_margin = float(permission_margin)
    if keep_margin < 0.0:
        raise ValueError('ERC-v17 keep_margin must be non-negative')
    if permission_margin <= keep_margin:
        raise ValueError(
            'ERC-v17 permission_margin must be greater than keep_margin'
        )

    advantage_raw = torch.as_tensor(advantage_raw).detach()
    actionable = torch.as_tensor(
        actionable, dtype=torch.bool, device=advantage_raw.device
    ).detach()
    if actionable.shape != advantage_raw.shape:
        raise ValueError('ERC-v17 actionable mask shape mismatch')

    negative = (advantage_raw.le(keep_margin) | (~actionable)).detach()
    positive = (
        advantage_raw.ge(permission_margin) & actionable
    ).detach()
    supervision_mask = (negative | positive).detach()
    target = positive.to(dtype=advantage_raw.dtype).detach()
    return target, supervision_mask, positive, negative


class ResponseSelectiveAbstentionWorthinessGateERCV17(
    ResponseInterventionGateERCV14
):
    """ERC-v13 calibration with hard permission and soft worthiness."""

    DESIGN = 'response_selective_abstention_worthiness_gate_v17'

    def __init__(self, *args, **kwargs):
        # The inherited intervention head becomes the continuous worthiness
        # head and keeps the same prior/initialization as v16.
        super().__init__(*args, **kwargs)
        self.design = self.DESIGN
        self.permission_head = nn.Linear(
            self.beta_head.in_features, 1
        )
        nn.init.zeros_(self.permission_head.weight)
        nn.init.zeros_(self.permission_head.bias)

    @property
    def worthiness_head(self):
        """Semantic alias without registering a duplicate state-dict module."""
        return self.intervention_head

    def forward(self, cls_response, cont_response, flag=None):
        # v14 supplies the unchanged v13 beta/G/D path, the continuous scalar
        # head (used here as worthiness), and training-only anchor predictions.
        output = super().forward(cls_response, cont_response, flag=flag)
        decision_features = output['v13_decision_features']

        permission_logit = self.permission_head(decision_features)
        permission_probability = permission_logit.sigmoid()
        permission_hard = permission_probability.gt(0.5).to(
            dtype=permission_probability.dtype
        )
        if self.training:
            permission_gate = (
                permission_hard
                + permission_probability
                - permission_probability.detach()
            )
        else:
            permission_gate = permission_hard

        predicted_worthiness = output['v14_intervention_gate']
        enabled_mask = output['active_mask'] & (not self.force_off) & (
            self.inference_mode == 'FULL_ERC'
        )
        effective_permission = torch.where(
            enabled_mask[:, None], permission_gate,
            torch.zeros_like(permission_gate),
        )
        effective_worthiness = torch.where(
            enabled_mask[:, None], predicted_worthiness,
            torch.zeros_like(predicted_worthiness),
        )
        effective_gate = effective_permission * effective_worthiness

        predicted_beta = output['v13_beta']
        effective_beta = effective_gate * predicted_beta
        correction_map = (
            effective_beta[:, :, None]
            * output['v13_spatial_gate']
            * output['v13_d_map']
        )
        factor = correction_map.exp().flatten(1)
        base_score = output['base_score']
        windowed_base_score = output['windowed_base_score']
        calibrated_score = base_score * factor
        windowed_calibrated_score = windowed_base_score * factor

        # Generic and v14 aliases intentionally expose the *effective* Gate.
        # This preserves compatibility with existing inference/audit utilities
        # while preventing them from mistaking w alone for actual intervention.
        output.update({
            'features': decision_features,
            'alpha': correction_map.flatten(1),
            'raw_alpha': predicted_beta,
            'z_alpha': output['v13_beta_logit'],
            'reliability': effective_gate,
            'reliability_gate': effective_gate,
            'intervention_gate': effective_gate,
            'gate_logit': permission_logit,
            'gate_decision_features': decision_features,
            'delta': correction_map.flatten(1),
            'factor': factor,
            'calibrated_score': calibrated_score,
            'windowed_calibrated_score': windowed_calibrated_score,
            'global_calibrated_score': base_score,
            'windowed_global_calibrated_score': windowed_base_score,
            'alpha_global': effective_beta,
            'alpha_region': correction_map,
            'alpha_final': correction_map,
            'v13_effective_beta': effective_beta,
            'v13_correction_map': correction_map,
            'v14_gate_logit': permission_logit,
            'v14_intervention_gate': effective_gate,
            'v14_effective_gate': effective_gate,
            'v14_effective_beta': effective_beta,
            'v17_permission_logit': permission_logit,
            'v17_permission_probability': permission_probability,
            'v17_permission_hard': permission_hard,
            'v17_permission_gate': effective_permission,
            'v17_worthiness_logit': output['v14_gate_logit'],
            'v17_worthiness_gate': predicted_worthiness,
            'v17_effective_worthiness': effective_worthiness,
            'v17_effective_gate': effective_gate,
            'v17_effective_beta': effective_beta,
        })
        if self.training:
            output['v17_anchor_beta_logit'] = output[
                'v14_anchor_beta_logit'
            ]
            output['v17_anchor_sigma_logits'] = output[
                'v14_anchor_sigma_logits'
            ]
        return output

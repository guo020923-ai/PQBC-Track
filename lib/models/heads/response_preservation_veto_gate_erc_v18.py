"""ERC-v18 response preservation veto.

The v17 first-divergence audit found that most attributable harmful changes
were False-Gate interventions, while wrong branch direction and wrong region
were rare.  ERC-v18 therefore leaves beta, the regional gate, permission and
worthiness unchanged and adds one independent response-only question:

    is the official fused response already trustworthy and best preserved?

The final correction is

    S_ERC(x) = S0(x) * exp(g_perm * (1 - g_keep) * w * beta * G(x) * D(x)).

Both permission and preservation are strict binary decisions at inference and
use straight-through estimators during training.  The preservation head sees
only the global encoded response evidence.  It consumes no box geometry,
candidate IoU, search feature or future trajectory signal.
"""

import torch
from torch import nn

from .response_selective_abstention_worthiness_gate_erc_v17 import (
    ResponseSelectiveAbstentionWorthinessGateERCV17,
)


class ResponsePreservationVetoGateERCV18(
    ResponseSelectiveAbstentionWorthinessGateERCV17
):
    """Add an independent hard baseline-preservation veto to ERC-v17."""

    DESIGN = 'response_preservation_veto_gate_v18'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.design = self.DESIGN
        self.preservation_head = nn.Linear(self.beta_head.in_features, 1)
        nn.init.zeros_(self.preservation_head.weight)
        nn.init.zeros_(self.preservation_head.bias)

    @staticmethod
    def _straight_through_hard(probability, training):
        hard = probability.gt(0.5).to(dtype=probability.dtype)
        if training:
            gate = hard + probability - probability.detach()
        else:
            gate = hard
        return hard, gate

    def forward(self, cls_response, cont_response, flag=None):
        output = super().forward(cls_response, cont_response, flag=flag)

        preservation_logit = self.preservation_head(
            output['v13_global_features']
        )
        preservation_probability = preservation_logit.sigmoid()
        preservation_hard, preservation_gate = (
            self._straight_through_hard(
                preservation_probability, self.training
            )
        )

        permission_probability = output['v17_permission_probability']
        permission_hard, permission_gate = self._straight_through_hard(
            permission_probability, self.training
        )
        predicted_worthiness = output['v17_worthiness_gate']

        enabled_mask = output['active_mask'] & (not self.force_off) & (
            self.inference_mode == 'FULL_ERC'
        )
        effective_permission = torch.where(
            enabled_mask[:, None], permission_gate,
            torch.zeros_like(permission_gate),
        )
        effective_preservation = torch.where(
            enabled_mask[:, None], preservation_gate,
            torch.zeros_like(preservation_gate),
        )
        effective_worthiness = torch.where(
            enabled_mask[:, None], predicted_worthiness,
            torch.zeros_like(predicted_worthiness),
        )
        allow_after_veto = effective_permission * (
            1.0 - effective_preservation
        )
        effective_gate = allow_after_veto * effective_worthiness

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

        output.update({
            'features': output['v13_decision_features'],
            'alpha': correction_map.flatten(1),
            'raw_alpha': predicted_beta,
            'z_alpha': output['v13_beta_logit'],
            'reliability': effective_gate,
            'reliability_gate': effective_gate,
            'intervention_gate': effective_gate,
            'gate_logit': output['v17_permission_logit'],
            'gate_decision_features': output['v13_decision_features'],
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
            'v14_intervention_gate': effective_gate,
            'v14_effective_gate': effective_gate,
            'v14_effective_beta': effective_beta,
            # Preserve v17 semantic outputs for the shared supervision path.
            'v17_permission_hard': permission_hard,
            'v17_permission_gate': effective_permission,
            'v17_effective_worthiness': effective_worthiness,
            'v17_effective_gate': effective_gate,
            'v17_effective_beta': effective_beta,
            # v18-specific preservation evidence and final decision.
            'v18_preservation_logit': preservation_logit,
            'v18_preservation_probability': preservation_probability,
            'v18_preservation_hard': preservation_hard,
            'v18_preservation_gate': effective_preservation,
            'v18_allow_after_veto': allow_after_veto,
            'v18_effective_gate': effective_gate,
            'v18_effective_beta': effective_beta,
        })
        return output

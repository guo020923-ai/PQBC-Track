"""ERC-v15 continuous intervention-worthiness gate.

ERC-v15 deliberately keeps the complete ERC-v14 inference architecture:

    S_ERC(x) = S0(x) * exp(q * beta * G(x) * D(x)).

Only the meaning of ``q`` changes.  It is trained as a continuous estimate of
whether the reachable response-level correction is worth applying, rather
than as the binary reachability label used by ERC-v14.  The detached target
construction and losses live in the actor; this module remains free of boxes,
geometry, IoU, and trajectory information.
"""

import torch

from .response_intervention_gate_erc_v14 import (
    ResponseInterventionGateERCV14,
)


def continuous_worthiness_target(
    advantage_raw, keep_margin=0.05, worthiness_tau=0.10
):
    """Map detached response-margin advantage to the ERC-v15 soft target.

    ``advantage_raw`` is ``M_oracle - M0``.  Values up to ``keep_margin``
    belong to the strict baseline-preservation region.  This helper detaches
    the input by construction so no gradient can pass through the response
    Oracle into the original response branches.
    """
    keep_margin = float(keep_margin)
    worthiness_tau = float(worthiness_tau)
    if keep_margin < 0.0:
        raise ValueError('ERC-v15 keep_margin must be non-negative')
    if worthiness_tau <= 0.0:
        raise ValueError('ERC-v15 worthiness_tau must be positive')

    advantage_raw = torch.as_tensor(advantage_raw).detach()
    advantage_effective = (advantage_raw - keep_margin).clamp_min(0.0)
    gate_target = advantage_effective / (
        advantage_effective + worthiness_tau
    )
    return (
        advantage_raw.detach(),
        advantage_effective.detach(),
        gate_target.detach(),
    )


class ResponseWorthinessGateERCV15(ResponseInterventionGateERCV14):
    """ERC-v14 architecture with ERC-v15 continuous Gate semantics."""

    DESIGN = 'response_continuous_worthiness_gate_v15'

    def forward(self, cls_response, cont_response, flag=None):
        output = super().forward(cls_response, cont_response, flag=flag)

        # Keep v14 aliases for checkpoint/tool compatibility while exposing
        # unambiguous v15 names to the actor and future audit tools.
        output.update({
            'v15_gate_logit': output['v14_gate_logit'],
            'v15_worthiness_gate': output['v14_intervention_gate'],
            'v15_effective_gate': output['v14_effective_gate'],
            'v15_effective_beta': output['v14_effective_beta'],
        })
        if self.training:
            output['v15_anchor_beta_logit'] = output[
                'v14_anchor_beta_logit'
            ]
            output['v15_anchor_sigma_logits'] = output[
                'v14_anchor_sigma_logits'
            ]
        return output

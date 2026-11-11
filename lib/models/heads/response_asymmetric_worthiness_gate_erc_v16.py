"""ERC-v16 asymmetric risk-calibrated intervention gate.

ERC-v16 keeps the exact ERC-v15 inference equation and response-only inputs::

    S_ERC(x) = S0(x) * exp(q * beta * G(x) * D(x)).

Only Gate supervision changes.  False intervention is asymmetric in an online
tracker because one harmful decision can alter every later crop.  The actor
therefore trains ``q`` with a strict, differentiable preservation ceiling for
low-advantage frames, cost-sensitive continuous calibration, and logit-space
ranking.  No hard test threshold, box target, IoU target, or trajectory input
is introduced here.
"""

from .response_worthiness_gate_erc_v15 import (
    ResponseWorthinessGateERCV15,
)


class ResponseAsymmetricWorthinessGateERCV16(
    ResponseWorthinessGateERCV15
):
    """ERC-v15 architecture with asymmetric Gate training semantics."""

    DESIGN = 'response_asymmetric_worthiness_gate_v16'

    def forward(self, cls_response, cont_response, flag=None):
        output = super().forward(
            cls_response, cont_response, flag=flag
        )
        output.update({
            'v16_gate_logit': output['v15_gate_logit'],
            'v16_worthiness_gate': output['v15_worthiness_gate'],
            'v16_effective_gate': output['v15_effective_gate'],
            'v16_effective_beta': output['v15_effective_beta'],
        })
        if self.training:
            output['v16_anchor_beta_logit'] = output[
                'v15_anchor_beta_logit'
            ]
            output['v16_anchor_sigma_logits'] = output[
                'v15_anchor_sigma_logits'
            ]
        return output

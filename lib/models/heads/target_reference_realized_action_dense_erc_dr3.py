"""Realized-action aligned target-reference dense ERC (DR3).

DR3 intentionally keeps DR2's deploy-time mechanism unchanged.  The model
still proposes a bounded dense response residual and deploys it through a
causal safety gate.  Its independent design name selects a different training
objective: the gate is supervised by the utility of actions that the current
residual can *actually* execute, instead of a hypothetical reachable target.

No box geometry, IoU, dataset identity, or RQC feature is consumed here.
"""

import numpy as np
import torch

from .target_reference_trajectory_safe_dense_erc_dr2 import (
    TargetReferenceTrajectorySafeDenseERCDR2,
)


class TargetReferenceRealizedActionDenseERCDR3(
        TargetReferenceTrajectorySafeDenseERCDR2):
    """DR2 deployment with realized-action aligned training supervision."""

    DESIGN = 'target_reference_realized_action_dense_erc_dr3'

    def __init__(self, *args, **kwargs):
        self.exact_deploy_score = kwargs.pop('exact_deploy_score', False)
        super().__init__(*args, **kwargs)
        self.design = self.DESIGN

    def _stable_response_channels(self, cls_response, cont_response):
        stable = super()._stable_response_channels(cls_response, cont_response)
        if self.exact_deploy_score:
            # Keep clipped log channels as NETWORK INPUTS only. Ranking uses
            # raw probabilities, NumPy's float64 Hann and tracker order C*H*T.
            hann = np.hanning(self.feat_sz)
            window = torch.as_tensor(np.outer(hann, hann),
                                     device=cls_response.device).unsqueeze(0)
            stable['base'] = cls_response * cont_response
            stable['windowed_base'] = cls_response.double() * window * cont_response.double()
        return stable

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        # Explicit aliases keep DR2 checkpoints/results isolated and make it
        # impossible for the DR3 loss to silently consume a wrong code path.
        for suffix in (
                'raw_correction', 'gated_correction', 'correction',
                'current_gate', 'temporal_gate', 'effective_gate',
                'proposal_distance', 'baseline_index', 'proposal_index'):
            output['dr3_' + suffix] = output['dr2_' + suffix]
        return output

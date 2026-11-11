"""TRC-v5b: deployment-calibrated selective evidence correction.

v5 learned a useful probability ordering under the natural correction prior,
but its fixed 0.5 action boundary made every deployed action KEEP.  v5b keeps
the calibrated probability learner and uses the cross-source audited action
boundary.  A below-boundary initial prior preserves the exact baseline before
training.
"""

import math

from torch import nn

from lib.models.heads.target_reference_erc_v5 import TargetReferenceERCV5


class TargetReferenceERCV5B(TargetReferenceERCV5):
    DESIGN = 'target_reference_erc_trc_v5b_action_calibrated'

    def __init__(self, feat_sz, hidden=32, alpha_max=0.5, temperature=1.0,
                 d_max=4.0, inference_mode='FULL_ERC', force_off=False,
                 permission_threshold=0.15,
                 permission_initial_probability=0.05):
        if not 0 < permission_initial_probability < permission_threshold < 1:
            raise ValueError(
                'Require 0 < initial permission probability < threshold < 1'
            )
        super().__init__(
            feat_sz, hidden, alpha_max, temperature, d_max,
            inference_mode, force_off,
        )
        self.design = self.DESIGN
        self.permission_threshold = float(permission_threshold)
        initial_logit = self.temperature * math.log(
            permission_initial_probability / (1.0 - permission_initial_probability)
        )
        nn.init.zeros_(self.permission_head[-1].weight)
        nn.init.constant_(self.permission_head[-1].bias, initial_logit)

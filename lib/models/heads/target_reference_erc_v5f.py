"""TRC-v5f: validation-calibrated hard C/T deployment boundary.

The learned direction probability, loss, Gate, strength and correction formula
are unchanged from v5d.  Only the hard deployment boundary for the rare T
action is made explicit.  The rounded 0.35 value is selected from a fixed VL
validation separability audit, not from TNL2K test results.
"""

from lib.models.heads.target_reference_erc_v5d import TargetReferenceERCV5D


class TargetReferenceERCV5F(TargetReferenceERCV5D):
    DESIGN = 'target_reference_erc_trc_v5f_direction_calibrated'

    def __init__(self, feat_sz, hidden=32, alpha_max=0.5, temperature=1.0,
                 d_max=4.0, inference_mode='FULL_ERC', force_off=False,
                 permission_threshold=0.15,
                 permission_initial_probability=0.05,
                 direction_t_threshold=0.35):
        if not 0 < direction_t_threshold < 1:
            raise ValueError('direction_t_threshold must be in (0, 1)')
        super().__init__(
            feat_sz, hidden, alpha_max, temperature, d_max,
            inference_mode, force_off, permission_threshold,
            permission_initial_probability,
        )
        self.direction_t_threshold = float(direction_t_threshold)
        self.design = self.DESIGN

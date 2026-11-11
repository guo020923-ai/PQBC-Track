"""TRC-v5d: balanced deployment gate with decoupled action supervision."""

from lib.models.heads.target_reference_erc_v5c import TargetReferenceERCV5C


class TargetReferenceERCV5D(TargetReferenceERCV5C):
    """Keep the v5c inference policy while separating conflicting losses."""

    DESIGN = 'target_reference_erc_trc_v5d_decoupled_balanced'

    def __init__(self, feat_sz, hidden=32, alpha_max=0.5, temperature=1.0,
                 d_max=4.0, inference_mode='FULL_ERC', force_off=False,
                 permission_threshold=0.15,
                 permission_initial_probability=0.05):
        super().__init__(
            feat_sz, hidden, alpha_max, temperature, d_max,
            inference_mode, force_off, permission_threshold,
            permission_initial_probability,
        )
        self.design = self.DESIGN

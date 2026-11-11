"""TRC-v6b: positive-anchored training for the v6 joint action policy.

Deployment is intentionally identical to TRC-v6.  The independent design ID
selects a loss that corrects the all-KEEP collapse observed in the v6 pilot.
"""

from lib.models.heads.target_reference_erc_v6 import TargetReferenceERCV6


class TargetReferenceERCV6B(TargetReferenceERCV6):
    DESIGN = 'target_reference_erc_trc_v6b_balanced_joint_action'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.design = self.DESIGN

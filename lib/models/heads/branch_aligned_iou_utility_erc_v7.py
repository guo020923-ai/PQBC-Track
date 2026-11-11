"""Deployment-aligned IoU-utility evidence calibration (FER-v7).

FER-v7 deliberately keeps the deployable FER-v6 policy unchanged.  It still
observes response/reference evidence only and emits one of the fixed
KEEP/GLOBAL_C/GLOBAL_T/REGIONAL_T calibration actions.  The new version name
selects an IoU-aligned training objective; candidate boxes are never inputs to
this module and are not modified by it.
"""

from .branch_aligned_factorized_reachability_erc_v6 import (
    BranchAlignedFactorizedReachabilityERCV6,
)


class BranchAlignedIoUUtilityERCV7(BranchAlignedFactorizedReachabilityERCV6):
    """FER-v6 inference policy trained with detached candidate-IoU utility."""

    DESIGN = 'branch_aligned_iou_utility_erc_v7'

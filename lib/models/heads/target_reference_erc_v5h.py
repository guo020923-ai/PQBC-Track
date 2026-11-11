"""TRC-v5h: positive-anchored branch-conditional permission.

The deployed response calibration is identical to v5g.  This independent
design identifier selects a class-balanced, positive-anchored supervision
objective so rare C/T actions are not collapsed by the KEEP majority.
"""

from lib.models.heads.target_reference_erc_v5g import TargetReferenceERCV5G


class TargetReferenceERCV5H(TargetReferenceERCV5G):
    DESIGN = 'target_reference_erc_trc_v5h_positive_anchored_permission'

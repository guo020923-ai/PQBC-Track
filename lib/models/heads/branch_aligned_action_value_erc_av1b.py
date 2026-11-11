"""Balanced unified action-value evidence calibration (ERC-AV1b).

ERC-AV1b deliberately keeps the exact AV1 deployment mechanism and its
single utility head.  The corresponding training loss separates two
questions on that same utility vector: whether any non-KEEP action is worth
taking, and which complete action is best after intervention is warranted.
This avoids the sparse-action KEEP collapse without reintroducing separate
permission/direction/scope heads.
"""

from .branch_aligned_action_value_erc_av1 import (
    BranchAlignedActionValueERCAV1,
)


class BranchAlignedActionValueERCAV1B(BranchAlignedActionValueERCAV1):
    """AV1 deployment path with balanced hierarchical utility supervision."""

    DESIGN = 'branch_aligned_action_value_erc_av1b_balanced'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.design = self.DESIGN

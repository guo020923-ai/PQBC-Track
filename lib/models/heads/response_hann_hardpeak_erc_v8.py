"""Hann-aware hard-peak Evidence Reliability Calibration (ERC-v8).

The learnable architecture intentionally stays identical to ERC-v7: a global
evidence-preference branch, a regional residual branch, and an intervention
gate.  ERC-v8 is an independent design identifier because its supervision is
different: all decision losses are aligned to the test-time ``Hann * C * T``
hard peak.  Hann remains a fixed decision prior and is never an input feature.

This module has no candidate boxes, geometry, candidate features, or IoU.  The
future RQC module owns candidate-quality reasoning.
"""

from .response_global_regional_erc_v7 import ResponseGlobalRegionalERCV7


class ResponseHannHardPeakERCV8(ResponseGlobalRegionalERCV7):
    """ERC-v7 architecture with an independent v8 hard-decision identity."""

    DESIGN = 'response_hann_hardpeak_v8'


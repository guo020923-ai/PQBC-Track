"""Regional-effective calibration ERC-v8d.

The module architecture and inference formula are intentionally identical to
ERC-v8c.  v8d is a distinct design identifier so its response-only regional
effective-difference supervision can be enabled without changing or
overwriting v8c experiments/checkpoints.
"""

from .response_hann_gate_erc_v8c import ResponseHannGateERCV8C


class ResponseRegionalEffectiveERCV8D(ResponseHannGateERCV8C):
    """ERC-v8c architecture trained with isolated regional supervision."""

    DESIGN = 'response_regional_effective_v8d'


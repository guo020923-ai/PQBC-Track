# Execution-aligned quarter-action protocol

The only scientific change from feedback round 71 is the action basis used by
the ERC. Training, TRAIN-only calibration, ranking, gating, output, and causal
crop feedback all use the same directly reachable quarter-mapped actions.
There is no post-selection identity change. The RQC30 source, feedback-consistent
30,000-row TRAIN cache, architecture size, optimizer, 100 epochs/187,500 steps,
fixed Epoch100 selection, calibration split, and FULL700 protocol remain fixed.

If strict TRAIN-only calibration is infeasible, the predeclared maximum-utility
coverage candidate is reported as a development-stage fallback so that exactly
one evaluation is still produced. TEST GT is not read by training or calibration.

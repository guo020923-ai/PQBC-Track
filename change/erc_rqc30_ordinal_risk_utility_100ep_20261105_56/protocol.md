# Unified ordinal risk utility

The previous follow-up failed because one head selected an action by predicted
mean utility while a separate head estimated benefit.  On held-out TRAIN
blocks, high benefit probability did not imply that the separately selected
action was beneficial.

This revision removes that disagreement.  A single ordinal head predicts
three mutually exclusive outcomes for every action: harmful (`utility <= 0`),
marginal (`0 < utility < 1/21`), and actionable (`utility >= 1/21`).  The same
risk score `P(actionable)-P(harmful)` both ranks actions and decides whether to
leave KEEP.  Zero-initialized logits produce a zero score and exact KEEP before
training.  The RQC30 source, 25-action lattice, 48-D evidence, output-only
boundary, 100-epoch budget, and clip-block-disjoint TRAIN calibration remain
unchanged.  TEST data is excluded from training and calibration.

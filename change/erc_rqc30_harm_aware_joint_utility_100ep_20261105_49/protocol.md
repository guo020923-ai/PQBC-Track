# Proposed revision: harm-aware joint utility ERC, fixed Epoch 100

## Evidence motivating the change

The completed balanced-selector development evaluation remained negative:
62.69719696044922 AUC, or -0.11072921752929688 percentage points relative
to its same-run native RQC control.  Its paired 95% BCa interval was entirely
negative.  Reweighting reduced action 24's non-KEEP share from about 83.8% to
69.46%, but 453 of 700 sequences still degraded.  This shows that class
frequency was a symptom rather than the governing error.

The factorized model has a structural inconsistency.  A scalar gate predicts
whether *some* action is useful, while an independently trained selector picks
which action to execute.  A correct positive gate therefore does not ensure
that the selected action has positive utility.  Hard selector labels also
discard the measured utility of the other 23 alternatives.

## Scientific revision

The gate and selector are replaced by one shared action-value function.  It
predicts all 25 utilities relative to the frozen RQC box; KEEP is exactly zero
and competes in the same decision.  Training uses the existing sealed TRAIN
cache only.  The target is the already stored success-AUC21 gain relative to
KEEP, with continuous IoU only as a tie breaker.

The objective contains three metric-aligned terms:

1. group-balanced Smooth-L1 regression across harmful, sub-threshold, and
   actionable TRAIN actions;
2. a soft full-action utility-distribution loss including KEEP;
3. an explicit penalty when a truly non-positive TRAIN action is assigned
   positive predicted gain.

Deployment selects a non-KEEP action only if its predicted gain reaches one
complete success-AUC21 step (1/21).  Otherwise it returns the RQC box exactly.
The runtime remains output-only and never feeds the ERC box into RQC state.

## Training and evaluation boundary

The source is fixed RQC Epoch 30, SHA256
`1524962167d8f5b60a33b29960feae47f82b8e2a17869fc628e2f543978267bd`.
RQC, the native head, and the backbone remain frozen.  The same sealed 30,000
TRAIN examples and their original VL-source proportions are reused.

Training is predetermined at 100 epochs (187,500 optimizer steps), with only
Epoch 100 reportable.  No validation result selects a checkpoint.  The
100-epoch schedule is fixed before training, uses five warmup epochs followed
by cosine decay, and reports only the final checkpoint.  The harm-aware
objective and joint KEEP/action decision are intended to prevent the late
action-frequency collapse observed with the earlier factorized selector.

The user explicitly authorized this fixed training and one post-training
TNL2K FULL700 evaluation on 2026-11-05.  The test may start only after a
read-only terminal audit verifies all 100 checkpoints, 187,500 steps, the
sealed cache identity, and complete process exit.  No checkpoint selection,
retry, additional training candidate, or automatic generalization is allowed.


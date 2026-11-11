# Round 123: independent Round73 training-seed replication

This run tests whether the Round73 result reproduces under an independently fixed
training seed. The only deliberately changed variable is the training seed, from
`20261105` to `20261109`. Architecture, loss, fixed TRAIN cache, action basis,
optimizer schedule, cooldown=8, calibration split and algorithm, 100 epochs,
187,500 optimizer steps, and fixed Epoch100 selection remain unchanged.

The TRAIN-only threshold is derived by the unchanged calibration procedure and is
frozen before TNL2K execution. Exactly one TNL2K NLBBOX FULL700 evaluation is
allowed. No retry, intermediate checkpoint selection, sequence-specific rule,
test-ground-truth calibration, or parameter change after observing results is
permitted. The result is retained whether it improves or regresses.

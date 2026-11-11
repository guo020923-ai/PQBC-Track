# Round 73: TRAIN-only selected temporal feedback cooldown

Round 72 showed a reproducible closed-loop accumulation mechanism: 191 sequences
with no commits exactly reproduced fixed RQC30; sequences with one to four commits
improved by 0.3615 pp on average, while every aggregate bin from five commits upward
was negative. Commit count and feedback-conditioned trajectory drift had Spearman
correlation 0.7815.

The only changed scientific variable is an eight-frame refractory period after a
feedback commit. The value was selected before this run by a declared TRAIN-only
ablation on 6,016 held-out rows grouped into 94 contiguous 64-frame clips. Of the
predeclared values 0, 4, 8, 16, 32, and 64, eight maximized held-out mean utility,
reduced selected-action harm from 37.70% to 35.48%, and reduced the maximum commits
per clip from 20 to 3. Completed TNL2K ground truth was used only to diagnose the
failure mechanism, never to select the cooldown value or the score threshold.

Architecture, loss, cache, action basis, optimizer schedule, 100 epochs, 187,500
steps, fixed Epoch100 selection, and TRAIN-only threshold calibration remain fixed.
After verified training, exactly one TNL2K FULL700 development-stage test is allowed.
No retry, intermediate-checkpoint selection, sequence identity rule, or TEST-derived
calibration is permitted.

# ERC30 negative-result diagnosis and local revision

## Result boundary

The completed TNL2K NLBBOX result is 62.6405143737793 AUC, compared with
63.02118818112782 for the reproduced RQC30 reference. The paired sequence
difference is -0.38067495772879756 percentage points, with a 95% BCa interval
of [-0.6876838144967318, -0.08034264779464437]. Of 700 sequences, 465 are
negative, 231 positive, and 4 unchanged. The negative result is therefore not
explained by one isolated showcase sequence.

## Common failure pattern

The v1 label gate treated every strictly positive utility difference as an
actionable resize. Because the utility contains a `1e-3 * IoU` tie breaker,
arbitrarily small continuous-IoU changes could turn a non-KEEP action into the
training label even when no official success threshold was crossed.

This is visible in the frozen TRAIN cache:

- v1 actionable-label fraction (`best_gain > 0`): 90.5667%;
- corresponding KEEP-label fraction: 9.4333%;
- fraction crossing one complete success-AUC21 step (`best_gain >= 1/21`):
  33.5067%;
- KEEP-label fraction under the metric-aligned gate: 66.4933%.

The deployed v1 policy selected non-KEEP on 81.7050% of official search
frames. Thus the high intervention rate is primarily a consequence of the
supervision rule rather than a rare inference anomaly.

Frame diagnostics show the same mechanism. Relative to the stored RQC30
reference, edits help when the reference IoU is already very low, but become
negative once the native result is usable:

- reference IoU 0-0.25: +2.556 pp mean success-AUC contribution per frame;
- 0.25-0.50: +0.843 pp;
- 0.50-0.75: -1.139 pp;
- 0.75-1.00: -1.872 pp.

The worst realized edit families are joint shrink (-0.967 pp/frame) and mixed
width-height directions (-0.734 pp/frame). The model lacks an effective
abstention target for already-correct RQC boxes.

## Attribution limitation discovered by the audit

The saved current outputs differ from the historical RQC30 prediction files
on 92.3984% of search frames, while the policy receipts contain only 81.7050%
non-KEEP actions. The 95th percentile center difference against the historical
reference is 111.73 pixels, which cannot be attributed to the center-preserving
size action alone. Therefore the official comparison is a valid full-system
comparison, but it is not a clean ERC-only causal estimate. A future evaluation
must serialize the same-run native RQC box alongside the ERC box during the
single source forward; no extra GPU baseline replay is required.

## Implemented scientific change

The single changed scientific variable is `minimum_actionable_gain`:

- old: 0;
- new: `1/21 = 0.047619047619047616`.

A non-KEEP label is now authorized only when it improves at least one complete
official success threshold. The continuous-IoU term remains a tie breaker among
already-actionable candidates. This value follows directly from the 21-point
success metric and was not selected from TNL2K sequences, subsets, or test
ground truth.

CPU validation passed with CUDA uninitialized, one model forward, zero
optimizer steps, and exact KEEP/action behavior on synthetic utilities below
and above the new gate. No training or test was launched.

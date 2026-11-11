# Round124 frozen generalization protocol

- Reuse exactly the audited Round124 Epoch100 checkpoint and numeric weights.
- Freeze threshold `0.36725926399230957`, training seed `20261109`, cooldown `8`, and `post_selection_keep_fallback` runtime.
- Run LaSOT first, then OTB99, once each, with eight persistent workers.
- Do not retrain, tune per dataset, retry automatically, reuse result roots, or overwrite any earlier stage.
- Evaluate both the official branch and the same-forward native shadow with the same official evaluator.
- Persist paired-sequence BCa statistics, action counts, feedback commits, suppressed proposals, and exact invalid-feedback fallback counts.


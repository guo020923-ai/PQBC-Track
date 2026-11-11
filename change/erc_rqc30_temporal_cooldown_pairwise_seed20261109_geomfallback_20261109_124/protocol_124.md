# Round124 geometry-fallback recovery

Round123 completed its independent 100-epoch training but its sole FULL700 run
terminated when a selected non-KEEP candidate could not legally become the next
tracker state. Round124 reuses the audited Round123 Epoch100 numeric artifact and
changes only the terminal behavior for that impossible commit: the frame follows
the native KEEP transition. Valid candidates, model scores, threshold, cooldown,
weights, training seed, dataset inventory, and evaluator remain unchanged.

The recovery writes to a new package, stage, result name, and run ID. It never
overwrites Round73 or Round123 artifacts and performs no retraining.

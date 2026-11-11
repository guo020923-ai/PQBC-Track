import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKAGE = "erc_rqc30_unary_objective_ablation_20261109_132"
FILES = [
    "__init__.py", "actions.py", "config.py", "model.py", "objective.py",
    "checkpoint.py", "training.py", "train_child.py", "calibration.py",
    "io_utils.py", "runtime.py", "fast_runner.py", "stage.py",
    "run_execution_pipeline.py", "check_cpu.py", "spawn_check_132.py",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


rows = []
for name in FILES:
    path = ROOT / name
    rows.append({
        "package": PACKAGE,
        "name": name,
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    })

manifest = {
    "status": "IMPLEMENTATION_SEALED_AND_AUTHORIZED",
    "authorization": "user_requested_pairwise_vs_unary_ablation_20261109",
    "scientific_variable": "unary_keep_relative_objective_instead_of_pairwise_objective",
    "controlled_reference": "Round124_pairwise_final",
    "unchanged": [
        "architecture", "48d_evidence", "25_actions", "TRAIN_cache",
        "training_seed", "optimizer", "100_epochs_187500_steps", "Epoch100_only",
        "TRAIN_only_calibration", "threshold_algorithm", "cooldown8",
        "post_selection_keep_fallback", "TNL2K_FULL700_evaluator",
    ],
    "files": rows,
}
(ROOT / "implementation_manifest_132.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps({"files": len(rows), "manifest_sha256": digest(ROOT / "implementation_manifest_132.json")}, sort_keys=True))

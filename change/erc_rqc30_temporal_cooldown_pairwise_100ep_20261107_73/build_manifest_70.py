import json
import os
from pathlib import Path

from .io_utils import digest


FILES = [
    "__init__.py", "authorization_70.json", "posttrain_test_authorization_70.json",
    "protocol_70.md", "config.py", "io_utils.py", "model.py", "objective.py",
    "runtime.py", "cache.py", "cache_child.py", "training.py", "train_child.py",
    "checkpoint.py", "calibration.py", "fast_runner.py", "stage.py", "check_cpu.py",
    "run_feedback_pipeline.py",
]


def main():
    root = Path(__file__).resolve().parent
    rows = []
    for name in FILES:
        path = root / name
        rows.append({
            "package": root.name, "name": name, "bytes": path.stat().st_size,
            "sha256": digest(path),
        })
    result = {
        "status": "IMPLEMENTATION_SEALED_AND_AUTHORIZED",
        "package": root.name, "scientific_variable": "quarter_shrink_causal_crop_feedback",
        "technical_integrity_change": "cudnn_deterministic_cublas_workspace_warn_only_unsupported_cuda_cumsum",
        "cache_examples": 30000, "training_epochs": 100, "optimizer_steps": 187500,
        "fixed_checkpoint": "Epoch100", "automatic_full700_test": True,
        "automatic_retry": False, "test_gt_parameter_selection": False,
        "files": rows,
    }
    path = root / "implementation_manifest_70.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())
    print(json.dumps({"status": result["status"], "files": len(rows), "sha256": digest(path)}, sort_keys=True))


if __name__ == "__main__":
    main()

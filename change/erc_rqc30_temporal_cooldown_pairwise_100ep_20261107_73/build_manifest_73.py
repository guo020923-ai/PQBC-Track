import hashlib
import json
from pathlib import Path


FILES = [
    "__init__.py", "actions.py", "config.py", "model.py", "objective.py",
    "training.py", "train_child.py", "checkpoint.py", "calibration.py",
    "runtime.py", "fast_runner.py", "stage.py", "io_utils.py", "check_cpu.py",
    "run_execution_pipeline.py", "authorization_73.json", "evidence_73.json",
    "posttrain_test_authorization_73.json", "protocol_73.md", "build_manifest_73.py",
]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


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
    "package": root.name,
    "scientific_variable": "train_only_selected_temporal_feedback_cooldown",
    "temporal_cooldown_frames": 8,
    "files": rows,
}
(root / "implementation_manifest_73.json").write_text(
    json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
)
print(json.dumps({"files": len(rows), "manifest": str(root / "implementation_manifest_73.json")}))

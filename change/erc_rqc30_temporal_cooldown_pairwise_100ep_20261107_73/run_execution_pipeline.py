"""Reuse the sealed TRAIN-only feedback cache, train fixed100, calibrate, and test once."""

import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from change.erc_rqc30_ordinal_risk_utility_100ep_20261105_56.run_authorized_pipeline import (
    identity, require_idle_gpu, write_new,
)
from .io_utils import digest


PACKAGE = "change.erc_rqc30_temporal_cooldown_pairwise_100ep_20261107_73"


def verify_manifest(project, package):
    path = package / "implementation_manifest_73.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "IMPLEMENTATION_SEALED_AND_AUTHORIZED":
        raise PermissionError("manifest")
    for row in manifest["files"]:
        target = project / "change" / row["package"] / row["name"]
        if target.stat().st_size != row["bytes"] or digest(target) != row["sha256"]:
            raise ValueError("implementation identity " + str(target))
    return manifest


def audit_cache(cache_dir):
    from .config import PROTOCOL
    from .training import load_cache
    manifest, arrays = load_cache(cache_dir)
    if digest(Path(cache_dir) / "manifest.json") != PROTOCOL["source_cache_manifest_sha256"]:
        raise ValueError("cache manifest")
    return {
        "status": "FIXED_FEEDBACK_CACHE_REUSE_VERIFIED",
        "examples": 30000,
        "shard_count": len(manifest["shards"]),
        "manifest_sha256": PROTOCOL["source_cache_manifest_sha256"],
        "evidence_shape": list(arrays["evidence"].shape),
        "feedback_consistent": True,
        "test_data_read": False,
    }


def audit_training(train_dir, cache_dir):
    import torch
    from .config import PROTOCOL
    from .model import VERSION

    train_dir, cache_dir = Path(train_dir), Path(cache_dir)
    completed = json.loads((train_dir / "completed.json").read_text(encoding="utf-8"))
    checkpoints = sorted((train_dir / "checkpoints").glob("ERC_epoch_*.pth.tar"))
    stats = [json.loads(line) for line in (train_dir / "stats.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = [f"ERC_epoch_{epoch:04d}.pth.tar" for epoch in range(1, 101)]
    if (completed.get("optimizer_steps") != 187500 or
            [path.name for path in checkpoints] != expected or
            [row["epoch"] for row in stats] != list(range(1, 101))):
        raise ValueError("training continuity")
    if any(not math.isfinite(float(value)) for row in stats for value in row["stats"].values()):
        raise FloatingPointError("training stats")
    final = checkpoints[-1]; saved = torch.load(str(final), map_location="cpu")
    cache_sha = digest(cache_dir / "manifest.json")
    if (saved.get("kind") != VERSION or saved.get("protocol") != PROTOCOL or
            saved.get("epoch") != 100 or saved.get("optimizer_steps") != 187500 or
            saved.get("cache_manifest_sha256") != cache_sha or
            completed.get("checkpoint_sha256") != digest(final)):
        raise ValueError("training identity")
    return {
        "status": "COMPLETE_VERIFIED", "checkpoint_count": 100, "stats_count": 100,
        "optimizer_steps": 187500, "fixed_epoch": 100,
        "final_checkpoint": str(final), "final_checkpoint_sha256": digest(final),
        "final_checkpoint_bytes": final.stat().st_size, "cache_manifest_sha256": cache_sha,
        "source_rqc_epoch": 30,
        "source_rqc_checkpoint_sha256": PROTOCOL["source_rqc_checkpoint_sha256"],
        "checkpoint_selection": "fixed_epoch100_no_best",
        "training_seconds": completed["training_seconds"],
        "train_rows": completed["train_rows"], "calibration_rows": completed["calibration_rows"],
        "last_train_stats": stats[-1]["stats"], "audited_unix": time.time(),
    }


def main(argv):
    if len(argv) != 4:
        raise SystemExit("usage: run_execution_pipeline PROJECT CACHE_DIR STAGE")
    project, cache_dir, stage = map(Path, argv[1:])
    package = project / "change" / "erc_rqc30_temporal_cooldown_pairwise_100ep_20261107_73"
    authorization = json.loads((package / "authorization_73.json").read_text(encoding="utf-8"))
    test_authorization = json.loads((package / "posttrain_test_authorization_73.json").read_text(encoding="utf-8"))
    if authorization.get("status") != "AUTHORIZED_EVIDENCE_BASED_TEMPORAL_COOLDOWN_ITERATION":
        raise PermissionError("training authorization")
    if test_authorization.get("status") != "AUTHORIZED_ONE_TNL2K_FULL700_AFTER_VERIFIED_COOLDOWN_TRAINING":
        raise PermissionError("test authorization")
    implementation = verify_manifest(project, package)
    if stage.exists():
        raise FileExistsError(stage)
    require_idle_gpu(); stage.mkdir(); os.chdir(project); sys.path.insert(0, str(project))
    phase = "LAUNCH"
    write_new(stage / "actual_launch.json", {
        "status": "ACTUAL_TEMPORAL_COOLDOWN_PAIRWISE_100EP_PIPELINE_LAUNCH",
        "process": identity(os.getpid()), "started_unix": time.time(),
        "scientific_variable": "train_only_selected_temporal_feedback_cooldown",
        "temporal_cooldown_frames": 8,
        "cache_reused_without_test_data": True,
        "training_epochs": 100, "optimizer_steps": 187500,
        "fixed_checkpoint": "Epoch100", "automatic_full700_test": True,
        "automatic_retry": False,
        "implementation_manifest_sha256": digest(package / "implementation_manifest_73.json"),
        "sealed_file_count": len(implementation["files"]),
    })
    try:
        phase = "CACHE_AUDIT"
        cache_audit = audit_cache(cache_dir); write_new(stage / "cache_audit.json", cache_audit)
        phase = "FIXED100_TRAINING"
        require_idle_gpu(); train_dir = stage / "train"; train_dir.mkdir()
        train_command = [sys.executable, "-B", "-m", PACKAGE + ".train_child",
                         str(train_dir), str(cache_dir)]
        child = subprocess.Popen(train_command, cwd=str(project))
        write_new(stage / "training_launch.json", {
            "status": "TRAINING_CHILD_STARTED", "process": identity(child.pid),
            "command": train_command, "started_unix": time.time(),
        })
        code = child.wait()
        if code:
            raise RuntimeError("training child " + str(code))
        phase = "TRAINING_AUDIT"
        training_audit = audit_training(train_dir, cache_dir)
        write_new(stage / "training_audit.json", training_audit)
        phase = "NUMERIC_EXPORT"
        from .checkpoint import export_numeric
        numeric_dir = stage / "numeric"
        numeric_manifest = export_numeric(training_audit["final_checkpoint"], numeric_dir)
        phase = "TRAIN_ONLY_CALIBRATION"
        from .calibration import calibrate
        calibration = calibrate(numeric_dir, cache_dir)
        write_new(stage / "calibration_complete.json", calibration)
        phase = "TNL2K_FULL700_TEST"
        require_idle_gpu()
        write_new(stage / "test_launch.json", {
            "status": "AUTHORIZED_TEMPORAL_COOLDOWN_TNL2K_FULL700_STARTED",
            "fixed_checkpoint_sha256": training_audit["final_checkpoint_sha256"],
            "numeric_npz_sha256": numeric_manifest["numeric_npz_sha256"],
            "calibration_status": calibration["status"],
            "threshold": calibration["threshold"],
            "reporting_label": calibration["reporting_label"],
            "sequences": 700, "frames": 516038, "searches": 515338,
            "persistent_workers": 8, "feedback_conditioned_native_shadow": True,
            "temporal_cooldown_frames": 8,
            "automatic_retry": False, "started_unix": time.time(),
        })
        from .stage import run_stage
        result = run_stage(training_audit, numeric_dir, stage / "test", processes=8)
        result["reporting_label"] = calibration["reporting_label"]
        result["admission_constraints_passed"] = calibration["admission_constraints_passed"]
        write_new(stage / "official_complete.json", result)
        write_new(stage / "terminal_status.json", {
            "status": "PIPELINE_COMPLETE_PENDING_CONTROLLER_OS0_EXIT",
            "process": identity(os.getpid()), "ended_unix": time.time(),
            "expected_os_exit": 0, "automatic_retry": False,
        })
        print(json.dumps(result, sort_keys=True), flush=True)
    except BaseException as error:
        write_new(stage / "terminal_failure.json", {
            "status": "TEMPORAL_COOLDOWN_PAIRWISE_PIPELINE_FAILED", "phase": phase,
            "error": repr(error), "traceback": traceback.format_exc(),
            "ended_unix": time.time(), "automatic_retry": False,
        })
        raise


if __name__ == "__main__":
    main(sys.argv)

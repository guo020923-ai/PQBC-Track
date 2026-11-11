"""Build feedback-consistent TRAIN cache, fixed100 train, calibrate, and test once."""

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


PACKAGE = "change.erc_rqc30_feedback_pairwise_100ep_20261107_70"


def verify_manifest(project, package):
    manifest = json.loads((package / "implementation_manifest_70.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "IMPLEMENTATION_SEALED_AND_AUTHORIZED":
        raise PermissionError("manifest")
    for row in manifest["files"]:
        path = project / "change" / row["package"] / row["name"]
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("implementation identity " + str(path))
    return manifest


def audit_cache(cache_dir):
    from .config import PROTOCOL
    from .training import load_cache
    manifest, arrays = load_cache(cache_dir)
    if (manifest.get("behavior_checkpoint_sha256") != PROTOCOL["behavior_checkpoint_sha256"] or
            manifest.get("behavior_numeric_sha256") != PROTOCOL["behavior_numeric_sha256"] or
            manifest.get("behavior_threshold") != PROTOCOL["behavior_threshold"] or
            sum(manifest.get("behavior_action_totals", [])) != 30000):
        raise ValueError("cache behavior audit")
    return {
        "status": "FEEDBACK_CACHE_COMPLETE_VERIFIED", "examples": 30000,
        "shard_count": len(manifest["shards"]),
        "manifest_sha256": digest(Path(cache_dir) / "manifest.json"),
        "evidence_shape": list(arrays["evidence"].shape),
        "feedback_commits": manifest["feedback_commits"],
        "behavior_action_totals": manifest["behavior_action_totals"],
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
        raise SystemExit("usage: run_feedback_pipeline PROJECT BEHAVIOR_NUMERIC STAGE")
    project, behavior_numeric, stage = map(Path, argv[1:])
    package = project / "change" / "erc_rqc30_feedback_pairwise_100ep_20261107_70"
    authorization = json.loads((package / "authorization_70.json").read_text(encoding="utf-8"))
    test_authorization = json.loads((package / "posttrain_test_authorization_70.json").read_text(encoding="utf-8"))
    if authorization.get("status") != "AUTHORIZED_EVIDENCE_BASED_FEEDBACK_ITERATION":
        raise PermissionError("training authorization")
    if test_authorization.get("status") != "AUTHORIZED_ONE_TNL2K_FULL700_AFTER_VERIFIED_FEEDBACK_TRAINING":
        raise PermissionError("test authorization")
    implementation = verify_manifest(project, package)
    if stage.exists():
        raise FileExistsError(stage)
    require_idle_gpu(); stage.mkdir(); os.chdir(project); sys.path.insert(0, str(project))
    phase = "LAUNCH"
    write_new(stage / "actual_launch.json", {
        "status": "ACTUAL_FEEDBACK_PAIRWISE_100EP_PIPELINE_LAUNCH",
        "process": identity(os.getpid()), "started_unix": time.time(),
        "scientific_variable": "quarter_shrink_causal_crop_feedback",
        "cache_examples": 30000, "training_epochs": 100, "optimizer_steps": 187500,
        "fixed_checkpoint": "Epoch100", "automatic_full700_test": True,
        "automatic_retry": False, "deterministic_source_execution": False,
        "determinism_mode": "cudnn_deterministic_cublas_workspace_warn_only_unsupported_cuda_cumsum",
        "implementation_manifest_sha256": digest(package / "implementation_manifest_70.json"),
        "sealed_file_count": len(implementation["files"]),
    })
    try:
        phase = "FEEDBACK_CACHE_EXTRACTION"
        cache_root = stage / "cache_build"
        cache_command = [sys.executable, "-B", "-m", PACKAGE + ".cache_child",
                         str(cache_root), str(behavior_numeric)]
        cache_child = subprocess.Popen(cache_command, cwd=str(project))
        write_new(stage / "cache_launch.json", {
            "status": "FEEDBACK_CACHE_CHILD_STARTED", "process": identity(cache_child.pid),
            "command": cache_command, "started_unix": time.time(),
        })
        cache_return = cache_child.wait()
        if cache_return:
            raise RuntimeError("cache child " + str(cache_return))
        cache_dir = cache_root / "cache"
        cache_audit = audit_cache(cache_dir); write_new(stage / "cache_audit.json", cache_audit)

        phase = "FIXED100_TRAINING"
        require_idle_gpu(); train_dir = stage / "train"; train_dir.mkdir()
        train_command = [sys.executable, "-B", "-m", PACKAGE + ".train_child",
                         str(train_dir), str(cache_dir)]
        train_child = subprocess.Popen(train_command, cwd=str(project))
        write_new(stage / "training_launch.json", {
            "status": "TRAINING_CHILD_STARTED", "process": identity(train_child.pid),
            "command": train_command, "started_unix": time.time(),
        })
        train_return = train_child.wait()
        if train_return:
            raise RuntimeError("training child " + str(train_return))
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
            "status": "AUTHORIZED_FEEDBACK_TNL2K_FULL700_STARTED",
            "fixed_checkpoint_sha256": training_audit["final_checkpoint_sha256"],
            "numeric_npz_sha256": numeric_manifest["numeric_npz_sha256"],
            "calibration_status": calibration["status"],
            "calibration_sha256": calibration["calibration_sha256"],
            "threshold": calibration["threshold"], "reporting_label": calibration["reporting_label"],
            "sequences": 700, "frames": 516038, "searches": 515338,
            "persistent_workers": 8, "same_forward_native_shadow": True,
            "feedback_conditioned_native_shadow": True, "independent_native_baseline": False,
            "deterministic_source_execution": False,
            "determinism_mode": "cudnn_deterministic_cublas_workspace_warn_only_unsupported_cuda_cumsum",
            "automatic_retry": False,
            "started_unix": time.time(),
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
            "status": "FEEDBACK_PAIRWISE_PIPELINE_FAILED", "phase": phase,
            "error": repr(error), "traceback": traceback.format_exc(),
            "ended_unix": time.time(), "automatic_retry": False,
        })
        raise


if __name__ == "__main__":
    main(sys.argv)

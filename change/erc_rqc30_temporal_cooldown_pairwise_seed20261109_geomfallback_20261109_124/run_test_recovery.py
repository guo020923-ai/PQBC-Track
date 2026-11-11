"""Run one non-overwriting FULL700 recovery from the audited Round123 artifact."""

import json
import os
import sys
import time
import traceback
from pathlib import Path

from change.erc_rqc30_ordinal_risk_utility_100ep_20261105_56.run_authorized_pipeline import (
    identity, require_idle_gpu, write_new,
)
from .io_utils import digest


PACKAGE = "change.erc_rqc30_temporal_cooldown_pairwise_seed20261109_geomfallback_20261109_124"
SOURCE_CHECKPOINT_SHA256 = "354e3723cfe36a0ef861e16c93fbacd680ea548f5d508aeb9b9b72dd8c4baf53"
SOURCE_NUMERIC_SHA256 = "bc19147bc61babaa64090eef360eb7463ee56977456fbd53dcbc44ee62a4da92"


def verify_manifest(project, package):
    path = package / "implementation_manifest_124.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ROUND124_IMPLEMENTATION_SEALED":
        raise PermissionError("manifest")
    for row in manifest["files"]:
        target = project / "change" / row["package"] / row["name"]
        if target.stat().st_size != row["bytes"] or digest(target) != row["sha256"]:
            raise ValueError("implementation identity " + str(target))
    return manifest


def verify_source(source_stage):
    audit = json.loads((source_stage / "training_audit.json").read_text(encoding="utf-8"))
    numeric_dir = source_stage / "numeric"
    numeric = json.loads((numeric_dir / "manifest.json").read_text(encoding="utf-8"))
    calibration = json.loads((source_stage / "calibration_complete.json").read_text(encoding="utf-8"))
    if (audit.get("status") != "COMPLETE_VERIFIED" or
            audit.get("fixed_epoch") != 100 or audit.get("optimizer_steps") != 187500 or
            audit.get("final_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256 or
            digest(Path(audit["final_checkpoint"])) != SOURCE_CHECKPOINT_SHA256):
        raise ValueError("Round123 training identity")
    if (numeric.get("numeric_npz_sha256") != SOURCE_NUMERIC_SHA256 or
            digest(numeric_dir / "erc_epoch100_numeric.npz") != SOURCE_NUMERIC_SHA256 or
            numeric.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256):
        raise ValueError("Round123 numeric identity")
    if calibration.get("threshold") != 0.36725926399230957:
        raise ValueError("Round123 TRAIN-only threshold")
    return audit, numeric_dir, numeric, calibration


def main(argv):
    if len(argv) != 4:
        raise SystemExit("usage: run_test_recovery PROJECT SOURCE_STAGE NEW_STAGE")
    project, source_stage, stage = map(Path, argv[1:])
    package = project / "change" / PACKAGE.rsplit(".", 1)[-1]
    authorization = json.loads((package / "authorization_124.json").read_text(encoding="utf-8"))
    if authorization.get("status") != "AUTHORIZED_ENGINEERING_RECOVERY_TEST":
        raise PermissionError("authorization")
    implementation = verify_manifest(project, package)
    audit, numeric_dir, numeric, calibration = verify_source(source_stage)
    if stage.exists():
        raise FileExistsError(stage)
    require_idle_gpu()
    stage.mkdir()
    os.chdir(project)
    sys.path.insert(0, str(project))
    phase = "TNL2K_FULL700_RECOVERY_TEST"
    write_new(stage / "actual_launch.json", {
        "status": "ROUND124_GEOMETRY_FALLBACK_RECOVERY_LAUNCH",
        "process": identity(os.getpid()),
        "started_unix": time.time(),
        "source_stage": str(source_stage),
        "fixed_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "numeric_npz_sha256": SOURCE_NUMERIC_SHA256,
        "threshold": calibration["threshold"],
        "training_seed": 20261109,
        "temporal_cooldown_frames": 8,
        "invalid_feedback_policy": "post_selection_keep_fallback",
        "retrain": False,
        "sequences": 700,
        "frames": 516038,
        "searches": 515338,
        "automatic_retry": False,
        "implementation_manifest_sha256": digest(package / "implementation_manifest_124.json"),
        "sealed_file_count": len(implementation["files"]),
    })
    try:
        from .stage import run_stage
        result = run_stage(audit, numeric_dir, stage / "test", processes=8)
        result["source_numeric_manifest"] = numeric
        result["calibration_status"] = calibration["status"]
        result["threshold"] = calibration["threshold"]
        write_new(stage / "official_complete.json", result)
        write_new(stage / "terminal_status.json", {
            "status": "ROUND124_COMPLETE_PENDING_CONTROLLER_OS0_EXIT",
            "process": identity(os.getpid()),
            "ended_unix": time.time(),
            "expected_os_exit": 0,
            "automatic_retry": False,
        })
        print(json.dumps(result, sort_keys=True), flush=True)
    except BaseException as error:
        write_new(stage / "terminal_failure.json", {
            "status": "ROUND124_GEOMETRY_FALLBACK_RECOVERY_FAILED",
            "phase": phase,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "ended_unix": time.time(),
            "automatic_retry": False,
        })
        raise


if __name__ == "__main__":
    main(sys.argv)

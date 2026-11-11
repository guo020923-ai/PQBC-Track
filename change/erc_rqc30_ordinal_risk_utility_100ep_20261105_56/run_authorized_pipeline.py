"""One authorized ordinal-risk training, TRAIN calibration, and conditional test."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


PACKAGE = "change.erc_rqc30_ordinal_risk_utility_100ep_20261105_56"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def identity(pid):
    proc = Path("/proc") / str(pid)
    fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "tick": fields[19],
        "ppid": int(fields[1]),
        "state": fields[0],
        "cwd": os.readlink(proc / "cwd"),
        "cmdline": (proc / "cmdline")
        .read_bytes()
        .replace(b"\0", b" ")
        .decode()
        .strip(),
    }


def require_idle_gpu():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if output:
        raise RuntimeError("GPU occupied: " + output)


def verify_manifest(project, package):
    manifest = json.loads((package / "implementation_manifest_56.json").read_text())
    if manifest.get("status") != "IMPLEMENTATION_SEALED_AND_AUTHORIZED":
        raise PermissionError("sealed implementation manifest required")
    for row in manifest["files"]:
        path = project / "change" / row["package"] / row["name"]
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("implementation mismatch: " + str(path))
    return manifest


def audit_training(train, cache):
    import torch
    from change.erc_rqc30_ordinal_risk_utility_100ep_20261105_56.config import (
        PROTOCOL,
    )
    from change.erc_rqc30_ordinal_risk_utility_100ep_20261105_56.model import (
        VERSION,
    )

    completed = json.loads((train / "completed.json").read_text())
    checkpoints = sorted((train / "checkpoints").glob("ERC_epoch_*.pth.tar"))
    stats = [json.loads(row) for row in (train / "stats.jsonl").read_text().splitlines()]
    expected = [f"ERC_epoch_{epoch:04d}.pth.tar" for epoch in range(1, 101)]
    if completed.get("optimizer_steps") != 187500:
        raise ValueError("optimizer-step audit failed")
    if [path.name for path in checkpoints] != expected:
        raise ValueError("checkpoint continuity audit failed")
    if [row["epoch"] for row in stats] != list(range(1, 101)):
        raise ValueError("stats continuity audit failed")
    final = checkpoints[-1]
    saved = torch.load(str(final), map_location="cpu")
    if (
        saved.get("kind") != VERSION
        or saved.get("protocol") != PROTOCOL
        or saved.get("epoch") != 100
        or saved.get("optimizer_steps") != 187500
    ):
        raise ValueError("fixed Epoch100 identity audit failed")
    if completed.get("checkpoint_sha256") != digest(final):
        raise ValueError("checkpoint digest audit failed")
    if digest(cache / "manifest.json") != PROTOCOL["cache_manifest_sha256"]:
        raise ValueError("cache identity audit failed")
    return {
        "status": "COMPLETE_VERIFIED",
        "checkpoint_count": 100,
        "stats_count": 100,
        "optimizer_steps": 187500,
        "fixed_epoch": 100,
        "final_checkpoint": str(final),
        "final_checkpoint_sha256": digest(final),
        "final_checkpoint_bytes": final.stat().st_size,
        "cache_manifest_sha256": PROTOCOL["cache_manifest_sha256"],
        "source_rqc_epoch": 30,
        "source_rqc_checkpoint_sha256": PROTOCOL[
            "source_rqc_checkpoint_sha256"
        ],
        "checkpoint_selection": "fixed_epoch100_no_best",
        "training_seconds": completed["training_seconds"],
        "train_rows": completed["train_rows"],
        "calibration_rows": completed["calibration_rows"],
        "last_train_stats": stats[-1]["stats"],
        "audited_unix": time.time(),
    }


def main(argv):
    if len(argv) != 4:
        raise SystemExit("usage: run_authorized_pipeline PROJECT CACHE STAGE")
    project, cache, stage = map(Path, argv[1:])
    package = project / "change" / "erc_rqc30_ordinal_risk_utility_100ep_20261105_56"
    authorization = json.loads((package / "authorization_56.json").read_text())
    test_authorization = json.loads(
        (package / "posttrain_test_authorization_56.json").read_text()
    )
    if authorization.get("status") != "AUTHORIZED_ONE_ORDINAL_RISK_100EP_TRAINING":
        raise PermissionError("training authorization required")
    if (
        test_authorization.get("status")
        != "AUTHORIZED_ONE_TEST_AFTER_COMPLETE_VERIFIED_TRAINING_AND_ORDINAL_CALIBRATION"
    ):
        raise PermissionError("conditional test authorization required")
    manifest = verify_manifest(project, package)
    if stage.exists():
        raise FileExistsError(stage)
    require_idle_gpu()
    stage.mkdir()
    os.chdir(project)
    sys.path.insert(0, str(project))
    phase = "LAUNCH"
    write_new(
        stage / "actual_launch.json",
        {
            "status": "ACTUAL_ORDINAL_RISK_100EP_PIPELINE_LAUNCH",
            "process": identity(os.getpid()),
            "started_unix": time.time(),
            "training_epochs": 100,
            "optimizer_steps": 187500,
            "fixed_checkpoint": "Epoch100",
            "automatic_test_only_after_train_calibration": True,
            "test_dataset": "TNL2K_NLBBOX_FULL700_development_stage",
            "persistent_workers": 8,
            "automatic_retry": False,
            "parallel_gpu_job": False,
            "scientific_variable": "single_unified_ordinal_risk_score_for_selection_and_gating",
            "authorization_sha256": digest(package / "authorization_56.json"),
            "test_authorization_sha256": digest(
                package / "posttrain_test_authorization_56.json"
            ),
            "implementation_manifest_sha256": digest(
                package / "implementation_manifest_56.json"
            ),
            "sealed_file_count": len(manifest["files"]),
        },
    )
    try:
        phase = "TRAINING"
        train = stage / "train"
        train.mkdir()
        command = [
            sys.executable,
            "-B",
            "-m",
            PACKAGE + ".train_child",
            str(train),
            str(cache),
        ]
        child = subprocess.Popen(command, cwd=str(project))
        write_new(
            stage / "training_launch.json",
            {
                "status": "TRAINING_CHILD_STARTED",
                "process": identity(child.pid),
                "command": command,
                "started_unix": time.time(),
            },
        )
        return_code = child.wait()
        if return_code:
            raise RuntimeError("training child exited " + str(return_code))

        phase = "TRAINING_AUDIT"
        audit = audit_training(train, cache)
        write_new(stage / "training_audit.json", audit)

        phase = "NUMERIC_EXPORT"
        from change.erc_rqc30_ordinal_risk_utility_100ep_20261105_56.checkpoint import (
            export_numeric,
        )

        numeric = stage / "numeric"
        numeric_manifest = export_numeric(audit["final_checkpoint"], numeric)

        phase = "TRAIN_ORDINAL_CALIBRATION"
        from change.erc_rqc30_ordinal_risk_utility_100ep_20261105_56.calibration import (
            calibrate,
        )

        calibration = calibrate(numeric, cache)
        write_new(stage / "calibration_complete.json", calibration)

        phase = "TNL2K_FULL700_TEST"
        require_idle_gpu()
        write_new(
            stage / "test_launch.json",
            {
                "status": "AUTHORIZED_TNL2K_FULL700_DEVELOPMENT_TEST_STARTED",
                "fixed_checkpoint_sha256": audit["final_checkpoint_sha256"],
                "numeric_npz_sha256": numeric_manifest["numeric_npz_sha256"],
                "ordinal_calibration_sha256": calibration["calibration_sha256"],
                "risk_threshold": calibration["threshold"],
                "sequences": 700,
                "frames": 516038,
                "searches": 515338,
                "persistent_workers": 8,
                "same_forward_native_shadow": True,
                "started_unix": time.time(),
                "automatic_retry": False,
            },
        )
        from change.erc_rqc30_ordinal_risk_utility_test_20261105_57.stage import (
            run_stage,
        )

        result = run_stage(audit, numeric, stage / "test", processes=8)
        write_new(stage / "official_complete.json", result)
        write_new(
            stage / "terminal_status.json",
            {
                "status": "PIPELINE_COMPLETE_PENDING_CONTROLLER_OS0_EXIT",
                "process": identity(os.getpid()),
                "ended_unix": time.time(),
                "expected_os_exit": 0,
                "automatic_retry": False,
            },
        )
        print(json.dumps(result, sort_keys=True), flush=True)
    except BaseException as error:
        write_new(
            stage / "terminal_failure.json",
            {
                "status": "ORDINAL_RISK_PIPELINE_FAILED",
                "phase": phase,
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "ended_unix": time.time(),
                "automatic_retry": False,
            },
        )
        raise


if __name__ == "__main__":
    main(sys.argv)

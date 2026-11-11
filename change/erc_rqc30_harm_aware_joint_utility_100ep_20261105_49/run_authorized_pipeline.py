"""One authorized fixed-100 training followed by one TNL2K FULL700 test."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


PACKAGE = "change.erc_rqc30_harm_aware_joint_utility_100ep_20261105_49"
TEST_PACKAGE = "change.erc_rqc30_harm_aware_joint_utility_test_20261105_50"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def replace_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def process_identity(pid):
    process = Path("/proc") / str(pid)
    fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid, "tick": fields[19], "ppid": int(fields[1]),
        "state": fields[0], "cwd": os.readlink(process / "cwd"),
        "cmdline": (process / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip(),
    }


def require_idle_gpu():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if output:
        raise RuntimeError("GPU is already occupied by compute PIDs: " + output)


def verify_implementation(project, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("status") != "IMPLEMENTATION_SEALED_AND_AUTHORIZED":
        raise PermissionError("sealed implementation manifest required")
    for row in manifest.get("files", []):
        path = Path(project) / "change" / row["package"] / row["name"]
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError("implementation size mismatch: " + str(path))
        if digest(path) != row["sha256"]:
            raise ValueError("implementation SHA mismatch: " + str(path))
    return manifest


def audit_training(train_dir, cache_dir):
    import torch
    from change.erc_rqc30_harm_aware_joint_utility_100ep_20261105_49.config import PROTOCOL

    completed = json.loads((train_dir / "completed.json").read_text())
    if completed.get("fixed_epoch") != 100 or completed.get("optimizer_steps") != 187500:
        raise ValueError("training terminal budget mismatch")
    checkpoints = sorted((train_dir / "checkpoints").glob("ERC_epoch_*.pth.tar"))
    expected = [f"ERC_epoch_{epoch:04d}.pth.tar" for epoch in range(1, 101)]
    if [path.name for path in checkpoints] != expected:
        raise ValueError("100 continuous checkpoints required")
    stats = [json.loads(line) for line in (train_dir / "stats.jsonl").read_text().splitlines()]
    if [row.get("epoch") for row in stats] != list(range(1, 101)):
        raise ValueError("100 continuous stats rows required")
    final = checkpoints[-1]
    final_sha = digest(final)
    if completed.get("checkpoint_sha256") != final_sha:
        raise ValueError("final checkpoint SHA mismatch")
    saved = torch.load(str(final), map_location="cpu")
    if saved.get("kind") != "rqc30_harm_aware_joint_utility_100ep_v1":
        raise ValueError("final checkpoint kind mismatch")
    if saved.get("protocol") != PROTOCOL or saved.get("epoch") != 100:
        raise ValueError("final protocol or epoch mismatch")
    if saved.get("optimizer_steps") != 187500:
        raise ValueError("final optimizer budget mismatch")
    cache_sha = digest(cache_dir / "manifest.json")
    if cache_sha != PROTOCOL["cache_manifest_sha256"]:
        raise ValueError("sealed TRAIN cache changed")
    return {
        "status": "COMPLETE_VERIFIED",
        "checkpoint_count": 100,
        "stats_count": 100,
        "optimizer_steps": 187500,
        "fixed_epoch": 100,
        "final_checkpoint": str(final),
        "final_checkpoint_sha256": final_sha,
        "final_checkpoint_bytes": final.stat().st_size,
        "cache_manifest_sha256": cache_sha,
        "source_rqc_epoch": 30,
        "source_rqc_checkpoint_sha256": PROTOCOL["source_rqc_checkpoint_sha256"],
        "checkpoint_selection": "fixed_epoch100_no_best",
        "training_seconds": completed["training_seconds"],
        "last_train_stats": stats[-1]["stats"],
        "audited_unix": time.time(),
    }


def main(argv):
    if len(argv) != 4:
        raise SystemExit("usage: run_authorized_pipeline.py PROJECT CACHE_DIR STAGE")
    project, cache_dir, stage = map(Path, argv[1:])
    package_dir = project / "change" / "erc_rqc30_harm_aware_joint_utility_100ep_20261105_49"
    authorization = json.loads((package_dir / "authorization_49.json").read_text())
    test_authorization = json.loads((package_dir / "posttrain_test_authorization_49.json").read_text())
    if authorization.get("status") != "AUTHORIZED":
        raise PermissionError("fixed training authorization required")
    if test_authorization.get("status") != "AUTHORIZED_AFTER_COMPLETE_VERIFIED_TRAINING_AUDIT":
        raise PermissionError("post-training test authorization required")
    verify_implementation(project, package_dir / "implementation_manifest_49.json")
    if stage.exists():
        raise FileExistsError("fresh pipeline stage required: " + str(stage))
    require_idle_gpu()
    stage.mkdir(parents=False)
    os.chdir(project)
    sys.path.insert(0, str(project))
    phase = "LAUNCH"
    write_new(stage / "actual_launch.json", {
        "status": "ACTUAL_HARM_AWARE_ERC100_PIPELINE_LAUNCH",
        "process": process_identity(os.getpid()),
        "started_unix": time.time(),
        "training_epochs": 100, "optimizer_steps": 187500,
        "fixed_checkpoint": "Epoch100", "automatic_test": True,
        "test_dataset": "TNL2K_NLBBOX_FULL700", "persistent_workers": 8,
        "automatic_retry": False, "parallel_gpu_job": False,
        "source_rqc_checkpoint_sha256": authorization["source_rqc_checkpoint_sha256"],
        "cache_manifest_sha256": authorization["sealed_train_cache_manifest_sha256"],
        "authorization_sha256": digest(package_dir / "authorization_49.json"),
        "test_authorization_sha256": digest(package_dir / "posttrain_test_authorization_49.json"),
        "implementation_manifest_sha256": digest(package_dir / "implementation_manifest_49.json"),
    })
    try:
        phase = "TRAINING"
        train_dir = stage / "train"
        train_dir.mkdir()
        command = [
            sys.executable, "-B", "-m", PACKAGE + ".train_child",
            str(train_dir), str(cache_dir),
        ]
        child = subprocess.Popen(command, cwd=str(project))
        write_new(stage / "training_launch.json", {
            "status": "TRAINING_CHILD_STARTED", "process": process_identity(child.pid),
            "command": command, "started_unix": time.time(),
        })
        return_code = child.wait()
        if return_code != 0:
            raise RuntimeError("training child exited " + str(return_code))

        phase = "TRAINING_AUDIT"
        audit = audit_training(train_dir, cache_dir)
        write_new(stage / "training_audit.json", audit)

        phase = "NUMERIC_EXPORT"
        from change.erc_rqc30_harm_aware_joint_utility_100ep_20261105_49.checkpoint import export_numeric
        numeric_dir = stage / "numeric"
        numeric_manifest = export_numeric(audit["final_checkpoint"], numeric_dir)
        write_new(stage / "numeric_export_complete.json", {
            "status": "NUMERIC_EXPORT_VERIFIED",
            "source_checkpoint_sha256": numeric_manifest["source_checkpoint_sha256"],
            "numeric_npz_sha256": numeric_manifest["numeric_npz_sha256"],
            "completed_unix": time.time(),
        })

        phase = "TNL2K_FULL700_TEST"
        require_idle_gpu()
        write_new(stage / "test_launch.json", {
            "status": "AUTHORIZED_TNL2K_FULL700_TEST_STARTED",
            "fixed_checkpoint_sha256": audit["final_checkpoint_sha256"],
            "numeric_npz_sha256": numeric_manifest["numeric_npz_sha256"],
            "sequences": 700, "frames": 516038, "searches": 515338,
            "persistent_workers": 8, "same_forward_native_shadow": True,
            "started_unix": time.time(), "automatic_retry": False,
        })
        from change.erc_rqc30_harm_aware_joint_utility_test_20261105_50.stage import run_stage
        result = run_stage(audit, numeric_dir, stage / "test", processes=8)
        write_new(stage / "official_complete.json", result)
        write_new(stage / "terminal_status.json", {
            "status": "PIPELINE_COMPLETE_PENDING_CONTROLLER_OS0_EXIT",
            "process": process_identity(os.getpid()), "ended_unix": time.time(),
            "expected_os_exit": 0, "automatic_retry": False,
        })
        print(json.dumps(result, sort_keys=True), flush=True)
    except BaseException as error:
        write_new(stage / "terminal_failure.json", {
            "status": "HARM_AWARE_ERC100_PIPELINE_FAILED",
            "phase": phase, "error": repr(error),
            "traceback": traceback.format_exc(), "ended_unix": time.time(),
            "automatic_retry": False,
        })
        raise


if __name__ == "__main__":
    main(sys.argv)

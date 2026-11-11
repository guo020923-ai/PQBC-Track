"""Run frozen Round124 generalization sequentially, with no retry or tuning."""

import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


PROJECT = Path("/root/autodl-tmp/UVLTrack-master/change/rqc_jq_20260929_isolated")
ROOT = Path("/root/autodl-tmp/UVLTrack-master/artifacts/pqbc_final/stages/erc_round124_generalization_20261109_125")
PACKAGE = "change.erc_round124_generalization_20261109_125.run_dataset"
ADAPTER_MANIFEST = Path(__file__).with_name("manifest.json")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def main():
    ROOT.mkdir(parents=False, exist_ok=False)
    write_new(ROOT / "launch.json", {
        "status": "ROUND124_GENERALIZATION_SEQUENCE_STARTED",
        "order": ["lasot", "otb99"], "automatic_retry": False,
        "dataset_specific_tuning": False, "retraining": False,
        "started_unix": time.time(), "controller_pid": os.getpid(),
        "adapter_manifest_sha256": digest(ADAPTER_MANIFEST),
    })
    try:
        for dataset in ("lasot", "otb99"):
            command = [sys.executable, "-B", "-m", PACKAGE, dataset, str(ROOT / dataset)]
            write_new(ROOT / (dataset + "_launch.json"), {
                "dataset": dataset, "command": command, "started_unix": time.time(),
            })
            code = subprocess.call(command, cwd=str(PROJECT))
            write_new(ROOT / (dataset + "_exit.json"), {
                "dataset": dataset, "exit_code": code, "ended_unix": time.time(),
            })
            if code != 0:
                raise RuntimeError(dataset + " exited " + str(code))
        write_new(ROOT / "complete.json", {
            "status": "ROUND124_GENERALIZATION_SEQUENCE_COMPLETE",
            "order": ["lasot", "otb99"], "ended_unix": time.time(),
        })
        write_new(ROOT / "terminal_status.json", {
            "status": "COMPLETE_PENDING_CONTROLLER_OS0_EXIT",
            "controller_pid": os.getpid(), "expected_os_exit": 0,
            "automatic_retry": False, "ended_unix": time.time(),
        })
    except BaseException as error:
        write_new(ROOT / "terminal_failure.json", {
            "status": "FAILED", "error_type": type(error).__name__,
            "error": str(error), "traceback": traceback.format_exc(),
            "ended_unix": time.time(),
        })
        raise


if __name__ == "__main__":
    main()


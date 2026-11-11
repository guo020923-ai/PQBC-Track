"""Run frozen Round73 generalization sequentially, with no retry or tuning."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "artifacts/pqbc_final/stages/erc_round73_generalization_20261108_122"
PACKAGE = "change.erc_round73_generalization_20261108_122.run_dataset"


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def main():
    ROOT.mkdir(parents=False, exist_ok=False)
    write_new(ROOT / "launch.json", {
        "status": "ROUND73_GENERALIZATION_SEQUENCE_STARTED",
        "order": ["lasot", "otb99"], "automatic_retry": False,
        "dataset_specific_tuning": False, "started_unix": time.time(),
    })
    for dataset in ("lasot", "otb99"):
        command = [sys.executable, "-B", "-m", PACKAGE, dataset, str(ROOT / dataset)]
        write_new(ROOT / (dataset + "_launch.json"), {
            "dataset": dataset, "command": command, "started_unix": time.time()
        })
        code = subprocess.call(command, cwd=str(PROJECT))
        write_new(ROOT / (dataset + "_exit.json"), {
            "dataset": dataset, "exit_code": code, "ended_unix": time.time()
        })
        if code != 0:
            raise SystemExit(code)
    write_new(ROOT / "complete.json", {
        "status": "ROUND73_GENERALIZATION_SEQUENCE_COMPLETE",
        "order": ["lasot", "otb99"], "ended_unix": time.time(),
    })


if __name__ == "__main__":
    main()


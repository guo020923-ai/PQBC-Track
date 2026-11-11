"""File-backed spawn-safe entrypoint for the one authorized official test."""

import json
import os
from pathlib import Path
import sys
import time
import traceback


def _process(pid):
    process = Path("/proc") / str(pid)
    fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid, "tick": fields[19], "ppid": int(fields[1]),
        "state": fields[0], "cwd": os.readlink(process / "cwd"),
        "cmdline": (process / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip(),
    }


def _write_new(path, value):
    with Path(path).open("x", encoding="utf8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())


def main(argv):
    if len(argv) != 5:
        raise SystemExit("usage: run_stage.py PROJECT NUMERIC_DIR STAGE AUDIT")
    project, numeric_dir, stage, audit_path = map(Path, argv[1:])
    ready = json.loads((stage / "execution_ready.json").read_text())
    audit = json.loads(audit_path.read_text())
    if ready.get("status") != "TEST_EXECUTION_READY" or (stage / "actual_launch.json").exists():
        raise RuntimeError("fresh ready stage required")
    sys.path.insert(0, str(project)); os.chdir(project)
    launch = {
        "status": "ACTUAL_FAST_TNL2K_LAUNCH",
        "tag": "rqc30_epoch_supervised_abstention_erc30_tnl110442_20261104_42",
        "process": _process(os.getpid()), "started_unix": time.time(),
        "runid": 110442, "persistent_workers": 8,
        "baseline_gpu_replay": False, "automatic_retry": False,
        "same_forward_native_shadow": True,
        "file_backed_spawn_entrypoint": True,
    }
    _write_new(stage / "actual_launch.json", launch)
    try:
        from change.erc_rqc30_epoch_supervised_abstention_test_20261104_42.stage import run_stage
        result = run_stage(audit, numeric_dir, stage / "run", processes=8)
        _write_new(stage / "official_complete.json", result)
        _write_new(stage / "terminal_status.json", {
            "status": "FAST_TNL2K_OS0_PENDING_PROCESS_EXIT",
            "process": _process(os.getpid()), "ended_unix": time.time(),
            "expected_os_exit": 0,
        })
        print(json.dumps(result, sort_keys=True), flush=True)
    except BaseException as error:
        _write_new(stage / "terminal_failure.json", {
            "status": "FAST_TNL2K_FAILED", "error": repr(error),
            "traceback": traceback.format_exc(), "ended_unix": time.time(),
            "automatic_retry": False,
        })
        raise


if __name__ == "__main__":
    main(sys.argv)

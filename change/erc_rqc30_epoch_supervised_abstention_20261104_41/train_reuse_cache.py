"""One v2 training launch from the sealed v1 TRAIN cache; no test or retry."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from .config import PROTOCOL, validate_protocol
from .io_utils import digest, write_new_json
from .training import train_epochs


EXPECTED_CACHE_MANIFEST_SHA256 = "993b20ade7aa8ab491c5d87fcdc58350d7f218dea87b470bd9b8f6c58711b56c"


def _protocol_digest():
    payload = json.dumps(
        PROTOCOL, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf8")
    return hashlib.sha256(payload).hexdigest()


def _process(pid):
    process = Path("/proc") / str(pid)
    fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "tick": fields[19],
        "ppid": int(fields[1]),
        "state": fields[0],
        "cwd": os.readlink(process / "cwd"),
        "cmdline": (process / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args(argv)
    output, cache = Path(args.output), Path(args.cache)
    authorization = json.loads(Path(args.authorization).read_text(encoding="utf8"))
    validate_protocol()
    if authorization.get("scope") != "one_fresh_rqc_style_erc30_v2_from_frozen_train_cache":
        raise PermissionError("wrong authorization scope")
    if authorization.get("explicit_user_training_authorization") is not True:
        raise PermissionError("explicit training authorization required")
    if authorization.get("maximum_training_launches") != 1:
        raise PermissionError("one training launch required")
    if authorization.get("automatic_test") is not False or authorization.get("automatic_retry") is not False:
        raise PermissionError("automatic continuation prohibited")
    if authorization.get("protocol_sha256") != _protocol_digest():
        raise PermissionError("authorization protocol digest mismatch")
    if authorization.get("cache_manifest_sha256") != EXPECTED_CACHE_MANIFEST_SHA256:
        raise PermissionError("authorization cache identity mismatch")
    if output.exists():
        raise FileExistsError("fresh output required: " + str(output))
    manifest_path = cache / "manifest.json"
    if digest(manifest_path) != EXPECTED_CACHE_MANIFEST_SHA256:
        raise ValueError("sealed TRAIN cache manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    if manifest.get("schema") != "rqc30_epoch_erc_cache_v1" or manifest.get("examples") != 30000:
        raise ValueError("incompatible cache schema")
    if manifest.get("train_only") is not True or manifest.get("fixed_rqc_trajectory") is not True:
        raise PermissionError("TRAIN-only fixed cache required")
    old_protocol = manifest.get("protocol", {})
    immutable = (
        "seed", "source_rqc_epoch", "source_rqc_checkpoint_sha256",
        "data_configuration", "datasets_name", "datasets_ratio",
        "erc_active_population", "cache_examples", "cache_warmup_frames",
        "cache_frames_per_clip", "candidate_actions", "evidence_dim",
        "frozen_rqc", "frozen_head", "frozen_backbone", "test_data_in_training",
    )
    if any(old_protocol.get(key) != PROTOCOL.get(key) for key in immutable):
        raise ValueError("cache-affecting protocol field changed")
    output.mkdir(parents=False)
    write_new_json(output / "actual_launch.json", {
        "status": "ACTUAL_V2_TRAIN_LAUNCH",
        "tag": "rqc30_epoch_supervised_abstention_erc30_20261104_41",
        "process": _process(os.getpid()),
        "started_unix": time.time(),
        "cache_reused": True,
        "cache_examples": 30000,
        "maximum_training_launches": 1,
        "automatic_test": False,
        "automatic_retry": False,
    })
    write_new_json(output / "authorization.json", authorization)
    write_new_json(output / "protocol.json", PROTOCOL)
    write_new_json(output / "cache_binding.json", {
        "status": "SEALED_TRAIN_CACHE_REUSED",
        "path": str(cache),
        "manifest_sha256": EXPECTED_CACHE_MANIFEST_SHA256,
        "examples": 30000,
        "shards": len(manifest["shards"]),
        "train_only": True,
        "cache_affecting_fields_equal": True,
    })
    try:
        result = train_epochs(output, cache_dir=cache)
        write_new_json(output / "terminal_status.json", {
            "status": "V2_TRAIN_OS0_PENDING_PROCESS_EXIT",
            "process": _process(os.getpid()),
            "ended_unix": time.time(),
            "expected_os_exit": 0,
            "automatic_test": False,
        })
        print(json.dumps(result, sort_keys=True), flush=True)
    except BaseException as error:
        write_new_json(output / "failed.json", {
            "status": "V2_TRAIN_FAILED",
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "ended_unix": time.time(),
            "automatic_retry": False,
            "automatic_test": False,
        })
        raise


if __name__ == "__main__":
    main(sys.argv[1:])

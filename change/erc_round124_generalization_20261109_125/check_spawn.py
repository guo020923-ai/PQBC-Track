"""CPU-only import and spawn validation for the Round125 adapter."""

import json
import multiprocessing


def child(value):
    from .fast_runner import AuditedRound124Worker
    from .run_dataset import CHECKPOINT_SHA256, THRESHOLD

    return {
        "value": value,
        "worker_class": AuditedRound124Worker.__name__,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "threshold": THRESHOLD,
    }


def main():
    context = multiprocessing.get_context("spawn")
    with context.Pool(1) as pool:
        result = pool.map(child, [125])
    if result[0]["value"] != 125:
        raise RuntimeError("spawn result mismatch")
    print(json.dumps({"status": "ROUND125_SPAWN_PASS", "result": result[0]}, sort_keys=True))


if __name__ == "__main__":
    main()


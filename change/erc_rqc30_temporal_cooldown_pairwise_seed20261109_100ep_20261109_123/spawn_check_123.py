import json
import multiprocessing as mp


def child(queue):
    from .config import PROTOCOL, validate_protocol

    validate_protocol()
    queue.put({
        "training_seed": PROTOCOL["training_seed"],
        "epochs": PROTOCOL["epochs"],
        "optimizer_steps": PROTOCOL["optimizer_steps"],
        "cooldown_frames": PROTOCOL["temporal_cooldown_frames"],
    })


def main():
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=child, args=(queue,))
    process.start()
    process.join(30)
    if process.exitcode != 0:
        raise RuntimeError(f"spawn child exit {process.exitcode}")
    identity = queue.get(timeout=5)
    expected = {
        "training_seed": 20261109,
        "epochs": 100,
        "optimizer_steps": 187500,
        "cooldown_frames": 8,
    }
    if identity != expected:
        raise ValueError(identity)
    print(json.dumps({"status": "SPAWN_PASS", **identity}, sort_keys=True))


if __name__ == "__main__":
    main()

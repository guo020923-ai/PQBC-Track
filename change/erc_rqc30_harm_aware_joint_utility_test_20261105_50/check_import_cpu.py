"""CPU-only import admission for the complete inherited evaluation stack."""

import json
import torch


def main():
    if torch.cuda.is_initialized():
        raise RuntimeError("test import admission must not initialize CUDA")
    from . import fast_runner, stage
    assert stage.SEQUENCES == 700
    assert stage.FRAMES == 516038
    assert stage.SEARCHES == 515338
    assert fast_runner.PersistentHarmAwareWorker.__module__ == __package__ + ".fast_runner"
    print(json.dumps({
        "status": "TEST_STACK_CPU_IMPORT_PASS",
        "sequences": stage.SEQUENCES,
        "frames": stage.FRAMES,
        "searches": stage.SEARCHES,
        "spawn_worker_module": fast_runner.PersistentHarmAwareWorker.__module__,
        "cuda_initialized": torch.cuda.is_initialized(),
        "gpu_operations": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

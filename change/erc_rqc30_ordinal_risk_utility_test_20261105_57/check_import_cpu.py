import json

import torch


def main():
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU only")
    from . import fast_runner, stage

    assert (
        fast_runner.PersistentOrdinalRiskWorker.__module__
        == __package__ + ".fast_runner"
    )
    assert stage.inherited.run_persistent_dataset is fast_runner.run_persistent_dataset
    print(
        json.dumps(
            {
                "status": "ORDINAL_RISK_TEST_IMPORT_CPU_PASS",
                "worker_module": fast_runner.PersistentOrdinalRiskWorker.__module__,
                "gpu_operations": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

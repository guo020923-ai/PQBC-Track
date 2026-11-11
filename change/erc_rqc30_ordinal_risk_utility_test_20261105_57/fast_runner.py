"""Spawn-safe persistent workers for unified ordinal-risk ERC."""

import atexit
import copy
import multiprocessing
from pathlib import Path
import time
from types import SimpleNamespace

from change.erc_rqc30_epoch_supervised_abstention_test_20261104_42 import fast_runner as legacy


BASE = legacy.BASE
_WORKER = None


class PersistentOrdinalRiskWorker(legacy.PersistentEpochWorker):
    def __init__(self, config):
        import cv2
        import torch
        from lib.test.evaluation.running import _save_tracker_output
        from change.erc_natural_reuse_fresh_20261021_01.bridge import (
            collector_class,
            load_legacy,
        )
        from change.erc_rqc30_epoch_supervised_abstention_20261104_41.provider import (
            causal_candidate_provider,
        )
        from change.erc_rqc30_ordinal_risk_utility_100ep_20261105_56.checkpoint import (
            load_numeric,
        )
        from change.erc_rqc30_ordinal_risk_utility_100ep_20261105_56.runtime import (
            OutputOnlyOrdinalRiskRuntime,
        )

        self.config = dict(config)
        torch.cuda.set_device(int(self.config.get("gpu_id", 0)))
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(1)
        torch.backends.cudnn.benchmark = True
        params = copy.deepcopy(load_legacy().infrastructure.parameters())
        if params.cfg.TEST.MODE != self.config["mode"]:
            raise PermissionError("RQC30 test mode mismatch")
        params.cfg = copy.deepcopy(params.cfg)
        params.cfg.TEST.EPOCH = int(self.config["source_epoch"])
        params.save_all_boxes = False
        self.source = collector_class()(params, self.config["dataset_name"])
        self.policy, self.policy_manifest = load_numeric(
            self.config["checkpoint_dir"], device="cuda"
        )
        self.policy.eval().requires_grad_(False)
        self.runtime = OutputOnlyOrdinalRiskRuntime(
            self.source,
            self.policy,
            causal_candidate_provider,
            Path(self.config["checkpoint_dir"]) / "ordinal_calibration.json",
        )
        writer_params = copy.deepcopy(params)
        writer_params.cfg = copy.deepcopy(params.cfg)
        writer_params.cfg.TEST.EPOCH = int(self.config["writer_epoch"])
        self.writer = SimpleNamespace(
            results_dir=self.config["results_dir"], params=writer_params
        )
        self.native_writer = SimpleNamespace(
            results_dir=self.config["native_results_dir"], params=writer_params
        )
        self.save_output = _save_tracker_output
        self.cv2, self.torch = cv2, torch
        self.sequences = self.frames = self.searches = 0
        atexit.register(self.close)


def _initialize_worker(config):
    global _WORKER
    if _WORKER is not None:
        raise RuntimeError("worker initialized twice")
    _WORKER = PersistentOrdinalRiskWorker(config)


def _run_one(sequence):
    if _WORKER is None:
        raise RuntimeError("worker not initialized")
    return _WORKER.run(sequence)


def run_persistent_dataset(dataset, config, processes=8):
    config = dict(config)
    required = {
        "checkpoint_dir",
        "results_dir",
        "native_results_dir",
        "scope_root",
        "dataset_name",
        "mode",
        "source_epoch",
        "writer_epoch",
        "gpu_id",
    }
    if set(config) != required:
        raise ValueError("exact config schema required")
    if any(
        Path(config[key]).exists()
        for key in ("results_dir", "native_results_dir", "scope_root")
    ):
        raise FileExistsError("fresh roots required")
    scope = Path(config["scope_root"])
    scope.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    receipts, recent = [], []
    total_frames = sum(len(sequence.frames) for sequence in dataset)
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        processes=int(processes),
        initializer=_initialize_worker,
        initargs=(config,),
        maxtasksperchild=None,
    ) as pool:
        last_time, last_frames = began, 0
        for receipt in pool.imap_unordered(_run_one, dataset, chunksize=1):
            receipts.append(receipt)
            now = time.monotonic()
            completed_frames = sum(row["frames"] for row in receipts)
            delta_frames = completed_frames - last_frames
            if delta_frames > 0:
                recent.append((now - last_time) / delta_frames)
                recent = recent[-50:]
            last_time, last_frames = now, completed_frames
            if (
                len(receipts) == 1
                or len(receipts) % 5 == 0
                or len(receipts) == len(dataset)
            ):
                seconds_per_frame = (now - began) / max(completed_frames, 1)
                ordered = sorted(recent)
                low = (
                    ordered[max(0, int(0.2 * (len(ordered) - 1)))]
                    if ordered
                    else seconds_per_frame
                )
                high = (
                    ordered[min(len(ordered) - 1, int(0.8 * (len(ordered) - 1)))]
                    if ordered
                    else seconds_per_frame
                )
                remaining = total_frames - completed_frames
                BASE._write_json_replace(
                    scope / "progress.json",
                    {
                        "phase": "TNL2K_PREDICTIONS",
                        "completed_sequences": len(receipts),
                        "total_sequences": len(dataset),
                        "completed_frames": completed_frames,
                        "total_frames": total_frames,
                        "elapsed_seconds": now - began,
                        "frames_per_second": completed_frames / max(now - began, 1e-9),
                        "eta_seconds": remaining * seconds_per_frame,
                        "eta_seconds_low": remaining * low,
                        "eta_seconds_high": remaining * high,
                        "persistent_workers": int(processes),
                        "updated_unix": time.time(),
                    },
                )
    if (
        len(receipts) != len(dataset)
        or sum(row["frames"] for row in receipts) != total_frames
    ):
        raise RuntimeError("incomplete test")
    result = {
        "status": "PREDICTIONS_COMPLETE_PENDING_OFFICIAL_METRICS",
        "sequences": len(receipts),
        "frames": total_frames,
        "searches": total_frames - len(receipts),
        "worker_processes": int(processes),
        "maximum_heavy_source_loads": int(processes),
        "eliminated_heavy_source_loads": len(dataset) - int(processes),
        "baseline_gpu_forwards": 0,
        "per_frame_audit_writes": 0,
        "same_forward_native_shadow": True,
        "seconds": time.monotonic() - began,
    }
    BASE._write_json_new(scope / "complete.json", result)
    return result

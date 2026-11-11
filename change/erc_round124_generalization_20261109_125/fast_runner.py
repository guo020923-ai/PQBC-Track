"""Persistent Round124 runner with sequence-level fallback-count receipts."""

import multiprocessing
import time
from pathlib import Path

from change.erc_rqc30_temporal_cooldown_pairwise_seed20261109_geomfallback_20261109_124 import fast_runner as source


BASE = source.BASE
_WORKER = None


class AuditedRound124Worker(source.PersistentExecutionAlignedPairwiseWorker):
    def run(self, sequence):
        receipt = super().run(sequence)
        audit = {
            "sequence": sequence.name,
            "invalid_feedback_fallbacks": int(self.runtime.invalid_feedback_fallbacks),
            "feedback_commits": int(self.runtime.commits),
            "suppressed_nonkeep_proposals": int(self.runtime.suppressed_nonkeep_proposals),
            "invalid_feedback_policy": "post_selection_keep_fallback",
        }
        fallback_root = Path(self.config["scope_root"]).parent / "fallback_counts"
        BASE._write_json_new(fallback_root / (sequence.name + ".json"), audit)
        receipt.update(audit)
        return receipt


def _initialize_worker(config):
    global _WORKER
    if _WORKER is not None:
        raise RuntimeError("worker initialized twice")
    _WORKER = AuditedRound124Worker(config)


def _run_one(sequence):
    if _WORKER is None:
        raise RuntimeError("worker not initialized")
    return _WORKER.run(sequence)


def run_persistent_dataset(dataset, config, processes=8):
    config = dict(config)
    required = {
        "checkpoint_dir", "results_dir", "native_results_dir", "scope_root",
        "dataset_name", "mode", "source_epoch", "writer_epoch", "gpu_id",
    }
    if set(config) != required:
        raise ValueError("exact Round124 generalization config schema required")
    scope = Path(config["scope_root"])
    fallback_root = scope.parent / "fallback_counts"
    if any(path.exists() for path in (
        Path(config["results_dir"]), Path(config["native_results_dir"]),
        scope, fallback_root,
    )):
        raise FileExistsError("fresh current, native, scope, and fallback roots required")
    scope.mkdir(parents=True, exist_ok=False)
    fallback_root.mkdir(parents=False, exist_ok=False)

    began = time.monotonic()
    receipts, recent = [], []
    total_frames = sum(len(sequence.frames) for sequence in dataset)
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        processes=int(processes), initializer=_initialize_worker,
        initargs=(config,), maxtasksperchild=None,
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
            if len(receipts) == 1 or len(receipts) % 5 == 0 or len(receipts) == len(dataset):
                seconds_per_frame = (now - began) / max(completed_frames, 1)
                ordered = sorted(recent)
                low = ordered[max(0, int(0.2 * (len(ordered) - 1)))] if ordered else seconds_per_frame
                high = ordered[min(len(ordered) - 1, int(0.8 * (len(ordered) - 1)))] if ordered else seconds_per_frame
                remaining = total_frames - completed_frames
                BASE._write_json_replace(scope / "progress.json", {
                    "phase": "GENERALIZATION_PREDICTIONS",
                    "completed_sequences": len(receipts), "total_sequences": len(dataset),
                    "completed_frames": completed_frames, "total_frames": total_frames,
                    "elapsed_seconds": now - began,
                    "frames_per_second": completed_frames / max(now - began, 1e-9),
                    "eta_seconds": remaining * seconds_per_frame,
                    "eta_seconds_low": remaining * low,
                    "eta_seconds_high": remaining * high,
                    "persistent_workers": int(processes),
                    "temporal_cooldown_frames": 8,
                    "invalid_feedback_policy": "post_selection_keep_fallback",
                    "invalid_feedback_fallbacks_completed": sum(
                        int(row["invalid_feedback_fallbacks"]) for row in receipts
                    ),
                    "updated_unix": time.time(),
                })
    if len(receipts) != len(dataset) or sum(row["frames"] for row in receipts) != total_frames:
        raise RuntimeError("incomplete persistent generalization test")
    if len(list(fallback_root.glob("*.json"))) != len(dataset):
        raise RuntimeError("fallback audit receipt closure failed")
    result = {
        "status": "PREDICTIONS_COMPLETE_PENDING_OFFICIAL_METRICS",
        "sequences": len(receipts), "frames": total_frames,
        "searches": total_frames - len(receipts),
        "worker_processes": int(processes),
        "maximum_heavy_source_loads": int(processes),
        "eliminated_heavy_source_loads": len(dataset) - int(processes),
        "baseline_gpu_forwards": 0, "per_frame_audit_writes": 0,
        "same_forward_native_shadow": True,
        "feedback_conditioned_native_shadow": True,
        "independent_native_baseline": False,
        "temporal_cooldown_frames": 8,
        "invalid_feedback_policy": "post_selection_keep_fallback",
        "invalid_feedback_fallbacks": sum(int(row["invalid_feedback_fallbacks"]) for row in receipts),
        "feedback_commits": sum(int(row["feedback_commits"]) for row in receipts),
        "suppressed_nonkeep_proposals": sum(
            int(row["suppressed_nonkeep_proposals"]) for row in receipts
        ),
        "seconds": time.monotonic() - began,
    }
    BASE._write_json_new(scope / "complete.json", result)
    return result


"""Persistent-worker official runner for the stateless output-only ERC."""

import atexit
import copy
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import struct
from types import SimpleNamespace
import time


def _load_base():
    path = Path(__file__).with_name("_base_fast_official_test.py")
    spec = importlib.util.spec_from_file_location("persistent_runner_base_40", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
_WORKER = None


class PersistentEpochWorker(BASE.PersistentPolicyWorker):
    def __init__(self, config):
        import cv2
        import torch
        from lib.test.evaluation.running import _save_tracker_output
        from change.erc_natural_reuse_fresh_20261021_01.bridge import load_legacy, collector_class
        from change.erc_rqc30_epoch_supervised_abstention_20261104_41.checkpoint import load_numeric
        from change.erc_rqc30_epoch_supervised_abstention_20261104_41.provider import causal_candidate_provider
        from change.erc_rqc30_epoch_supervised_abstention_20261104_41.runtime import OutputOnlyEpochRuntime

        self.config = dict(config)
        torch.cuda.set_device(int(self.config.get("gpu_id", 0)))
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(1)
        torch.backends.cudnn.benchmark = True
        params = copy.deepcopy(load_legacy().infrastructure.parameters())
        if params.cfg.TEST.MODE != self.config["mode"]:
            raise PermissionError("RQC30 test mode identity mismatch")
        params.cfg = copy.deepcopy(params.cfg)
        params.cfg.TEST.EPOCH = int(self.config["source_epoch"])
        params.save_all_boxes = False
        self.source = collector_class()(params, self.config["dataset_name"])
        self.policy, self.policy_manifest = load_numeric(
            self.config["checkpoint_dir"], device="cuda"
        )
        self.policy.eval().requires_grad_(False)
        self.runtime = OutputOnlyEpochRuntime(
            self.source, self.policy, causal_candidate_provider
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

    def run(self, sequence):
        """Write ERC and same-forward native RQC boxes with one source pass."""
        scope = Path(self.config["scope_root"]) / (sequence.name + ".json")
        if scope.exists():
            raise FileExistsError("no partial sequence overwrite: " + sequence.name)
        init_info = sequence.init_info()
        image = self._read_rgb(sequence.frames[0])
        started = time.perf_counter()
        with self.torch.inference_mode():
            result = self.runtime.initialize(image, {
                "init_bbox": init_info["init_bbox"], "language": sequence.language,
            })
        first = init_info["init_bbox"] if not result else result.get(
            "target_bbox", init_info["init_bbox"]
        )
        if not BASE._finite_box(first):
            raise ValueError("invalid initialization box")
        current_boxes, native_boxes = [list(first)], [list(first)]
        timings = [time.perf_counter() - started]
        actions = [0] * 25
        current_trace, native_trace = hashlib.sha256(), hashlib.sha256()
        for trace in (current_trace, native_trace):
            trace.update(struct.pack("<4d", *[float(value) for value in first]))
        for frame_path in sequence.frames[1:]:
            image = self._read_rgb(frame_path)
            started = time.perf_counter()
            with self.torch.inference_mode():
                result = self.runtime.track(image, explore=False)
            timings.append(time.perf_counter() - started)
            current = result["target_bbox"]
            native = self.runtime.last["rqc_box"]
            if not BASE._finite_box(current) or not BASE._finite_box(native):
                raise ValueError("invalid current or native same-run box")
            index = int(self.runtime.last["index"])
            if not 0 <= index < 25:
                raise ValueError("action index outside fixed lattice")
            if index == 0 and [float(v) for v in current] != [float(v) for v in native]:
                raise RuntimeError("KEEP must reproduce same-run native RQC exactly")
            actions[index] += 1
            current_boxes.append(list(current)); native_boxes.append(list(native))
            current_trace.update(struct.pack("<4d", *[float(value) for value in current]))
            native_trace.update(struct.pack("<4d", *[float(value) for value in native]))
        if len(current_boxes) != len(sequence.frames) or len(native_boxes) != len(sequence.frames):
            raise RuntimeError("incomplete sequence")
        if sum(actions) != len(sequence.frames) - 1:
            raise RuntimeError("action ledger mismatch")
        self.save_output(sequence, self.writer, {"target_bbox": current_boxes, "time": timings})
        self.save_output(sequence, self.native_writer, {"target_bbox": native_boxes, "time": timings})
        receipt = {
            "sequence": sequence.name, "frames": len(current_boxes),
            "source_forwards": len(current_boxes), "policy_forwards": len(current_boxes) - 1,
            "source_model_loads_in_this_sequence": 0, "baseline_gpu_forwards": 0,
            "actions": actions, "current_trace_sha256": current_trace.hexdigest(),
            "native_same_run_trace_sha256": native_trace.hexdigest(),
            "same_forward_native_shadow": True, "seconds": sum(timings),
            "official_integer_writer": True, "per_frame_audit_writes": 0,
        }
        BASE._write_json_new(scope, receipt)
        self.sequences += 1; self.frames += len(current_boxes)
        self.searches += len(current_boxes) - 1
        return receipt


def _initialize_worker(config):
    global _WORKER
    if _WORKER is not None:
        raise RuntimeError("worker initialized twice")
    _WORKER = PersistentEpochWorker(config)


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
        raise ValueError("exact fast-test config schema required")
    if any(Path(config[key]).exists() for key in ("results_dir", "native_results_dir", "scope_root")):
        raise FileExistsError("fresh current, native, and audit roots required")
    scope = Path(config["scope_root"])
    scope.mkdir(parents=True, exist_ok=False)
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
                    "heavy_model_load_upper_bound": int(processes),
                    "updated_unix": time.time(),
                })
    if len(receipts) != len(dataset) or sum(row["frames"] for row in receipts) != total_frames:
        raise RuntimeError("incomplete persistent official test")
    result = {
        "status": "PREDICTIONS_COMPLETE_PENDING_OFFICIAL_METRICS",
        "sequences": len(receipts), "frames": total_frames,
        "searches": total_frames - len(receipts),
        "worker_processes": int(processes),
        "maximum_heavy_source_loads": int(processes),
        "eliminated_heavy_source_loads": len(dataset) - int(processes),
        "baseline_gpu_forwards": 0, "per_frame_audit_writes": 0,
        "same_forward_native_shadow": True, "seconds": time.monotonic() - began,
    }
    BASE._write_json_new(scope / "complete.json", result)
    return result

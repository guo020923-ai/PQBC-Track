"""Persistent-worker official test runner for the fixed paired-rollout policy.

The stock UVLTrack runner sends one sequence per multiprocessing task.  Each
task calls ``Tracker.create_tracker`` and therefore reloads the frozen source
checkpoint.  TNL2K has 700 sequences, so that path performs hundreds of
redundant model constructions.

This isolated runner keeps exactly one frozen RQC30 source and one small policy
inside each worker process and reinitializes their sequence state.  It retains
the original integer result writer.  The already completed, identity-bound
RQC30 prediction set is used as the comparison reference; it is never replayed
on the GPU.
"""

import atexit
import copy
import hashlib
import json
import math
import multiprocessing
import os
import struct
import time
from pathlib import Path
from types import SimpleNamespace


_WORKER = None


def _write_json_new(path, value):
    path = Path(path)
    with path.open("x", encoding="utf8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_replace(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with temporary.open("w", encoding="utf8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _parse_prediction_lines(lines, expected_rows, label="prediction"):
    """Strict integer-writer parser without NumPy or pickle."""
    rows = []
    for line in lines:
        fields = line.rstrip("\r\n").split("\t")
        if len(fields) != 4:
            raise ValueError("prediction row must contain four integers: " + str(label))
        row = [int(field) for field in fields]
        if row[2] <= 0 or row[3] <= 0:
            raise ValueError("prediction width/height must be positive: " + str(label))
        rows.append(row)
    if len(rows) != expected_rows:
        raise ValueError("prediction row count mismatch: " + str(label))
    return rows


def _read_prediction(path, expected_rows):
    with Path(path).open("r", encoding="utf8") as handle:
        return _parse_prediction_lines(handle, expected_rows, path)


def validate_existing_reference(dataset, reference_dir):
    """Bind the complete reproduced RQC30 outputs before any new GPU work."""
    reference_dir = Path(reference_dir)
    expected = {sequence.name + ".txt" for sequence in dataset}
    actual = {
        path.name for path in reference_dir.glob("*.txt")
        if not path.name.endswith("_time.txt")
    }
    if actual != expected:
        raise ValueError("existing RQC30 reference is not the exact dataset inventory")
    digest = hashlib.sha256()
    frames = 0
    for sequence in dataset:
        path = reference_dir / (sequence.name + ".txt")
        rows = _read_prediction(path, len(sequence.frames))
        frames += len(rows)
        digest.update(sequence.name.encode("utf8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return {
        "sequences": len(dataset),
        "frames": frames,
        "prediction_set_sha256": digest.hexdigest(),
        "reference_dir": str(reference_dir),
        "gpu_replay": False,
    }


def _finite_box(box):
    return (
        isinstance(box, (list, tuple)) and len(box) == 4
        and all(math.isfinite(float(value)) for value in box)
        and float(box[2]) > 0 and float(box[3]) > 0
    )


class PersistentPolicyWorker:
    """One heavyweight source load for all sequences handled by a process."""

    def __init__(self, config):
        import cv2
        import torch
        from lib.test.evaluation.running import _save_tracker_output
        from change.erc_natural_reuse_fresh_20261021_01.bridge import load_legacy
        from .checkpoint import load_numeric
        from .provider import causal_candidate_provider, collector_class
        from .runtime import ClosedLoopResidualRuntime

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

        # The only heavyweight construction in this process.
        self.source = collector_class()(params, self.config["dataset_name"])
        self.policy, self.policy_manifest = load_numeric(
            self.config["checkpoint_dir"], device="cuda"
        )
        self.policy.eval()
        self.policy.requires_grad_(False)
        self.runtime = ClosedLoopResidualRuntime(
            self.source, self.policy, causal_candidate_provider
        )

        writer_params = copy.deepcopy(params)
        writer_params.cfg = copy.deepcopy(params.cfg)
        writer_params.cfg.TEST.EPOCH = int(self.config["writer_epoch"])
        self.writer = SimpleNamespace(
            results_dir=self.config["results_dir"], params=writer_params
        )
        self.save_output = _save_tracker_output
        self.cv2 = cv2
        self.torch = torch
        self.sequences = 0
        self.frames = 0
        self.searches = 0
        atexit.register(self.close)

    def _read_rgb(self, path):
        image = self.cv2.imread(str(path), self.cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cannot read test frame: " + str(path))
        return self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)

    def run(self, sequence):
        """Original frame order and writer, with no per-frame disk audit."""
        scope = Path(self.config["scope_root"]) / (sequence.name + ".json")
        if scope.exists():
            raise FileExistsError("no partial sequence resume or overwrite: " + sequence.name)

        init_info = sequence.init_info()
        image = self._read_rgb(sequence.frames[0])
        started = time.perf_counter()
        with self.torch.inference_mode():
            result = self.runtime.initialize(image, {
                "init_bbox": init_info["init_bbox"],
                "language": sequence.language,
            })
        first_box = init_info["init_bbox"] if not result else result.get(
            "target_bbox", init_info["init_bbox"]
        )
        if not _finite_box(first_box):
            raise ValueError("invalid initialization box")
        boxes = [list(first_box)]
        timings = [time.perf_counter() - started]
        actions = [0] * 25
        trace = hashlib.sha256()
        trace.update(struct.pack("<4d", *[float(value) for value in first_box]))

        for frame_path in sequence.frames[1:]:
            image = self._read_rgb(frame_path)
            started = time.perf_counter()
            with self.torch.inference_mode():
                result = self.runtime.track(image, explore=False)
            elapsed = time.perf_counter() - started
            box = result["target_bbox"]
            if not _finite_box(box):
                raise ValueError("non-finite or non-positive policy box")
            index = int(self.runtime.last["index"])
            if not 0 <= index < len(actions):
                raise ValueError("policy action index outside fixed 25-action lattice")
            actions[index] += 1
            boxes.append(list(box))
            timings.append(elapsed)
            trace.update(struct.pack("<4d", *[float(value) for value in box]))

        if len(boxes) != len(sequence.frames) or sum(actions) != len(boxes) - 1:
            raise RuntimeError("incomplete sequence execution")
        output = {"target_bbox": boxes, "time": timings}
        self.save_output(sequence, self.writer, output)

        receipt = {
            "sequence": sequence.name,
            "frames": len(boxes),
            "source_forwards": len(boxes),
            "policy_forwards": len(boxes) - 1,
            "source_model_loads_in_this_sequence": 0,
            "baseline_gpu_forwards": 0,
            "actions": actions,
            "prediction_trace_sha256": trace.hexdigest(),
            "seconds": sum(timings),
            "official_integer_writer": True,
            "per_frame_audit_writes": 0,
        }
        _write_json_new(scope, receipt)
        self.sequences += 1
        self.frames += len(boxes)
        self.searches += len(boxes) - 1
        return receipt

    def close(self):
        source = getattr(self, "source", None)
        if source is not None:
            self.source = None
            try:
                source.close()
            except Exception:
                # Process teardown must not hide a completed sequence error.
                pass


def _initialize_worker(config):
    global _WORKER
    if _WORKER is not None:
        raise RuntimeError("worker initialized twice")
    _WORKER = PersistentPolicyWorker(config)


def _run_one(sequence):
    if _WORKER is None:
        raise RuntimeError("persistent worker is not initialized")
    return _WORKER.run(sequence)


def run_persistent_dataset(dataset, config, processes=8):
    """Run a fresh complete dataset with one model load per worker."""
    config = dict(config)
    required = {
        "checkpoint_dir", "results_dir", "reference_results_dir", "scope_root",
        "dataset_name", "mode", "source_epoch", "writer_epoch", "gpu_id",
    }
    if set(config) != required:
        raise ValueError("exact fast-test config schema required")
    if int(processes) < 1:
        raise ValueError("at least one persistent worker required")
    if Path(config["results_dir"]).exists() or Path(config["scope_root"]).exists():
        raise FileExistsError("fresh result and audit roots are required")

    reference = validate_existing_reference(dataset, config["reference_results_dir"])
    Path(config["scope_root"]).mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    context = multiprocessing.get_context("spawn")
    receipts = []
    with context.Pool(
        processes=int(processes), initializer=_initialize_worker,
        initargs=(config,), maxtasksperchild=None,
    ) as pool:
        for receipt in pool.imap_unordered(_run_one, dataset, chunksize=1):
            receipts.append(receipt)
            if len(receipts) == 1 or len(receipts) % 10 == 0 or len(receipts) == len(dataset):
                _write_json_replace(Path(config["scope_root"]) / "progress.json", {
                    "completed_sequences": len(receipts),
                    "total_sequences": len(dataset),
                    "completed_frames": sum(row["frames"] for row in receipts),
                    "elapsed_seconds": time.monotonic() - began,
                    "heavy_model_load_upper_bound": int(processes),
                })

    expected_frames = sum(len(sequence.frames) for sequence in dataset)
    if len(receipts) != len(dataset) or sum(row["frames"] for row in receipts) != expected_frames:
        raise RuntimeError("persistent official test did not complete the full dataset")
    result = {
        "status": "PREDICTIONS_COMPLETE_PENDING_OFFICIAL_METRICS",
        "sequences": len(receipts),
        "frames": expected_frames,
        "searches": expected_frames - len(receipts),
        "worker_processes": int(processes),
        "maximum_heavy_source_loads": int(processes),
        "legacy_heavy_source_loads": len(dataset),
        "eliminated_heavy_source_loads": len(dataset) - int(processes),
        "baseline_gpu_forwards": 0,
        "per_frame_audit_writes": 0,
        "reference": reference,
        "seconds": time.monotonic() - began,
    }
    _write_json_new(Path(config["scope_root"]) / "complete.json", result)
    return result

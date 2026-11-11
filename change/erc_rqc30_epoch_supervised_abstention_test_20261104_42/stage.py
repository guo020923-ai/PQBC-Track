"""One fixed TNL2K FULL700 prediction run and one official evaluation."""

import hashlib
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
import sys
import time

from .fast_runner import run_persistent_dataset


DATASET, MODE = "tnl2k", "NLBBOX"
SEQUENCES, FRAMES, SEARCHES = 700, 516038, 515338
RESULT_NAME = "baseline_base_head_rqc30_epoch_supervised_abstention_erc30_20261104_42"
RUNID = 110442
NATIVE_NAME = "baseline_base_head_rqc30_epoch_supervised_abstention_native_same_run_20261104_42"
NATIVE_RUNID = 110443
REFERENCE_METRICS = {
    "AUC": 63.02118818112782,
    "OP50": 73.85304531233652,
    "OP75": 59.13824240182127,
    "Precision": 66.49108568055289,
    "Norm Precision": 80.40470445666995,
}


def _write_new(path, value):
    with Path(path).open("x", encoding="utf8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())


def _replace(path, value):
    path = Path(path); temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _evaluate_once(tracker, dataset, report_root):
    # The legacy official adapter imports its sibling modules by their
    # historical top-level names.  Package import alone adds the project root,
    # but not this sibling directory, so make that dependency explicit before
    # importing the unchanged metric adapter.
    legacy_dir = Path(__file__).resolve().parents[1] / "rqc_dev_v3"
    required = legacy_dir / "directional_center_joint_contract_20261001.py"
    if not required.is_file():
        raise FileNotFoundError(required)
    if str(legacy_dir) not in sys.path:
        sys.path.insert(0, str(legacy_dir))
    from lib.test.analysis import plot_results
    from change.rqc_dev_v3.evaluate_directional_center_joint_20261001 import run_official_metrics
    report_root = Path(report_root)
    if report_root.exists():
        raise FileExistsError(report_root)
    formatter = plot_results.generate_formatted_report
    loader = plot_results.check_and_load_precomputed_results
    captured = {}

    def format_report(names, scores, *args, **kwargs):
        if len(names) != 1:
            raise RuntimeError("one official stream required")
        captured["metrics"] = {
            name: float(value.detach().cpu().reshape(-1)[0])
            for name, value in scores.items()
        }
        return formatter(names, scores, *args, **kwargs)

    def load(*args, **kwargs):
        if kwargs.get("force_evaluation") is not True:
            raise PermissionError("fresh forced official evaluation required")
        result = loader(*args, **kwargs)
        if len(result["valid_sequence"]) != SEQUENCES or not all(result["valid_sequence"]):
            raise RuntimeError("all 700 sequences must be valid")
        return result

    plot_results.generate_formatted_report = format_report
    plot_results.check_and_load_precomputed_results = load
    began = time.monotonic()
    try:
        run_official_metrics([tracker], dataset, "tnl2k_NLBBOX_030", report_root)
    finally:
        plot_results.generate_formatted_report = formatter
        plot_results.check_and_load_precomputed_results = loader
    metrics = captured.get("metrics")
    if set(metrics or {}) != set(REFERENCE_METRICS):
        raise ValueError("exact five official metrics required")
    if any(not math.isfinite(value) or not 0 <= value <= 100 for value in metrics.values()):
        raise ValueError("invalid official metric")
    return {"metrics": metrics, "seconds": time.monotonic() - began}


def _bca(values):
    import numpy as np
    data = np.asarray(values, dtype=np.float64)
    theta = float(data.mean())
    rng = np.random.default_rng(20261104)
    boot = np.empty(10000, dtype=np.float64)
    for begin in range(0, len(boot), 250):
        count = min(250, len(boot) - begin)
        indices = rng.integers(0, len(data), size=(count, len(data)))
        boot[begin:begin + count] = data[indices].mean(1)
    normal = NormalDist()
    proportion = (float((boot < theta).sum()) + 0.5) / (len(boot) + 1.0)
    z0 = normal.inv_cdf(min(max(proportion, 1e-12), 1 - 1e-12))
    jack = (data.sum() - data) / (len(data) - 1)
    centered = jack.mean() - jack
    denominator = 6 * float(np.square(centered).sum()) ** 1.5
    acceleration = float(np.power(centered, 3).sum()) / denominator if denominator else 0.0
    def adjusted(probability):
        z = normal.inv_cdf(probability)
        value = normal.cdf(z0 + (z0 + z) / (1 - acceleration * (z0 + z)))
        return min(max(value, 0.0), 1.0)
    return {
        "method": "paired_sequence_BCa", "confidence": 0.95,
        "replicates": 10000, "seed": 20261104, "estimate": theta,
        "lower": float(np.quantile(boot, adjusted(0.025))),
        "upper": float(np.quantile(boot, adjusted(0.975))),
    }


def _paired(dataset, current_dir, native_dir, scope_root):
    import numpy as np
    import torch
    from lib.test.analysis.extract_results import calc_seq_err_robust
    from lib.test.utils.load_text import load_text
    thresholds = torch.arange(0.0, 1.01, 0.05, dtype=torch.float64)
    rows, actions = [], [0] * 25
    for sequence in dataset:
        truth = torch.tensor(sequence.ground_truth_rect, dtype=torch.float64)
        visible = torch.tensor(sequence.target_visible, dtype=torch.uint8) if sequence.target_visible is not None else None
        aucs = []
        for folder in (Path(current_dir), Path(native_dir)):
            boxes = torch.tensor(load_text(str(folder / (sequence.name + ".txt")), delimiter=("\t", ","), dtype=np.float64)).reshape(-1, 4)
            if boxes.shape != truth.shape or not torch.isfinite(boxes).all() or not boxes[:, 2:].gt(0).all():
                raise ValueError("invalid integer output: " + sequence.name)
            overlap, _, _, _ = calc_seq_err_robust(boxes, truth, sequence.dataset, visible)
            aucs.append(float((overlap[:, None] > thresholds).double().mean() * 100))
        receipt = json.loads((Path(scope_root) / (sequence.name + ".json")).read_text())
        if receipt["frames"] != len(sequence.frames) or len(receipt["actions"]) != 25:
            raise ValueError("action receipt mismatch")
        actions = [a + int(b) for a, b in zip(actions, receipt["actions"])]
        rows.append({"sequence": sequence.name, "frames": len(sequence.frames),
                     "current_auc": aucs[0], "native_same_run_auc": aucs[1],
                     "delta_auc_pp": aucs[0] - aucs[1],
                     "nonkeep": sum(receipt["actions"][1:])})
    if len(rows) != SEQUENCES or sum(actions) != SEARCHES:
        raise RuntimeError("complete paired closure required")
    interval = _bca([row["delta_auc_pp"] for row in rows])
    return {"rows": rows, "action_totals": actions, "paired_interval": interval}


def run_stage(training_audit, numeric_dir, stage_root, processes=8):
    if training_audit.get("status") != "COMPLETE_VERIFIED":
        raise PermissionError("verified training audit required")
    if training_audit.get("final_checkpoint_sha256") != "d7b47795746c7de05c9dce0ac846e2f27e7f06a1b624cae73feaaf92e4944d14":
        raise ValueError("fixed Epoch-30 identity mismatch")
    numeric_dir, stage_root = Path(numeric_dir), Path(stage_root)
    manifest = json.loads((numeric_dir / "manifest.json").read_text())
    if _digest(numeric_dir / "erc_epoch30_numeric.npz") != manifest["numeric_npz_sha256"]:
        raise ValueError("numeric weight changed")
    from lib.test.evaluation import get_dataset
    from lib.test.evaluation.tracker import Tracker
    dataset = get_dataset(DATASET)
    if len(dataset) != SEQUENCES or sum(len(sequence.frames) for sequence in dataset) != FRAMES:
        raise ValueError("complete TNL2K inventory required")
    current = Tracker("uvltrack", RESULT_NAME, DATASET, run_id=RUNID, eval=True)
    native = Tracker("uvltrack", NATIVE_NAME, DATASET, run_id=NATIVE_RUNID, eval=True)
    current_dir = Path(current.results_dir) / "tnl2k_NLBBOX_030"
    native_dir = Path(native.results_dir) / "tnl2k_NLBBOX_030"
    stage_root.mkdir(parents=False)
    _write_new(stage_root / "manifest.json", {
        "dataset": DATASET, "mode": MODE, "sequences": SEQUENCES,
        "frames": FRAMES, "searches": SEARCHES, "result_name": RESULT_NAME,
        "runid": RUNID, "native_name": NATIVE_NAME,
        "native_runid": NATIVE_RUNID, "historical_reference_metrics": REFERENCE_METRICS,
        "fixed_checkpoint_sha256": training_audit["final_checkpoint_sha256"],
        "numeric_npz_sha256": manifest["numeric_npz_sha256"],
        "persistent_workers": int(processes), "baseline_gpu_replay": False,
        "official_evaluations": 2, "same_forward_native_shadow": True,
        "automatic_retry": False,
    })
    began = time.monotonic()
    prediction = run_persistent_dataset(dataset, {
        "checkpoint_dir": str(numeric_dir), "results_dir": str(Path(current.results_dir)),
        "native_results_dir": str(Path(native.results_dir)),
        "scope_root": str(stage_root / "sequence_scopes"),
        "dataset_name": DATASET, "mode": MODE, "source_epoch": 30,
        "writer_epoch": 30, "gpu_id": 0,
    }, processes=processes)
    _replace(stage_root / "progress.json", {"phase": "OFFICIAL_METRICS", "updated_unix": time.time()})
    official = _evaluate_once(current, dataset, stage_root / "official_report_current")
    native_official = _evaluate_once(native, dataset, stage_root / "official_report_native_same_run")
    _replace(stage_root / "progress.json", {"phase": "PAIRED_AUDIT", "updated_unix": time.time()})
    paired = _paired(dataset, current_dir, native_dir, stage_root / "sequence_scopes")
    result = {
        "status": "TEST_COMPLETE_PENDING_READONLY_AUDIT",
        "prediction": prediction, "official": official,
        "native_same_run_official": native_official,
        "paired_interval": paired["paired_interval"],
        "action_totals": paired["action_totals"],
        "delta_auc_pp_vs_native_same_run": official["metrics"]["AUC"] - native_official["metrics"]["AUC"],
        "delta_auc_pp_vs_historical_rqc30": official["metrics"]["AUC"] - REFERENCE_METRICS["AUC"],
        "seconds": time.monotonic() - began, "automatic_retrain": False,
    }
    _write_new(stage_root / "paired_sequences.json", paired["rows"])
    _write_new(stage_root / "official_complete.json", result)
    _replace(stage_root / "progress.json", {"phase": "TEST_COMPLETE", "updated_unix": time.time()})
    return result

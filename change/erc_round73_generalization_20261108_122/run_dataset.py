"""Evaluate the sealed Round73 runtime on one frozen generalization dataset."""

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import NormalDist


PROJECT = Path("/root/autodl-tmp/UVLTrack-master/change/rqc_jq_20260929_isolated")
ROUND73_PACKAGE = PROJECT / "change" / "erc_rqc30_temporal_cooldown_pairwise_100ep_20261107_73"
ROUND73_STAGE = Path("/root/autodl-tmp/UVLTrack-master/artifacts/pqbc_final/stages/erc_rqc30_temporal_cooldown_pairwise_pipeline_20261107_73")
NUMERIC_DIR = ROUND73_STAGE / "numeric"
CHECKPOINT = ROUND73_STAGE / "train" / "checkpoints" / "ERC_epoch_0100.pth.tar"
CHECKPOINT_SHA256 = "65ce9732add17da4e1790f0c2b53bd36e0cea0c0a1e56bc18c1e2125135bebb3"
NUMERIC_SHA256 = "fe65022c45b8f6db318c6422399967a3b37f3913df704ae3f3ae854d493db935"
IMPLEMENTATION_MANIFEST_SHA256 = "cc3f366ab4357dda9381afc670906eeb24161a5ea4d131d3da4c6f5d3c0366c4"
THRESHOLD = 0.44855380058288574

DATASETS = {
    "lasot": {
        "sequences": 280,
        "frames": 685360,
        "folder": "lasot_NLBBOX_030",
        "result_name": "baseline_base_head_rqc30_round73_cooldown8_erc100_lasot_20261108_122",
        "runid": 11081221,
        "native_name": "baseline_base_head_rqc30_round73_cooldown8_native_lasot_20261108_122",
        "native_runid": 11081222,
        "reference": {
            "AUC": 68.50761342048645,
            "OP50": 79.45545966114317,
            "OP75": 67.79640516809054,
            "Precision": 74.38041511263167,
            "Norm Precision": 77.95683300835746,
        },
    },
    "otb99": {
        "sequences": 48,
        "frames": 29077,
        "folder": "otb99_NLBBOX_030",
        "result_name": "baseline_base_head_rqc30_round73_cooldown8_erc100_otb99_20261108_122",
        "runid": 11081223,
        "native_name": "baseline_base_head_rqc30_round73_cooldown8_native_otb99_20261108_122",
        "native_runid": 11081224,
        "reference": {
            "AUC": 69.4640643397967,
            "OP50": 86.80936818321545,
            "OP75": 56.78332799921433,
            "Precision": 89.68370133638382,
            "Norm Precision": 83.6798628171285,
        },
    },
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def replace(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def verify_round73_identity():
    manifest_path = ROUND73_PACKAGE / "implementation_manifest_73.json"
    if digest(manifest_path) != IMPLEMENTATION_MANIFEST_SHA256:
        raise ValueError("Round73 implementation manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = PROJECT / "change" / row["package"] / row["name"]
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("Round73 sealed source changed: " + str(path))
    numeric_manifest = json.loads((NUMERIC_DIR / "manifest.json").read_text(encoding="utf-8"))
    calibration = json.loads((NUMERIC_DIR / "pairwise_evaluation_fallback.json").read_text(encoding="utf-8"))
    if digest(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("Round73 Epoch100 checkpoint changed")
    if digest(NUMERIC_DIR / "erc_epoch100_numeric.npz") != NUMERIC_SHA256:
        raise ValueError("Round73 numeric weights changed")
    if numeric_manifest.get("source_checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("numeric/checkpoint binding")
    if calibration.get("threshold") != THRESHOLD or calibration.get("test_data_read") is not False:
        raise ValueError("Round73 threshold identity")
    if numeric_manifest["protocol"].get("temporal_cooldown_frames") != 8:
        raise ValueError("Round73 cooldown identity")
    return manifest, numeric_manifest, calibration


def evaluate_once(tracker, dataset, config, report_root):
    legacy_dir = PROJECT / "change" / "rqc_dev_v3"
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
            name: float(value.detach().cpu().reshape(-1)[0]) for name, value in scores.items()
        }
        return formatter(names, scores, *args, **kwargs)

    def load(*args, **kwargs):
        if kwargs.get("force_evaluation") is not True:
            raise PermissionError("fresh forced official evaluation required")
        result = loader(*args, **kwargs)
        if len(result["valid_sequence"]) != config["sequences"] or not all(result["valid_sequence"]):
            raise RuntimeError("complete dataset required")
        return result

    plot_results.generate_formatted_report = format_report
    plot_results.check_and_load_precomputed_results = load
    began = time.monotonic()
    try:
        run_official_metrics([tracker], dataset, config["folder"], report_root)
    finally:
        plot_results.generate_formatted_report = formatter
        plot_results.check_and_load_precomputed_results = loader
    metrics = captured.get("metrics")
    if set(metrics or {}) != set(config["reference"]):
        raise ValueError("exact five official metrics required")
    if any(not math.isfinite(value) or not 0 <= value <= 100 for value in metrics.values()):
        raise ValueError("invalid official metric")
    return {"metrics": metrics, "seconds": time.monotonic() - began}


def bca(values):
    import numpy as np
    data = np.asarray(values, dtype=np.float64)
    theta = float(data.mean())
    rng = np.random.default_rng(20261108)
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
    denominator = 6 * float((centered ** 2).sum()) ** 1.5
    acceleration = float((centered ** 3).sum()) / denominator if denominator else 0.0

    def adjusted(probability):
        z = normal.inv_cdf(probability)
        value = normal.cdf(z0 + (z0 + z) / (1 - acceleration * (z0 + z)))
        return min(max(value, 0.0), 1.0)

    return {
        "method": "paired_sequence_BCa", "confidence": 0.95,
        "replicates": 10000, "seed": 20261108, "estimate": theta,
        "lower": float(np.quantile(boot, adjusted(0.025))),
        "upper": float(np.quantile(boot, adjusted(0.975))),
    }


def paired(dataset, current_dir, native_dir, scope_root, searches):
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
                raise ValueError("invalid output: " + sequence.name)
            overlap, _, _, _ = calc_seq_err_robust(boxes, truth, sequence.dataset, visible)
            aucs.append(float((overlap[:, None] > thresholds).double().mean() * 100))
        receipt = json.loads((Path(scope_root) / (sequence.name + ".json")).read_text(encoding="utf-8"))
        actions = [a + int(b) for a, b in zip(actions, receipt["actions"])]
        rows.append({"sequence": sequence.name, "frames": len(sequence.frames),
                     "current_auc": aucs[0], "native_same_run_auc": aucs[1],
                     "delta_auc_pp": aucs[0] - aucs[1],
                     "nonkeep": sum(receipt["actions"][1:])})
    if sum(actions) != searches:
        raise RuntimeError("complete action closure required")
    return rows, actions, bca([row["delta_auc_pp"] for row in rows])


def main(dataset_name, stage_root):
    stage_root = Path(stage_root)
    if dataset_name not in DATASETS:
        raise ValueError(dataset_name)
    config = DATASETS[dataset_name]
    if stage_root.exists():
        raise FileExistsError(stage_root)
    os.chdir(PROJECT)
    sys.path.insert(0, str(PROJECT))
    manifest, numeric_manifest, calibration = verify_round73_identity()
    from lib.test.evaluation import get_dataset
    from lib.test.evaluation.tracker import Tracker
    from change.erc_rqc30_temporal_cooldown_pairwise_100ep_20261107_73.fast_runner import run_persistent_dataset

    dataset = get_dataset(dataset_name)
    frames = sum(len(sequence.frames) for sequence in dataset)
    if len(dataset) != config["sequences"] or frames != config["frames"]:
        raise ValueError("frozen dataset inventory changed")
    searches = frames - len(dataset)
    current = Tracker("uvltrack", config["result_name"], dataset_name, run_id=config["runid"], eval=True)
    native = Tracker("uvltrack", config["native_name"], dataset_name, run_id=config["native_runid"], eval=True)
    current_dir = Path(current.results_dir) / config["folder"]
    native_dir = Path(native.results_dir) / config["folder"]
    if any(path.exists() for path in (stage_root, Path(current.results_dir), Path(native.results_dir))):
        raise FileExistsError("fresh roots required")
    stage_root.mkdir(parents=False)
    write_new(stage_root / "launch.json", {
        "status": "ROUND73_FROZEN_GENERALIZATION_STARTED",
        "dataset": dataset_name, "sequences": len(dataset), "frames": frames,
        "searches": searches, "round73_checkpoint_sha256": CHECKPOINT_SHA256,
        "round73_numeric_sha256": NUMERIC_SHA256,
        "round73_manifest_sha256": IMPLEMENTATION_MANIFEST_SHA256,
        "threshold": THRESHOLD, "temporal_cooldown_frames": 8,
        "sealed_file_count": len(manifest["files"]), "test_data_calibration": False,
        "dataset_specific_tuning": False, "started_unix": time.time(),
    })
    began = time.monotonic()
    prediction = run_persistent_dataset(dataset, {
        "checkpoint_dir": str(NUMERIC_DIR), "results_dir": str(Path(current.results_dir)),
        "native_results_dir": str(Path(native.results_dir)),
        "scope_root": str(stage_root / "sequence_scopes"),
        "dataset_name": dataset_name, "mode": "NLBBOX", "source_epoch": 30,
        "writer_epoch": 30, "gpu_id": 0,
    }, processes=8)
    replace(stage_root / "progress.json", {"phase": "OFFICIAL_METRICS", "updated_unix": time.time()})
    official = evaluate_once(current, dataset, config, stage_root / "official_report_current")
    native_official = evaluate_once(native, dataset, config, stage_root / "official_report_native")
    replace(stage_root / "progress.json", {"phase": "PAIRED_AUDIT", "updated_unix": time.time()})
    rows, actions, interval = paired(dataset, current_dir, native_dir, stage_root / "sequence_scopes", searches)
    result = {
        "status": "ROUND73_FROZEN_GENERALIZATION_COMPLETE",
        "dataset": dataset_name, "prediction": prediction,
        "official": official, "native_same_run_official": native_official,
        "paired_interval": interval, "action_totals": actions,
        "reference_fixed_rqc30": config["reference"],
        "delta_vs_fixed_rqc30_pp": {
            key: official["metrics"][key] - config["reference"][key] for key in config["reference"]
        },
        "round73_checkpoint_sha256": CHECKPOINT_SHA256,
        "round73_numeric_sha256": NUMERIC_SHA256,
        "threshold": calibration["threshold"], "temporal_cooldown_frames": 8,
        "dataset_specific_tuning": False, "seconds": time.monotonic() - began,
    }
    write_new(stage_root / "paired_sequences.json", rows)
    write_new(stage_root / "official_complete.json", result)
    replace(stage_root / "progress.json", {"phase": "COMPLETE", "updated_unix": time.time()})
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: run_dataset DATASET STAGE_ROOT")
    main(sys.argv[1], sys.argv[2])


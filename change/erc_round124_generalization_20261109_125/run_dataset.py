"""Evaluate the sealed Round124 runtime on one frozen generalization dataset."""

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

from change.erc_round73_generalization_20261108_122.run_dataset import bca, evaluate_once


PROJECT = Path(__file__).resolve().parents[2]
ROUND124_PACKAGE_NAME = "erc_rqc30_temporal_cooldown_pairwise_seed20261109_geomfallback_20261109_124"
ROUND124_PACKAGE = PROJECT / "change" / ROUND124_PACKAGE_NAME
ROUND123_STAGE = PROJECT / "artifacts/pqbc_final/stages/erc_round73_seed20261109_replication_20261109_123"
NUMERIC_DIR = ROUND123_STAGE / "numeric"
CHECKPOINT = ROUND123_STAGE / "train" / "checkpoints" / "ERC_epoch_0100.pth.tar"
CHECKPOINT_SHA256 = "354e3723cfe36a0ef861e16c93fbacd680ea548f5d508aeb9b9b72dd8c4baf53"
NUMERIC_SHA256 = "bc19147bc61babaa64090eef360eb7463ee56977456fbd53dcbc44ee62a4da92"
IMPLEMENTATION_MANIFEST_SHA256 = "d20633c883343c5f46afc442468b1c07c68f6ffe6008732aa6361f34c7398e28"
THRESHOLD = 0.36725926399230957

DATASETS = {
    "lasot": {
        "sequences": 280, "frames": 685360, "folder": "lasot_NLBBOX_030",
        "result_name": "baseline_base_head_rqc30_round124_cooldown8_geomfallback_erc100_lasot_20261109_125",
        "runid": 11091251,
        "native_name": "baseline_base_head_rqc30_round124_cooldown8_geomfallback_native_lasot_20261109_125",
        "native_runid": 11091252,
        "reference": {
            "AUC": 68.50761342048645, "OP50": 79.45545966114317,
            "OP75": 67.79640516809054, "Precision": 74.38041511263167,
            "Norm Precision": 77.95683300835746,
        },
    },
    "otb99": {
        "sequences": 48, "frames": 29077, "folder": "otb99_NLBBOX_030",
        "result_name": "baseline_base_head_rqc30_round124_cooldown8_geomfallback_erc100_otb99_20261109_125",
        "runid": 11091253,
        "native_name": "baseline_base_head_rqc30_round124_cooldown8_geomfallback_native_otb99_20261109_125",
        "native_runid": 11091254,
        "reference": {
            "AUC": 69.4640643397967, "OP50": 86.80936818321545,
            "OP75": 56.78332799921433, "Precision": 89.68370133638382,
            "Norm Precision": 83.6798628171285,
        },
    },
}

EVALUATOR_FILES = [
    "lib/test/evaluation/running.py", "lib/test/evaluation/tracker.py",
    "lib/test/analysis/extract_results.py", "lib/test/analysis/plot_results.py",
    "lib/test/tracker/uvltrack.py", "lib/test/parameter/uvltrack.py",
    "change/rqc_dev_v3/evaluate_directional_center_joint_20261001.py",
]


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


def verify_round124_identity():
    manifest_path = ROUND124_PACKAGE / "implementation_manifest_124.json"
    if digest(manifest_path) != IMPLEMENTATION_MANIFEST_SHA256:
        raise ValueError("Round124 implementation manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = PROJECT / "change" / row["package"] / row["name"]
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("Round124 sealed source changed: " + str(path))
    numeric_manifest = json.loads((NUMERIC_DIR / "manifest.json").read_text(encoding="utf-8"))
    calibration = json.loads((NUMERIC_DIR / "pairwise_evaluation_fallback.json").read_text(encoding="utf-8"))
    if digest(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("Round124 Epoch100 checkpoint changed")
    if digest(NUMERIC_DIR / "erc_epoch100_numeric.npz") != NUMERIC_SHA256:
        raise ValueError("Round124 numeric weights changed")
    if numeric_manifest.get("source_checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("numeric/checkpoint binding mismatch")
    if calibration.get("threshold") != THRESHOLD or calibration.get("test_data_read") is not False:
        raise ValueError("Round124 TRAIN-only threshold identity")
    protocol = numeric_manifest.get("protocol", {})
    if protocol.get("temporal_cooldown_frames") != 8 or protocol.get("training_seed") != 20261109:
        raise ValueError("Round124 cooldown or seed identity")
    return manifest, numeric_manifest, calibration


def paired(dataset, current_dir, native_dir, scope_root, fallback_root, searches):
    import numpy as np
    import torch
    from lib.test.analysis.extract_results import calc_seq_err_robust
    from lib.test.utils.load_text import load_text

    thresholds = torch.arange(0.0, 1.01, 0.05, dtype=torch.float64)
    rows, actions = [], [0] * 25
    fallback_total = commit_total = suppressed_total = 0
    for sequence in dataset:
        truth = torch.tensor(sequence.ground_truth_rect, dtype=torch.float64)
        visible = torch.tensor(sequence.target_visible, dtype=torch.uint8) if sequence.target_visible is not None else None
        aucs = []
        for folder in (Path(current_dir), Path(native_dir)):
            boxes = torch.tensor(load_text(
                str(folder / (sequence.name + ".txt")), delimiter=("\t", ","), dtype=np.float64,
            )).reshape(-1, 4)
            if boxes.shape != truth.shape or not torch.isfinite(boxes).all() or not boxes[:, 2:].gt(0).all():
                raise ValueError("invalid output: " + sequence.name)
            overlap, _, _, _ = calc_seq_err_robust(boxes, truth, sequence.dataset, visible)
            aucs.append(float((overlap[:, None] > thresholds).double().mean() * 100))
        receipt = json.loads((Path(scope_root) / (sequence.name + ".json")).read_text(encoding="utf-8"))
        fallback = json.loads((Path(fallback_root) / (sequence.name + ".json")).read_text(encoding="utf-8"))
        actions = [a + int(b) for a, b in zip(actions, receipt["actions"])]
        fallback_total += int(fallback["invalid_feedback_fallbacks"])
        commit_total += int(fallback["feedback_commits"])
        suppressed_total += int(fallback["suppressed_nonkeep_proposals"])
        rows.append({
            "sequence": sequence.name, "frames": len(sequence.frames),
            "current_auc": aucs[0], "native_same_run_auc": aucs[1],
            "delta_auc_pp": aucs[0] - aucs[1],
            "nonkeep": sum(receipt["actions"][1:]),
            "invalid_feedback_fallbacks": int(fallback["invalid_feedback_fallbacks"]),
        })
    if sum(actions) != searches:
        raise RuntimeError("complete action closure required")
    return {
        "rows": rows, "actions": actions,
        "paired_interval": bca([row["delta_auc_pp"] for row in rows]),
        "invalid_feedback_fallbacks": fallback_total,
        "feedback_commits": commit_total,
        "suppressed_nonkeep_proposals": suppressed_total,
    }


def main(dataset_name, stage_root):
    stage_root = Path(stage_root)
    if dataset_name not in DATASETS:
        raise ValueError(dataset_name)
    config = DATASETS[dataset_name]
    if stage_root.exists():
        raise FileExistsError(stage_root)
    os.chdir(PROJECT)
    sys.path.insert(0, str(PROJECT))
    manifest, numeric_manifest, calibration = verify_round124_identity()
    from lib.test.evaluation import get_dataset
    from lib.test.evaluation.tracker import Tracker
    from change.erc_round124_generalization_20261109_125.fast_runner import run_persistent_dataset

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
        raise FileExistsError("fresh stage and result roots required")
    evaluator_hashes = {path: digest(PROJECT / path) for path in EVALUATOR_FILES}
    adapter_manifest_sha256 = digest(Path(__file__).with_name("manifest.json"))
    stage_root.mkdir(parents=False)
    write_new(stage_root / "launch.json", {
        "status": "ROUND124_FROZEN_GENERALIZATION_STARTED",
        "dataset": dataset_name, "sequences": len(dataset), "frames": frames,
        "searches": searches, "round124_checkpoint_sha256": CHECKPOINT_SHA256,
        "round124_numeric_sha256": NUMERIC_SHA256,
        "round124_manifest_sha256": IMPLEMENTATION_MANIFEST_SHA256,
        "threshold": THRESHOLD, "temporal_cooldown_frames": 8,
        "training_seed": 20261109, "sealed_file_count": len(manifest["files"]),
        "invalid_feedback_policy": "post_selection_keep_fallback",
        "test_data_calibration": False, "dataset_specific_tuning": False,
        "retraining": False, "evaluator_sha256": evaluator_hashes,
        "adapter_manifest_sha256": adapter_manifest_sha256,
        "started_unix": time.time(),
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
    audit = paired(
        dataset, current_dir, native_dir, stage_root / "sequence_scopes",
        stage_root / "fallback_counts", searches,
    )
    if audit["invalid_feedback_fallbacks"] != prediction["invalid_feedback_fallbacks"]:
        raise RuntimeError("fallback count closure mismatch")
    result = {
        "status": "ROUND124_FROZEN_GENERALIZATION_COMPLETE",
        "dataset": dataset_name, "prediction": prediction,
        "official": official, "native_same_run_official": native_official,
        "paired_interval": audit["paired_interval"], "action_totals": audit["actions"],
        "invalid_feedback_fallbacks": audit["invalid_feedback_fallbacks"],
        "feedback_commits": audit["feedback_commits"],
        "suppressed_nonkeep_proposals": audit["suppressed_nonkeep_proposals"],
        "reference_fixed_rqc30": config["reference"],
        "delta_vs_fixed_rqc30_pp": {
            key: official["metrics"][key] - config["reference"][key] for key in config["reference"]
        },
        "delta_vs_native_same_run_pp": {
            key: official["metrics"][key] - native_official["metrics"][key]
            for key in config["reference"]
        },
        "round124_checkpoint_sha256": CHECKPOINT_SHA256,
        "round124_numeric_sha256": NUMERIC_SHA256,
        "round124_manifest_sha256": IMPLEMENTATION_MANIFEST_SHA256,
        "threshold": calibration["threshold"], "temporal_cooldown_frames": 8,
        "training_seed": 20261109,
        "invalid_feedback_policy": "post_selection_keep_fallback",
        "dataset_specific_tuning": False, "retraining": False,
        "evaluator_sha256": evaluator_hashes,
        "adapter_manifest_sha256": adapter_manifest_sha256,
        "seconds": time.monotonic() - began,
    }
    write_new(stage_root / "paired_sequences.json", audit["rows"])
    write_new(stage_root / "official_complete.json", result)
    replace(stage_root / "progress.json", {"phase": "COMPLETE", "updated_unix": time.time()})
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: run_dataset DATASET STAGE_ROOT")
    main(sys.argv[1], sys.argv[2])


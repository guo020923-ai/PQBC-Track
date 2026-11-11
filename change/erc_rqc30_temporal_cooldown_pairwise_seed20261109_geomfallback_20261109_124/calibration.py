import json
import os
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_numeric
from .actions import execution_mask
from .config import PROTOCOL, calibration_mask
from .io_utils import digest
from .training import load_cache


def _write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())


def calibrate(numeric_dir, cache_dir):
    numeric_dir = Path(numeric_dir)
    _, arrays = load_cache(cache_dir)
    model, manifest = load_numeric(numeric_dir); model.eval()
    index = np.arange(30000)[calibration_mask(np.arange(30000))]
    scores, masks = [], []
    with torch.inference_mode():
        for start in range(0, len(index), 256):
            chosen = index[start:start + 256]
            evidence = torch.from_numpy(arrays["evidence"][chosen]).float()
            source_executable = torch.from_numpy(arrays["executable"][chosen]).bool()
            executable = execution_mask(evidence, source_executable)
            masks.append(executable)
            scores.append(model(evidence, executable,
                                torch.zeros(len(chosen), dtype=torch.long))["utility_score"])
    score = torch.cat(scores).numpy()
    executable = torch.cat(masks).numpy(); executable[:, 0] = False
    utility = arrays["utility"][index].astype(float); source = arrays["source"][index]
    rows = np.arange(len(index)); best = np.where(executable, score, -np.inf).max(1)
    curve = []
    for target_coverage in PROTOCOL["calibration_target_coverage"]:
        threshold = float(np.quantile(best, 1 - target_coverage, interpolation="higher"))
        allowed = executable & (score >= threshold)
        active = allowed.any(1); candidate = np.where(allowed, score, -np.inf).argmax(1)
        candidate_utility = utility[rows, candidate]; chosen_utility = np.where(active, candidate_utility, 0.0)
        worst, per_source = 0.0, {}
        for value in sorted(np.unique(source)):
            selected = (source == value) & active; source_rows = source == value
            harm = float((candidate_utility[selected] <= 0).mean()) if selected.any() else 0.0
            worst = max(worst, harm)
            per_source[str(int(value))] = {
                "selected": int(selected.sum()), "coverage": float(selected.sum() / source_rows.sum()),
                "harm_rate": harm, "mean_utility_all": float(chosen_utility[source_rows].mean()),
            }
        curve.append({
            "target_coverage": target_coverage, "threshold": threshold,
            "coverage": float(active.mean()), "selected": int(active.sum()),
            "mean_utility_all": float(chosen_utility.mean()),
            "mean_utility_selected": float(candidate_utility[active].mean()),
            "harm_rate": float((candidate_utility[active] <= 0).mean()),
            "worst_source_harm_rate": worst, "per_source": per_source,
        })
    diagnostic = {
        "status": "TRAIN_EXECUTION_ALIGNED_PAIRWISE_CALIBRATION_DIAGNOSTIC", "curve": curve,
        "calibration_rows": len(index), "checkpoint_sha256": manifest["source_checkpoint_sha256"],
        "numeric_npz_sha256": manifest["numeric_npz_sha256"], "test_data_read": False,
    }
    diagnostic_path = numeric_dir / "pairwise_calibration_diagnostic.json"
    _write_new(diagnostic_path, diagnostic)
    feasible = [row for row in curve if
                row["coverage"] >= PROTOCOL["minimum_calibration_coverage"] and
                row["mean_utility_all"] > 0 and
                row["worst_source_harm_rate"] <= PROTOCOL["maximum_worst_source_harm_rate"]]
    if feasible:
        selected = max(feasible, key=lambda row: (row["mean_utility_all"], row["threshold"]))
        status = "TRAIN_FEEDBACK_PAIRWISE_CALIBRATION_VERIFIED"
        filename = "pairwise_calibration.json"; admitted = True
        label = "development-stage"
    else:
        candidates = [row for row in curve if row["coverage"] >= PROTOCOL["minimum_calibration_coverage"]]
        selected = max(candidates, key=lambda row: (row["mean_utility_all"], -row["coverage"], row["threshold"]))
        status = "PAIRWISE_EVALUATION_FALLBACK"; filename = "pairwise_evaluation_fallback.json"
        admitted = False; label = "development-stage_evaluation_fallback_not_safety_admitted"
    result = {
        "status": status, "threshold": selected["threshold"], "selected": selected,
        "curve": curve, "calibration_rows": len(index),
        "checkpoint_sha256": manifest["source_checkpoint_sha256"],
        "numeric_npz_sha256": manifest["numeric_npz_sha256"],
        "cache_manifest_sha256": manifest["cache_manifest_sha256"],
        "test_data_read": False, "admission_constraints_passed": admitted,
        "temporal_cooldown_frames": PROTOCOL["temporal_cooldown_frames"],
        "temporal_cooldown_selection": PROTOCOL["temporal_cooldown_selection"],
        "reporting_label": label,
    }
    result_path = numeric_dir / filename; _write_new(result_path, result)
    result["calibration_sha256"] = digest(result_path)
    result["diagnostic_sha256"] = digest(diagnostic_path)
    return result

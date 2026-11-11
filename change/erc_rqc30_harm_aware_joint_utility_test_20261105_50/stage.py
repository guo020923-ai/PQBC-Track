"""One fixed harm-aware joint-utility Epoch-100 TNL2K development evaluation."""

import json
from pathlib import Path
import time

from change.erc_rqc30_epoch_supervised_abstention_test_20261104_42 import stage as base
from .fast_runner import run_persistent_dataset


DATASET, MODE = "tnl2k", "NLBBOX"
SEQUENCES, FRAMES, SEARCHES = 700, 516038, 515338
RESULT_NAME = "baseline_base_head_rqc30_harm_aware_joint_utility_erc100_20261105_50"
RUNID = 110550
NATIVE_NAME = "baseline_base_head_rqc30_harm_aware_native_same_run_20261105_50"
NATIVE_RUNID = 110551
REFERENCE_METRICS = {
    "AUC": 63.02118818112782,
    "OP50": 73.85304531233652,
    "OP75": 59.13824240182127,
    "Precision": 66.49108568055289,
    "Norm Precision": 80.40470445666995,
}


def run_stage(training_audit, numeric_dir, stage_root, processes=8):
    if training_audit.get("status") != "COMPLETE_VERIFIED":
        raise PermissionError("verified training audit required")
    numeric_dir, stage_root = Path(numeric_dir), Path(stage_root)
    manifest = json.loads((numeric_dir / "manifest.json").read_text())
    if manifest.get("epoch") != 100 or manifest.get("optimizer_steps") != 187500:
        raise ValueError("fixed Epoch-100 numeric identity required")
    if manifest.get("source_checkpoint_sha256") != training_audit.get("final_checkpoint_sha256"):
        raise ValueError("audited checkpoint binding mismatch")
    numeric = numeric_dir / "erc_epoch100_numeric.npz"
    if base._digest(numeric) != manifest.get("numeric_npz_sha256"):
        raise ValueError("numeric checkpoint changed")

    base.RESULT_NAME, base.RUNID = RESULT_NAME, RUNID
    base.NATIVE_NAME, base.NATIVE_RUNID = NATIVE_NAME, NATIVE_RUNID
    base.REFERENCE_METRICS = REFERENCE_METRICS
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
    base._write_new(stage_root / "manifest.json", {
        "dataset": DATASET, "mode": MODE, "sequences": SEQUENCES, "frames": FRAMES,
        "searches": SEARCHES, "result_name": RESULT_NAME, "runid": RUNID,
        "native_name": NATIVE_NAME, "native_runid": NATIVE_RUNID,
        "historical_reference_metrics": REFERENCE_METRICS,
        "fixed_checkpoint_sha256": training_audit["final_checkpoint_sha256"],
        "numeric_npz_sha256": manifest["numeric_npz_sha256"], "persistent_workers": int(processes),
        "baseline_gpu_replay": False, "official_evaluations": 2,
        "same_forward_native_shadow": True, "automatic_retry": False,
        "writer_epoch_compatibility_name": 30, "scientific_epoch": 100,
        "reporting_scope": "development-stage evaluation informed by prior benchmark feedback",
    })
    began = time.monotonic()
    prediction = run_persistent_dataset(dataset, {
        "checkpoint_dir": str(numeric_dir), "results_dir": str(Path(current.results_dir)),
        "native_results_dir": str(Path(native.results_dir)), "scope_root": str(stage_root / "sequence_scopes"),
        "dataset_name": DATASET, "mode": MODE, "source_epoch": 30, "writer_epoch": 30, "gpu_id": 0,
    }, processes=processes)
    base._replace(stage_root / "progress.json", {"phase": "OFFICIAL_METRICS", "updated_unix": time.time()})
    official = base._evaluate_once(current, dataset, stage_root / "official_report_current")
    native_official = base._evaluate_once(native, dataset, stage_root / "official_report_native_same_run")
    base._replace(stage_root / "progress.json", {"phase": "PAIRED_AUDIT", "updated_unix": time.time()})
    paired = base._paired(dataset, current_dir, native_dir, stage_root / "sequence_scopes")
    result = {
        "status": "TEST_COMPLETE_PENDING_READONLY_AUDIT", "prediction": prediction,
        "official": official, "native_same_run_official": native_official,
        "paired_interval": paired["paired_interval"], "action_totals": paired["action_totals"],
        "delta_auc_pp_vs_native_same_run": official["metrics"]["AUC"] - native_official["metrics"]["AUC"],
        "delta_auc_pp_vs_historical_rqc30": official["metrics"]["AUC"] - REFERENCE_METRICS["AUC"],
        "seconds": time.monotonic() - began, "automatic_retrain": False,
        "reporting_scope": "development-stage evaluation",
    }
    base._write_new(stage_root / "paired_sequences.json", paired["rows"])
    base._write_new(stage_root / "official_complete.json", result)
    base._replace(stage_root / "progress.json", {"phase": "TEST_COMPLETE", "updated_unix": time.time()})
    return result


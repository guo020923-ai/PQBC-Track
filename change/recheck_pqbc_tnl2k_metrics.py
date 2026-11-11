"""Recompute official TNL2K metrics from the completed PQBC-Track predictions."""

import argparse
import json
import time
from pathlib import Path

RESULT_NAME = (
    "baseline_base_head_rqc30_temporal_cooldown8_pairwise_erc100_"
    "seed20261109_geomfallback_20261109_124"
)
RUN_ID = 110925


def main():
    parser = argparse.ArgumentParser(
        description="Recompute the five official metrics from existing FULL700 predictions."
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    print("Loading the official TNL2K evaluator...", flush=True)
    from lib.test.evaluation import get_dataset
    from lib.test.evaluation.tracker import Tracker
    from change.erc_rqc30_epoch_supervised_abstention_test_20261104_42.stage import (
        _evaluate_once,
    )

    report = args.report or Path("outputs") / (
        "pqbc_track_metric_recheck_" + time.strftime("%Y%m%d_%H%M%S")
    )
    if report.exists():
        raise FileExistsError(report)
    tracker = Tracker(
        "uvltrack", RESULT_NAME, "tnl2k", run_id=RUN_ID, eval=True
    )
    dataset = get_dataset("tnl2k")
    print("Recomputing official metrics from 700 completed sequences...", flush=True)
    result = _evaluate_once(tracker, dataset, report)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print("REPORT_DIR", report, flush=True)


if __name__ == "__main__":
    main()

"""Fresh result identities for ordinal-risk FULL700 evaluation."""

from change.erc_rqc30_harm_aware_joint_utility_test_20261105_50 import stage as inherited

from .fast_runner import run_persistent_dataset


inherited.RESULT_NAME = "baseline_base_head_rqc30_ordinal_risk_erc100_20261105_57"
inherited.RUNID = 110560
inherited.NATIVE_NAME = "baseline_base_head_rqc30_native_same_run_ordinal_risk_20261105_57"
inherited.NATIVE_RUNID = 110561
inherited.run_persistent_dataset = run_persistent_dataset


def run_stage(training_audit, numeric_dir, stage_root, processes=8):
    return inherited.run_stage(
        training_audit, numeric_dir, stage_root, processes=processes
    )

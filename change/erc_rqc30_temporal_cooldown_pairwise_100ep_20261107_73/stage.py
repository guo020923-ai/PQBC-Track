from change.erc_rqc30_harm_aware_joint_utility_test_20261105_50 import stage as inherited
from .fast_runner import run_persistent_dataset


inherited.RESULT_NAME = "baseline_base_head_rqc30_temporal_cooldown8_pairwise_erc100_20261107_73"
inherited.RUNID = 110774
inherited.NATIVE_NAME = "baseline_base_head_rqc30_temporal_cooldown8_feedback_shadow_20261107_73"
inherited.NATIVE_RUNID = 110775
inherited.run_persistent_dataset = run_persistent_dataset


def run_stage(training_audit, numeric_dir, stage_root, processes=8):
    result = inherited.run_stage(training_audit, numeric_dir, stage_root, processes=processes)
    result["scientific_variable"] = "train_only_selected_temporal_feedback_cooldown"
    result["temporal_cooldown_frames"] = 8
    result["cooldown_selection"] = "TRAIN_only_64frame_clip_ablation"
    result["feedback_conditioned_native_shadow"] = True
    result["independent_native_baseline"] = False
    result["deterministic_source_execution"] = False
    result["determinism_mode"] = "cudnn_deterministic_cublas_workspace_warn_only_unsupported_cuda_cumsum"
    return result

from change.erc_rqc30_harm_aware_joint_utility_test_20261105_50 import stage as inherited
from .fast_runner import run_persistent_dataset


inherited.RESULT_NAME = "baseline_base_head_rqc30_pairwise_erc100_outputonly_ablation_20261109_130"
inherited.RUNID = 110930
inherited.NATIVE_NAME = "baseline_base_head_rqc30_outputonly_native_shadow_20261109_130"
inherited.NATIVE_RUNID = 110931
inherited.run_persistent_dataset = run_persistent_dataset


def run_stage(training_audit, numeric_dir, stage_root, processes=8):
    result = inherited.run_stage(training_audit, numeric_dir, stage_root, processes=processes)
    result["scientific_variable"] = "accepted_erc_output_without_next_state_feedback"
    result["temporal_cooldown_frames"] = 8
    result["cooldown_selection"] = "TRAIN_only_64frame_clip_ablation"
    result["feedback_conditioned_native_shadow"] = False
    result["independent_native_baseline"] = False
    result["deterministic_source_execution"] = False
    result["determinism_mode"] = "cudnn_deterministic_cublas_workspace_warn_only_unsupported_cuda_cumsum"
    result["training_seed"] = 20261109
    result["ablation_reference"] = "Round124_final_feedback_enabled"
    result["source_history_feedback"] = False
    result["invalid_feedback_policy"] = "post_selection_keep_fallback"
    return result

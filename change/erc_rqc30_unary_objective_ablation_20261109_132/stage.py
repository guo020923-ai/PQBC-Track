from change.erc_rqc30_harm_aware_joint_utility_test_20261105_50 import stage as inherited
from .fast_runner import run_persistent_dataset


inherited.RESULT_NAME = "baseline_base_head_rqc30_unaryobjective_erc100_cooldown8_geomfallback_20261109_132"
inherited.RUNID = 110932
inherited.NATIVE_NAME = "baseline_base_head_rqc30_unaryobjective_native_shadow_20261109_132"
inherited.NATIVE_RUNID = 110933
inherited.run_persistent_dataset = run_persistent_dataset


def run_stage(training_audit, numeric_dir, stage_root, processes=8):
    result = inherited.run_stage(training_audit, numeric_dir, stage_root, processes=processes)
    result["scientific_variable"] = "unary_keep_relative_objective_instead_of_pairwise_objective"
    result["temporal_cooldown_frames"] = 8
    result["cooldown_selection"] = "TRAIN_only_64frame_clip_ablation"
    result["feedback_conditioned_native_shadow"] = True
    result["independent_native_baseline"] = False
    result["deterministic_source_execution"] = False
    result["determinism_mode"] = "cudnn_deterministic_cublas_workspace_warn_only_unsupported_cuda_cumsum"
    result["training_seed"] = 20261109
    result["ablation_reference"] = "Round124_pairwise_final"
    result["invalid_feedback_policy"] = "post_selection_keep_fallback"
    return result

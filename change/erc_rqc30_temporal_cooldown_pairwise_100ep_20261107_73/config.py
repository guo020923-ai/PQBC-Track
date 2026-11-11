PROTOCOL = {
    "version": "rqc30_execution_aligned_pairwise_cooldown8_100ep_v1",
    "cache_seed": 20261104,
    "training_seed": 20261105,
    "source_rqc_epoch": 30,
    "source_rqc_checkpoint_sha256": "1524962167d8f5b60a33b29960feae47f82b8e2a17869fc628e2f543978267bd",
    "source_cache_manifest_sha256": "26162a2efa52becdbb1f95e2a1a81eeecdfc25bc912f737591749d5f44ac9d1d",
    "cache_examples": 30000,
    "epochs": 100,
    "batch_size": 16,
    "steps_per_epoch": 1875,
    "optimizer_steps": 187500,
    "peak_learning_rate": 5e-5,
    "minimum_learning_rate": 1e-6,
    "warmup_epochs": 5,
    "weight_decay": 1e-4,
    "gradient_clip": 1.0,
    "candidate_actions": 25,
    "evidence_dim": 48,
    "hidden_dim": 64,
    "pair_temperature": 1.0 / 21.0,
    "action_basis": "direct_quarter_mapped_execution_image",
    "shrink_factor": 0.25,
    "feedback_mode": "quarter_shrink_crop_feedback",
    "temporal_cooldown_frames": 8,
    "temporal_cooldown_selection": "TRAIN_only_64frame_clip_ablation",
    "deterministic_source_execution": False,
    "determinism_mode": "cudnn_deterministic_cublas_workspace_warn_only_unsupported_cuda_cumsum",
    "calibration_block_rows": 64,
    "calibration_block_modulus": 5,
    "calibration_block_remainder": 0,
    "calibration_target_coverage": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8],
    "minimum_calibration_coverage": 0.005,
    "maximum_worst_source_harm_rate": 0.15,
    "checkpoint_selection": "fixed_epoch100_no_best",
    "test_data_in_training_or_calibration": False,
}


def calibration_mask(rows):
    return ((rows // PROTOCOL["calibration_block_rows"])
            % PROTOCOL["calibration_block_modulus"]
            == PROTOCOL["calibration_block_remainder"])


def validate_protocol():
    assert PROTOCOL["cache_examples"] == 30000
    assert PROTOCOL["epochs"] == 100 and PROTOCOL["optimizer_steps"] == 187500
    assert PROTOCOL["action_basis"] == "direct_quarter_mapped_execution_image"
    assert PROTOCOL["shrink_factor"] == 0.25
    assert PROTOCOL["feedback_mode"] == "quarter_shrink_crop_feedback"
    assert PROTOCOL["temporal_cooldown_frames"] == 8
    assert PROTOCOL["temporal_cooldown_selection"] == "TRAIN_only_64frame_clip_ablation"
    assert PROTOCOL["test_data_in_training_or_calibration"] is False
    return PROTOCOL

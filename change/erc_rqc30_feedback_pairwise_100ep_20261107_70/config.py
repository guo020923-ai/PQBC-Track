PROTOCOL = {
    "version": "rqc30_feedback_pairwise_100ep_v1",
    "cache_seed": 20261104,
    "training_seed": 20261105,
    "source_rqc_epoch": 30,
    "source_rqc_checkpoint_sha256": "1524962167d8f5b60a33b29960feae47f82b8e2a17869fc628e2f543978267bd",
    "behavior_checkpoint_sha256": "aa3311c321a9bd4c4d27a288b87abe12b5e588cc5c396f6168fc7ecbb8873c1f",
    "behavior_numeric_sha256": "67fcbaab808e141ef4625e4d671b4f25177cf221007e95647c0d920b34e1bf54",
    "behavior_threshold": 0.5843265056610107,
    "cache_examples": 30000,
    "cache_warmup_frames": 16,
    "cache_frames_per_clip": 64,
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
    "feedback_mode": "quarter_shrink_crop_feedback",
    "shrink_factor": 0.25,
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
    assert PROTOCOL["feedback_mode"] == "quarter_shrink_crop_feedback"
    assert PROTOCOL["shrink_factor"] == 0.25
    assert PROTOCOL["deterministic_source_execution"] is False
    assert PROTOCOL["determinism_mode"].startswith("cudnn_deterministic_cublas_workspace")
    assert PROTOCOL["test_data_in_training_or_calibration"] is False
    return PROTOCOL


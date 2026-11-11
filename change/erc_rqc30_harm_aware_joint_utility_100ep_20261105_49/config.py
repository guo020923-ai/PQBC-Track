"""Predetermined TRAIN-only protocol for the harm-aware joint utility ERC."""

PROTOCOL = {
    "version": "rqc30_harm_aware_joint_utility_100ep_v1",
    "cache_seed": 20261104,
    "training_seed": 20261105,
    "source_rqc_epoch": 30,
    "source_rqc_checkpoint_sha256": "1524962167d8f5b60a33b29960feae47f82b8e2a17869fc628e2f543978267bd",
    "data_configuration": "baseline_base_head_directional_center_guard_bs16_30ep_20261001.yaml",
    "datasets_name": ["GOT10K_vottrain", "LASOT", "TNL2K", "OTB99"],
    "datasets_ratio": [1, 1, 1, 0.2],
    "effective_vl_ratio": {"LASOT": 5.0 / 11.0, "TNL2K": 5.0 / 11.0, "OTB99": 1.0 / 11.0},
    "erc_active_population": "original_loader_VL_branch_only",
    "cache_examples": 30000,
    "cache_manifest_sha256": "993b20ade7aa8ab491c5d87fcdc58350d7f218dea87b470bd9b8f6c58711b56c",
    "epochs": 100,
    "batch_size": 16,
    "steps_per_epoch": 1875,
    "optimizer_steps": 187500,
    "optimizer": "AdamW",
    "peak_learning_rate": 5e-5,
    "minimum_learning_rate": 1e-6,
    "warmup_epochs": 5,
    "scheduler": "linear_warmup5_then_cosine95_no_restart",
    "weight_decay": 1e-4,
    "gradient_clip": 1.0,
    "candidate_actions": 25,
    "evidence_dim": 48,
    "minimum_deploy_gain": 1.0 / 21.0,
    "utility_scale": 1.0 / 21.0,
    "loss": "group_balanced_smooth_l1_plus_soft_utility_KL_plus_false_positive_harm",
    "decision": "joint_KEEP_and_24_action_competition_with_metric_aligned_gain_floor",
    "checkpoint_selection": "fixed_epoch100_no_best",
    "frozen_rqc": True,
    "frozen_head": True,
    "frozen_backbone": True,
    "test_data_in_training": False,
    "automatic_test": True,
}


def validate_protocol():
    assert PROTOCOL["epochs"] == 100
    assert PROTOCOL["optimizer_steps"] == 187500
    assert PROTOCOL["cache_examples"] == 30000
    assert PROTOCOL["batch_size"] == 16
    assert PROTOCOL["steps_per_epoch"] == 1875
    assert PROTOCOL["minimum_deploy_gain"] == 1.0 / 21.0
    assert PROTOCOL["source_rqc_epoch"] == 30
    assert PROTOCOL["frozen_rqc"] and PROTOCOL["frozen_head"] and PROTOCOL["frozen_backbone"]
    assert PROTOCOL["automatic_test"]
    return PROTOCOL


"""Frozen protocol for the single authorized RQC-style ERC training."""

PROTOCOL = {
    "version": "rqc30_epoch_supervised_erc_v1",
    "seed": 20261104,
    "source_rqc_epoch": 30,
    "source_rqc_checkpoint_sha256": "1524962167d8f5b60a33b29960feae47f82b8e2a17869fc628e2f543978267bd",
    "data_configuration": "baseline_base_head_directional_center_guard_bs16_30ep_20261001.yaml",
    "datasets_name": ["GOT10K_vottrain", "LASOT", "TNL2K", "OTB99"],
    "datasets_ratio": [1, 1, 1, 0.2],
    "erc_active_population": "original_loader_VL_branch_only",
    "cache_examples": 30000,
    "epochs": 30,
    "batch_size": 16,
    "steps_per_epoch": 1875,
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "gradient_clip": 1.0,
    "scheduler": "CosineAnnealingLR_epoch30_to_zero",
    "cache_warmup_frames": 16,
    "cache_frames_per_clip": 64,
    "candidate_actions": 25,
    "evidence_dim": 48,
    "label": "official_style_success_auc21_with_continuous_iou_tiebreak_1e-3",
    "loss": "complete_action_structured_regret_hinge",
    "checkpoint_selection": "fixed_epoch30_no_best",
    "frozen_rqc": True,
    "frozen_head": True,
    "frozen_backbone": True,
    "test_data_in_training": False,
    "automatic_test": False,
    "automatic_retry": False,
}


def validate_protocol():
    assert PROTOCOL["cache_examples"] == 30000
    assert PROTOCOL["epochs"] == 30
    assert PROTOCOL["batch_size"] == 16
    assert PROTOCOL["steps_per_epoch"] == 30000 // 16 == 1875
    assert PROTOCOL["candidate_actions"] == 25
    assert PROTOCOL["evidence_dim"] == 48
    return PROTOCOL

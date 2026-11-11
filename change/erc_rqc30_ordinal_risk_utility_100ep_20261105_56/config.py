"""Protocol for unified harmful/marginal/actionable action scoring."""

PROTOCOL={
 "version":"rqc30_unified_ordinal_risk_utility_100ep_v1",
 "training_seed":20261105,"source_rqc_epoch":30,
 "source_rqc_checkpoint_sha256":"1524962167d8f5b60a33b29960feae47f82b8e2a17869fc628e2f543978267bd",
 "cache_examples":30000,"cache_manifest_sha256":"993b20ade7aa8ab491c5d87fcdc58350d7f218dea87b470bd9b8f6c58711b56c",
 "epochs":100,"batch_size":16,"steps_per_epoch":1875,"optimizer_steps":187500,
 "peak_learning_rate":5e-5,"minimum_learning_rate":1e-6,"warmup_epochs":5,"weight_decay":1e-4,"gradient_clip":1.0,
 "candidate_actions":25,"evidence_dim":48,"hidden_dim":64,
 "harmful_upper_utility":0.0,"actionable_lower_utility":1.0/21.0,
 "calibration_block_rows":64,"calibration_block_modulus":5,"calibration_block_remainder":0,
 "risk_threshold_grid":[0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.70,0.80],
 "minimum_calibration_coverage":0.005,"maximum_worst_source_harm_rate":0.15,
 "threshold_selection":"maximize_mean_calibration_utility_subject_to_positive_mean_minimum_coverage_and_each_source_harm_le_0.15",
 "risk_score":"P(actionable)-P(harmful)","checkpoint_selection":"fixed_epoch100_no_best",
 "frozen_rqc":True,"frozen_native_head":True,"frozen_backbone":True,"test_data_in_training_or_calibration":False,
}
def calibration_mask(rows):return (rows//PROTOCOL['calibration_block_rows'])%PROTOCOL['calibration_block_modulus']==PROTOCOL['calibration_block_remainder']
def validate_protocol():
 assert PROTOCOL['epochs']==100 and PROTOCOL['optimizer_steps']==187500 and PROTOCOL['candidate_actions']==25
 assert PROTOCOL['actionable_lower_utility']==1.0/21.0
 assert PROTOCOL['frozen_rqc'] and PROTOCOL['frozen_native_head'] and PROTOCOL['frozen_backbone']
 assert not PROTOCOL['test_data_in_training_or_calibration'];return PROTOCOL

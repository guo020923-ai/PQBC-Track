from lib import registry
from .modality_adaptive_box_head import ModalityAdaptiveBoxHead

@registry.HEADS.register('modality_adaptive_box_head')
def build_modality_adaptive_box_head(cfg):
    stride = 16
    feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
    channel = cfg.MODEL.HEAD.HEAD_DIM
    head = ModalityAdaptiveBoxHead(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel, feat_sz=feat_sz, stride=stride, 
                        cls_tokenize=cfg.MODEL.HEAD.CLS_TOKENIZE, offset_sigmoid=cfg.MODEL.HEAD.OFFSET_SIGMOID,
                        joint_cls=cfg.MODEL.HEAD.JOINT_CLS, drop_rate=cfg.MODEL.HEAD.DROP, softmax_one=cfg.MODEL.HEAD.SOFTMAX_ONE,
                        grounding_dilation=cfg.MODEL.HEAD.GROUNDING_DILATION, contrastive_conv=cfg.MODEL.HEAD.CONTRASTIVE_CONV,
                        erc_enabled=cfg.MODEL.HEAD.ERC.ENABLED, erc_hidden_dim=cfg.MODEL.HEAD.ERC.HIDDEN_DIM,
                        erc_drop_rate=cfg.MODEL.HEAD.ERC.DROP, erc_max_alpha=cfg.MODEL.HEAD.ERC.MAX_ALPHA,
                        erc_eps=cfg.MODEL.HEAD.ERC.EPS, erc_log_ratio_clip=cfg.MODEL.HEAD.ERC.LOG_RATIO_CLIP,
                        erc_detach_features=cfg.MODEL.HEAD.ERC.DETACH_FEATURES,
                        erc_reliability_gated_alpha=cfg.MODEL.HEAD.ERC.RELIABILITY_GATED_ALPHA,
                        erc_detach_reliability_gate=cfg.MODEL.HEAD.ERC.DETACH_RELIABILITY_GATE,
                        erc_reliability_gate_floor=cfg.MODEL.HEAD.ERC.RELIABILITY_GATE_FLOOR,
                        erc_apply_vl_tracking_only=cfg.MODEL.HEAD.ERC.APPLY_VL_TRACKING_ONLY,
                        erc_match_test_hann=cfg.MODEL.HEAD.ERC.MATCH_TEST_HANN,
                        erc_design=cfg.MODEL.HEAD.ERC.DESIGN,
                        erc_direction_scale=cfg.MODEL.HEAD.ERC.DIRECTION_SCALE,
                        erc_delta_scale=cfg.MODEL.HEAD.ERC.DELTA_SCALE,
                        erc_gate_prior=cfg.MODEL.HEAD.ERC.GATE_PRIOR,
                        erc_regional_gate_prior=cfg.MODEL.HEAD.ERC.REGIONAL_GATE_PRIOR,
                        erc_detach_reliability_for_alpha=cfg.MODEL.HEAD.ERC.DETACH_RELIABILITY_FOR_ALPHA,
                        erc_force_off=cfg.MODEL.HEAD.ERC.FORCE_OFF,
                        erc_global_alpha_scale=cfg.MODEL.HEAD.ERC.GLOBAL_ALPHA_SCALE,
                        erc_region_alpha_scale=cfg.MODEL.HEAD.ERC.REGION_ALPHA_SCALE,
                        erc_regional_hidden_dim=cfg.MODEL.HEAD.ERC.REGIONAL_HIDDEN_DIM,
                        erc_response_topk=cfg.MODEL.HEAD.ERC.RESPONSE_TOPK,
                        erc_inference_mode=cfg.MODEL.HEAD.ERC.INFERENCE_MODE,
                        erc_pair_beta_scale=cfg.MODEL.HEAD.ERC.PAIR_BETA_SCALE,
                        erc_pair_exclusion_radius=cfg.MODEL.HEAD.ERC.PAIR_EXCLUSION_RADIUS,
                        erc_pair_mask_sigma=cfg.MODEL.HEAD.ERC.PAIR_MASK_SIGMA,
                        erc_local_window=cfg.MODEL.HEAD.ERC.LOCAL_WINDOW,
                        erc_local_hidden_dim=cfg.MODEL.HEAD.ERC.LOCAL_HIDDEN_DIM,
                        erc_local_delta_max=cfg.MODEL.HEAD.ERC.LOCAL_DELTA_MAX,
                        erc_local_reliability_prior=cfg.MODEL.HEAD.ERC.LOCAL_RELIABILITY_PRIOR,
                        erc_v13_d_max=cfg.MODEL.HEAD.ERC.V13_D_MAX,
                        erc_v13_sigmas=cfg.MODEL.HEAD.ERC.V13_SIGMAS,
                        erc_v13_log_c_min=cfg.MODEL.HEAD.ERC.V13_LOG_C_MIN,
                        erc_v13_log_t_min=cfg.MODEL.HEAD.ERC.V13_LOG_T_MIN,
                        erc_v13_log_s0_min=cfg.MODEL.HEAD.ERC.V13_LOG_S0_MIN,
                        erc_v14_gate_prior=cfg.MODEL.HEAD.ERC.V14_GATE_PRIOR,
                        erc_trc_hidden_dim=cfg.MODEL.HEAD.ERC.TRC_HIDDEN_DIM,
                        erc_trc_temperature=cfg.MODEL.HEAD.ERC.TRC_TEMPERATURE,
                        erc_trc_d_max=cfg.MODEL.HEAD.ERC.TRC_D_MAX,
                        erc_trc_permission_threshold=cfg.MODEL.HEAD.ERC.TRC_PERMISSION_THRESHOLD,
                        erc_trc_permission_initial_probability=cfg.MODEL.HEAD.ERC.TRC_PERMISSION_INITIAL_PROBABILITY,
                        erc_trc_direction_t_threshold=cfg.MODEL.HEAD.ERC.TRC_DIRECTION_T_THRESHOLD,
                        erc_trc_branch_permission_threshold=cfg.MODEL.HEAD.ERC.TRC_BRANCH_PERMISSION_THRESHOLD,
                        erc_trc_branch_permission_initial_probability=cfg.MODEL.HEAD.ERC.TRC_BRANCH_PERMISSION_INITIAL_PROBABILITY,
                        erc_trc_utility_threshold=cfg.MODEL.HEAD.ERC.TRC_UTILITY_THRESHOLD,
                        erc_feature_hidden_dim=cfg.MODEL.HEAD.ERC.FEATURE_HIDDEN_DIM,
                        erc_feature_max_delta=cfg.MODEL.HEAD.ERC.FEATURE_MAX_DELTA,
                        erc_feature_reliability_temperature=cfg.MODEL.HEAD.ERC.FEATURE_RELIABILITY_TEMPERATURE,
                        erc_feature_response_temperature=cfg.MODEL.HEAD.ERC.FEATURE_RESPONSE_TEMPERATURE,
                        erc_dense_d_max=cfg.MODEL.HEAD.ERC.DENSE_D_MAX,
                        erc_dense_spatial_prior=cfg.MODEL.HEAD.ERC.DENSE_SPATIAL_PRIOR,
                        erc_action_alpha_values=cfg.MODEL.HEAD.ERC.ACTION_ALPHA_VALUES,
                        erc_action_temperature=cfg.MODEL.HEAD.ERC.ACTION_TEMPERATURE,
                        erc_reachability_magnitudes=cfg.MODEL.HEAD.ERC.REACHABILITY_MAGNITUDES,
                        erc_av1_utility_temperature=cfg.MODEL.HEAD.ERC.AV1_UTILITY_TEMPERATURE,
                        erc_av1_gate_scale=cfg.MODEL.HEAD.ERC.AV1_GATE_SCALE,
                        erc_av2_temporal_ema=cfg.MODEL.HEAD.ERC.AV2_TEMPORAL_EMA,
                        erc_av2_release_margin=cfg.MODEL.HEAD.ERC.AV2_RELEASE_MARGIN,
                        erc_av2_enable_memory=cfg.MODEL.HEAD.ERC.AV2_ENABLE_MEMORY,
                        erc_dr2_safety_gate_prior=cfg.MODEL.HEAD.ERC.DR2_SAFETY_GATE_PRIOR,
                        erc_dr2_temporal_ema=cfg.MODEL.HEAD.ERC.DR2_TEMPORAL_EMA,
                        erc_dr2_enable_memory=cfg.MODEL.HEAD.ERC.DR2_ENABLE_MEMORY)
    if cfg.MODEL.HEAD.ERC.DR4_EXACT_DEPLOY_SCORE:
        if cfg.MODEL.HEAD.ERC.DESIGN != 'target_reference_realized_action_dense_erc_dr3':
            raise ValueError('DR4 exact deployment scoring requires the DR3 response architecture')
        if not cfg.MODEL.HEAD.SOFTMAX_ONE:
            raise ValueError('DR4 exact deployment scoring currently requires SOFTMAX_ONE')
        head.erc.exact_deploy_score = True
    return head

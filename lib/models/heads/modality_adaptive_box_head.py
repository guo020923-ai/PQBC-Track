import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.models.heads.utils import conv, DistributionBasedCrossAttention
from lib.models.heads.evidence_reliability_calibration import EvidenceReliabilityCalibration
from lib.models.heads.response_global_regional_erc_v7 import ResponseGlobalRegionalERCV7
from lib.models.heads.response_hann_hardpeak_erc_v8 import ResponseHannHardPeakERCV8
from lib.models.heads.response_hann_gate_erc_v8c import ResponseHannGateERCV8C
from lib.models.heads.response_regional_effective_erc_v8d import ResponseRegionalEffectiveERCV8D
from lib.models.heads.response_precision_incremental_erc_v9 import ResponsePrecisionIncrementalERCV9
from lib.models.heads.response_competition_pair_erc_v10 import ResponseCompetitionPairERCV10
from lib.models.heads.response_local_spatial_erc_v11 import ResponseLocalSpatialERCV11
from lib.models.heads.response_local_reliability_erc_v12 import ResponseLocalReliabilityERCV12
from lib.models.heads.response_branch_spatial_gate_erc_v13 import ResponseBranchSpatialGateERCV13
from lib.models.heads.response_intervention_gate_erc_v14 import ResponseInterventionGateERCV14
from lib.models.heads.response_worthiness_gate_erc_v15 import ResponseWorthinessGateERCV15
from lib.models.heads.response_asymmetric_worthiness_gate_erc_v16 import ResponseAsymmetricWorthinessGateERCV16
from lib.models.heads.response_selective_abstention_worthiness_gate_erc_v17 import ResponseSelectiveAbstentionWorthinessGateERCV17
from lib.models.heads.response_preservation_veto_gate_erc_v18 import ResponsePreservationVetoGateERCV18
from lib.models.heads.target_reference_erc import TargetReferenceERC
from lib.models.heads.target_reference_erc_v2 import TargetReferenceERCV2
from lib.models.heads.target_reference_erc_v3 import TargetReferenceERCV3
from lib.models.heads.target_reference_erc_v4 import TargetReferenceERCV4
from lib.models.heads.target_reference_erc_v4b import TargetReferenceERCV4B
from lib.models.heads.target_reference_erc_v5 import TargetReferenceERCV5
from lib.models.heads.target_reference_erc_v5b import TargetReferenceERCV5B
from lib.models.heads.target_reference_erc_v5c import TargetReferenceERCV5C
from lib.models.heads.target_reference_erc_v5d import TargetReferenceERCV5D
from lib.models.heads.target_reference_erc_v5f import TargetReferenceERCV5F
from lib.models.heads.target_reference_erc_v5g import TargetReferenceERCV5G
from lib.models.heads.target_reference_erc_v5h import TargetReferenceERCV5H
from lib.models.heads.target_reference_erc_v6 import TargetReferenceERCV6
from lib.models.heads.target_reference_erc_v6b import TargetReferenceERCV6B
from lib.models.heads.reference_reliability_feature_erc import (
    ReferenceReliabilityFeatureERC,
)
from lib.models.heads.reference_reliability_feature_erc_v2 import (
    ReferenceReliabilityFeatureERCV2,
)
from lib.models.heads.reference_reliability_dense_response_erc_v3 import (
    ReferenceReliabilityDenseResponseERCV3,
)
from lib.models.heads.branch_aligned_dense_response_erc_v4 import (
    BranchAlignedDenseResponseERCV4,
)
from lib.models.heads.branch_aligned_tristate_multiaction_erc_v5 import (
    BranchAlignedTriStateMultiActionERCV5,
)
from lib.models.heads.branch_aligned_factorized_reachability_erc_v6 import (
    BranchAlignedFactorizedReachabilityERCV6,
)
from lib.models.heads.branch_aligned_iou_utility_erc_v7 import (
    BranchAlignedIoUUtilityERCV7,
)
from lib.models.heads.branch_aligned_action_value_erc_av1 import (
    BranchAlignedActionValueERCAV1,
)
from lib.models.heads.branch_aligned_action_value_erc_av1b import (
    BranchAlignedActionValueERCAV1B,
)
from lib.models.heads.branch_aligned_temporal_consensus_erc_av2 import (
    BranchAlignedTemporalConsensusERCAV2,
)
from lib.models.heads.cross_domain_global_regional_erc_v1 import (
    CrossDomainGlobalRegionalERCV1,
)
from lib.models.heads.cross_modal_consensus_residual_erc_av3 import (
    CrossModalConsensusResidualERCAV3,
)
from lib.models.heads.cross_domain_normalized_consensus_erc_av4 import (
    CrossDomainNormalizedConsensusERCAV4,
)
from lib.models.heads.cross_domain_continuous_oracle_erc_av5 import (
    CrossDomainContinuousOracleERCAV5,
)
from lib.models.heads.peak_quality_calibrated_erc_pq1 import (
    PeakQualityCalibratedERCPQ1,
)
from lib.models.heads.peak_quality_decision_aligned_erc_pq2 import (
    PeakQualityDecisionAlignedERCPQ2,
)
from lib.models.heads.peak_quality_independent_branch_erc_pq3 import (
    PeakQualityIndependentBranchERCPQ3,
)
from lib.models.heads.peak_ambiguity_target_verifier_erc_pq4 import (
    PeakAmbiguityTargetVerifierERCPQ4,
)
from lib.models.heads.target_reference_direct_dense_erc_dr1 import (
    TargetReferenceDirectDenseERCDR1,
)
from lib.models.heads.target_reference_trajectory_safe_dense_erc_dr2 import (
    TargetReferenceTrajectorySafeDenseERCDR2,
)
from lib.models.heads.target_reference_realized_action_dense_erc_dr3 import (
    TargetReferenceRealizedActionDenseERCDR3,
)
import numpy as np
from .foreground_reference_erc_fr1 import ForegroundReferenceERC, forward_fr1

class ModalityAdaptiveBoxHead(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False,
                 num_t=2, num_b=4, offset_sigmoid=True, cls_tokenize=True, joint_cls=False, 
                 drop_rate=0.0, softmax_one=False, grounding_dilation=1, contrastive_conv=False,
                 erc_enabled=False, erc_hidden_dim=64, erc_drop_rate=0.0, erc_max_alpha=0.5,
                 erc_eps=1e-6, erc_log_ratio_clip=8.0, erc_detach_features=True,
                 erc_reliability_gated_alpha=False,
                 erc_detach_reliability_gate=True,
                 erc_reliability_gate_floor=0.0,
                 erc_apply_vl_tracking_only=False,
                 erc_match_test_hann=False,
                 erc_design='legacy', erc_direction_scale=3.0,
                 erc_delta_scale=0.5, erc_gate_prior=0.05,
                 erc_regional_gate_prior=0.02,
                 erc_detach_reliability_for_alpha=True,
                 erc_force_off=False, erc_global_alpha_scale=0.10,
                 erc_region_alpha_scale=0.10, erc_regional_hidden_dim=32,
                 erc_response_topk=10, erc_inference_mode='FULL_ERC',
                 erc_pair_beta_scale=0.30, erc_pair_exclusion_radius=2.0,
                 erc_pair_mask_sigma=1.25, erc_local_window=5,
                 erc_local_hidden_dim=32, erc_local_delta_max=0.65,
                 erc_local_reliability_prior=0.25,
                 erc_v13_d_max=4.0,
                 erc_v13_sigmas=(0.75, 1.25, 2.0, 3.0),
                 erc_v13_log_c_min=-8.0, erc_v13_log_t_min=-16.0,
                 erc_v13_log_s0_min=-16.0,
                 erc_v14_gate_prior=0.20, erc_trc_hidden_dim=32,
                 erc_trc_temperature=1.0, erc_trc_d_max=4.0,
                 erc_trc_permission_threshold=0.5,
                 erc_trc_permission_initial_probability=0.5,
                 erc_trc_direction_t_threshold=0.5,
                 erc_trc_branch_permission_threshold=0.5,
                 erc_trc_branch_permission_initial_probability=0.5,
                 erc_trc_utility_threshold=0.05,
                 erc_feature_hidden_dim=64,
                 erc_feature_max_delta=0.10,
                  erc_feature_reliability_temperature=1.0,
                  erc_feature_response_temperature=0.10,
                  erc_dense_d_max=4.0,
                   erc_dense_spatial_prior=0.5,
                   erc_action_alpha_values=(0.0, 0.10, 0.20, -0.10, -0.20),
                   erc_action_temperature=1.0,
                   erc_reachability_magnitudes=(0.10, 0.20, 0.40, 0.60, 0.80),
                   erc_av1_utility_temperature=0.25,
                   erc_av1_gate_scale=0.50,
                   erc_av2_temporal_ema=0.60,
                   erc_av2_release_margin=0.02,
                   erc_av2_enable_memory=True,
                   erc_dr2_safety_gate_prior=0.20,
                   erc_dr2_temporal_ema=0.60,
                   erc_dr2_enable_memory=True):
        super(ModalityAdaptiveBoxHead, self).__init__()
        self.num_t = num_t
        self.num_b = num_b
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        self.offset_sigmoid = offset_sigmoid
        self.cls_tokenize = cls_tokenize
        self.joint_cls = joint_cls
        self.softmax_one = softmax_one
        self.contrastive_conv = contrastive_conv
        self.erc_enabled = erc_enabled
        self.erc_apply_vl_tracking_only = erc_apply_vl_tracking_only
        self.erc_match_test_hann = erc_match_test_hann
        self.erc_design = erc_design.lower()

        self.conv_cls = nn.Sequential(conv(inplanes, channel), 
                                        conv(channel, channel // 2),
                                        conv(channel // 2, channel // 4),
                                        conv(channel // 4, channel // 8),
                                        nn.Conv2d(channel // 8, 1, kernel_size=1))

        self.conv_offset = nn.Sequential(conv(inplanes, channel), 
                                        conv(channel, channel // 2),
                                        conv(channel // 2, channel // 4),
                                        conv(channel // 4, channel // 8),
                                        nn.Conv2d(channel // 8, 2, kernel_size=1))

        self.conv_bbox = nn.Sequential(conv(inplanes, channel), 
                                        conv(channel, channel // 2),
                                        conv(channel // 2, channel // 4),
                                        conv(channel // 4, channel // 8),
                                        nn.Conv2d(channel // 8, 2, kernel_size=1))

        self.conv_bbox_grounding = nn.Sequential(conv(inplanes, channel), 
                                        conv(channel, channel // 2),
                                        conv(channel // 2, channel // 4),
                                        conv(channel // 4, channel // 8),
                                        nn.Conv2d(channel // 8, 2, kernel_size=1))

        self.prompter = DistributionBasedCrossAttention(inplanes, drop_rate=drop_rate)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if not erc_enabled:
            self.erc = None
        elif self.erc_design == ForegroundReferenceERC.DESIGN:
            self.erc = ForegroundReferenceERC(
                inplanes, erc_feature_hidden_dim, erc_feature_max_delta, feat_sz,
                erc_force_off, erc_inference_mode)
        elif self.erc_design == ReferenceReliabilityFeatureERC.DESIGN:
            if not erc_apply_vl_tracking_only:
                raise ValueError('Feature ERC must be limited to flag=2 VL tracking')
            self.erc = ReferenceReliabilityFeatureERC(
                dim=inplanes,
                hidden=erc_feature_hidden_dim,
                max_delta=erc_feature_max_delta,
                reliability_temperature=erc_feature_reliability_temperature,
                response_temperature=erc_feature_response_temperature,
                inference_mode=erc_inference_mode,
                force_off=erc_force_off,
            )
        elif self.erc_design == ReferenceReliabilityFeatureERCV2.DESIGN:
            if not erc_apply_vl_tracking_only:
                raise ValueError('Feature ERC must be limited to flag=2 VL tracking')
            self.erc = ReferenceReliabilityFeatureERCV2(
                dim=inplanes,
                hidden=erc_feature_hidden_dim,
                max_delta=erc_feature_max_delta,
                reliability_temperature=erc_feature_reliability_temperature,
                response_temperature=erc_feature_response_temperature,
                inference_mode=erc_inference_mode,
                force_off=erc_force_off,
            )
        elif self.erc_design == ReferenceReliabilityDenseResponseERCV3.DESIGN:
            if not erc_apply_vl_tracking_only or not erc_match_test_hann:
                raise ValueError(
                    'FER-v3 requires VL-only and Hann-aware calibration')
            self.erc = ReferenceReliabilityDenseResponseERCV3(
                feat_sz=feat_sz,
                dim=inplanes,
                hidden=erc_feature_hidden_dim,
                max_alpha=erc_max_alpha,
                d_max=erc_dense_d_max,
                eps=erc_eps,
                reliability_temperature=erc_feature_reliability_temperature,
                response_temperature=erc_feature_response_temperature,
                detach_features=erc_detach_features,
                apply_vl_tracking_only=erc_apply_vl_tracking_only,
                inference_mode=erc_inference_mode,
                force_off=erc_force_off,
            )
        elif self.erc_design == BranchAlignedDenseResponseERCV4.DESIGN:
            if not erc_apply_vl_tracking_only or not erc_match_test_hann:
                raise ValueError(
                    'FER-v4 requires VL-only and Hann-aware calibration')
            self.erc = BranchAlignedDenseResponseERCV4(
                feat_sz=feat_sz,
                dim=inplanes,
                hidden=erc_feature_hidden_dim,
                max_alpha=erc_max_alpha,
                d_max=erc_dense_d_max,
                eps=erc_eps,
                reliability_temperature=erc_feature_reliability_temperature,
                response_temperature=erc_feature_response_temperature,
                spatial_prior=erc_dense_spatial_prior,
                detach_features=erc_detach_features,
                apply_vl_tracking_only=erc_apply_vl_tracking_only,
                inference_mode=erc_inference_mode,
                force_off=erc_force_off,
            )
        elif self.erc_design == BranchAlignedTriStateMultiActionERCV5.DESIGN:
            if not erc_apply_vl_tracking_only or not erc_match_test_hann:
                raise ValueError(
                    'FER-v5 requires VL-only and Hann-aware calibration')
            if not softmax_one:
                raise ValueError('FER-v5 requires SOFTMAX_ONE for deploy-aligned T')
            self.erc = BranchAlignedTriStateMultiActionERCV5(
                feat_sz=feat_sz,
                dim=inplanes,
                hidden=erc_feature_hidden_dim,
                action_alpha_values=erc_action_alpha_values,
                max_alpha=erc_max_alpha,
                d_max=erc_dense_d_max,
                eps=erc_eps,
                action_temperature=erc_action_temperature,
                response_temperature=erc_feature_response_temperature,
                detach_features=erc_detach_features,
                apply_vl_tracking_only=erc_apply_vl_tracking_only,
                inference_mode=erc_inference_mode,
                force_off=erc_force_off,
            )
        elif self.erc_design in (
            BranchAlignedFactorizedReachabilityERCV6.DESIGN,
            BranchAlignedIoUUtilityERCV7.DESIGN,
            BranchAlignedActionValueERCAV1.DESIGN,
            BranchAlignedActionValueERCAV1B.DESIGN,
            BranchAlignedTemporalConsensusERCAV2.DESIGN,
            CrossDomainGlobalRegionalERCV1.DESIGN,
            CrossModalConsensusResidualERCAV3.DESIGN,
            CrossDomainNormalizedConsensusERCAV4.DESIGN,
            CrossDomainContinuousOracleERCAV5.DESIGN,
            PeakQualityCalibratedERCPQ1.DESIGN,
            PeakQualityDecisionAlignedERCPQ2.DESIGN,
            PeakQualityIndependentBranchERCPQ3.DESIGN,
            PeakAmbiguityTargetVerifierERCPQ4.DESIGN,
            TargetReferenceDirectDenseERCDR1.DESIGN,
            TargetReferenceTrajectorySafeDenseERCDR2.DESIGN,
            TargetReferenceRealizedActionDenseERCDR3.DESIGN,
        ):
            if not erc_apply_vl_tracking_only or not erc_match_test_hann:
                raise ValueError(
                    'FER-v6/v7 and ERC-AV1 require VL-only, Hann-aware calibration')
            if not softmax_one:
                raise ValueError(
                    'FER-v6/v7 and ERC-AV1 require SOFTMAX_ONE for deploy-aligned T')
            erc_class = {
                BranchAlignedFactorizedReachabilityERCV6.DESIGN:
                    BranchAlignedFactorizedReachabilityERCV6,
                BranchAlignedIoUUtilityERCV7.DESIGN:
                    BranchAlignedIoUUtilityERCV7,
                BranchAlignedActionValueERCAV1.DESIGN:
                    BranchAlignedActionValueERCAV1,
                BranchAlignedActionValueERCAV1B.DESIGN:
                    BranchAlignedActionValueERCAV1B,
                BranchAlignedTemporalConsensusERCAV2.DESIGN:
                    BranchAlignedTemporalConsensusERCAV2,
                CrossDomainGlobalRegionalERCV1.DESIGN:
                    CrossDomainGlobalRegionalERCV1,
                CrossModalConsensusResidualERCAV3.DESIGN:
                    CrossModalConsensusResidualERCAV3,
                CrossDomainNormalizedConsensusERCAV4.DESIGN:
                    CrossDomainNormalizedConsensusERCAV4,
                CrossDomainContinuousOracleERCAV5.DESIGN:
                    CrossDomainContinuousOracleERCAV5,
                PeakQualityCalibratedERCPQ1.DESIGN:
                    PeakQualityCalibratedERCPQ1,
                PeakQualityDecisionAlignedERCPQ2.DESIGN:
                    PeakQualityDecisionAlignedERCPQ2,
                PeakQualityIndependentBranchERCPQ3.DESIGN:
                    PeakQualityIndependentBranchERCPQ3,
                PeakAmbiguityTargetVerifierERCPQ4.DESIGN:
                    PeakAmbiguityTargetVerifierERCPQ4,
                TargetReferenceDirectDenseERCDR1.DESIGN:
                    TargetReferenceDirectDenseERCDR1,
                TargetReferenceTrajectorySafeDenseERCDR2.DESIGN:
                    TargetReferenceTrajectorySafeDenseERCDR2,
                TargetReferenceRealizedActionDenseERCDR3.DESIGN:
                    TargetReferenceRealizedActionDenseERCDR3,
            }[self.erc_design]
            erc_kwargs = {}
            if erc_class is PeakAmbiguityTargetVerifierERCPQ4:
                erc_kwargs['response_topk'] = erc_response_topk
            if erc_class in (
                    BranchAlignedActionValueERCAV1,
                    BranchAlignedActionValueERCAV1B,
                    BranchAlignedTemporalConsensusERCAV2,
                    CrossModalConsensusResidualERCAV3,
                    CrossDomainNormalizedConsensusERCAV4):
                erc_kwargs.update(
                    utility_temperature=erc_av1_utility_temperature,
                    utility_gate_scale=erc_av1_gate_scale,
                )
            if erc_class in (
                    BranchAlignedTemporalConsensusERCAV2,
                    CrossModalConsensusResidualERCAV3,
                    CrossDomainNormalizedConsensusERCAV4):
                erc_kwargs.update(
                    temporal_ema=erc_av2_temporal_ema,
                    release_margin=erc_av2_release_margin,
                    enable_temporal_memory=erc_av2_enable_memory,
                )
            if erc_class is CrossDomainGlobalRegionalERCV1:
                erc_kwargs.update(
                    temporal_ema=erc_av2_temporal_ema,
                    enable_temporal_memory=erc_av2_enable_memory,
                    direction_scale=erc_direction_scale,
                )
            if erc_class is CrossDomainContinuousOracleERCAV5:
                erc_kwargs.update(
                    temporal_ema=erc_av2_temporal_ema,
                    enable_temporal_memory=erc_av2_enable_memory,
                )
            if erc_class in (
                    TargetReferenceTrajectorySafeDenseERCDR2,
                    TargetReferenceRealizedActionDenseERCDR3):
                erc_kwargs.update(
                    safety_gate_prior=erc_dr2_safety_gate_prior,
                    temporal_ema=erc_dr2_temporal_ema,
                    enable_temporal_memory=erc_dr2_enable_memory,
                )
            self.erc = erc_class(
                feat_sz=feat_sz, dim=inplanes, hidden=erc_feature_hidden_dim,
                magnitudes=erc_reachability_magnitudes,
                max_alpha=erc_max_alpha, d_max=erc_dense_d_max,
                eps=erc_eps, action_temperature=erc_action_temperature,
                response_temperature=erc_feature_response_temperature,
                detach_features=erc_detach_features,
                apply_vl_tracking_only=erc_apply_vl_tracking_only,
                inference_mode=erc_inference_mode, force_off=erc_force_off,
                **erc_kwargs,
            )
        elif self.erc_design in (
            TargetReferenceERC.DESIGN, TargetReferenceERCV2.DESIGN,
            TargetReferenceERCV3.DESIGN, TargetReferenceERCV4.DESIGN,
            TargetReferenceERCV4B.DESIGN, TargetReferenceERCV5.DESIGN,
            TargetReferenceERCV5B.DESIGN,
            TargetReferenceERCV5C.DESIGN,
            TargetReferenceERCV5D.DESIGN,
            TargetReferenceERCV5F.DESIGN,
            TargetReferenceERCV5G.DESIGN,
            TargetReferenceERCV5H.DESIGN,
            TargetReferenceERCV6.DESIGN,
            TargetReferenceERCV6B.DESIGN,
        ):
            if not softmax_one or not erc_apply_vl_tracking_only or not erc_match_test_hann:
                raise ValueError('TRC requires SOFTMAX_ONE, VL-only and Hann-aware calibration')
            erc_class = {
                TargetReferenceERC.DESIGN: TargetReferenceERC,
                TargetReferenceERCV2.DESIGN: TargetReferenceERCV2,
                TargetReferenceERCV3.DESIGN: TargetReferenceERCV3,
                TargetReferenceERCV4.DESIGN: TargetReferenceERCV4,
                TargetReferenceERCV4B.DESIGN: TargetReferenceERCV4B,
                TargetReferenceERCV5.DESIGN: TargetReferenceERCV5,
                TargetReferenceERCV5B.DESIGN: TargetReferenceERCV5B,
                TargetReferenceERCV5C.DESIGN: TargetReferenceERCV5C,
                TargetReferenceERCV5D.DESIGN: TargetReferenceERCV5D,
                TargetReferenceERCV5F.DESIGN: TargetReferenceERCV5F,
                TargetReferenceERCV5G.DESIGN: TargetReferenceERCV5G,
                TargetReferenceERCV5H.DESIGN: TargetReferenceERCV5H,
                TargetReferenceERCV6.DESIGN: TargetReferenceERCV6,
                TargetReferenceERCV6B.DESIGN: TargetReferenceERCV6B,
            }[self.erc_design]
            erc_kwargs = {}
            if erc_class in (
                TargetReferenceERCV5B, TargetReferenceERCV5C,
                TargetReferenceERCV5D, TargetReferenceERCV5F,
                TargetReferenceERCV5G, TargetReferenceERCV5H,
            ):
                erc_kwargs.update({
                    'permission_threshold': erc_trc_permission_threshold,
                    'permission_initial_probability':
                        erc_trc_permission_initial_probability,
                })
            if erc_class in (
                TargetReferenceERCV5F, TargetReferenceERCV5G,
                TargetReferenceERCV5H,
            ):
                erc_kwargs['direction_t_threshold'] = (
                    erc_trc_direction_t_threshold
                )
            if erc_class in (TargetReferenceERCV5G, TargetReferenceERCV5H):
                erc_kwargs.update({
                    'branch_permission_threshold':
                        erc_trc_branch_permission_threshold,
                    'branch_permission_initial_probability':
                        erc_trc_branch_permission_initial_probability,
                })
            if erc_class in (TargetReferenceERCV6, TargetReferenceERCV6B):
                erc_kwargs['utility_threshold'] = erc_trc_utility_threshold
            self.erc = erc_class(
                feat_sz, erc_trc_hidden_dim, erc_max_alpha, erc_trc_temperature,
                erc_trc_d_max, erc_inference_mode, erc_force_off,
                **erc_kwargs)
        elif self.erc_design == ResponseGlobalRegionalERCV7.DESIGN:
            self.erc = ResponseGlobalRegionalERCV7(
                feat_sz=feat_sz,
                hidden_dim=erc_hidden_dim,
                regional_hidden_dim=erc_regional_hidden_dim,
                drop_rate=erc_drop_rate,
                max_alpha=erc_max_alpha,
                global_alpha_scale=erc_global_alpha_scale,
                region_alpha_scale=erc_region_alpha_scale,
                eps=erc_eps,
                log_ratio_clip=erc_log_ratio_clip,
                detach_features=erc_detach_features,
                apply_vl_tracking_only=erc_apply_vl_tracking_only,
                gate_prior=erc_gate_prior,
                detach_reliability_for_alpha=(
                    erc_detach_reliability_for_alpha
                ),
                force_off=erc_force_off,
                inference_mode=erc_inference_mode,
                response_topk=erc_response_topk,
            )
        elif self.erc_design == ResponseHannHardPeakERCV8.DESIGN:
            self.erc = ResponseHannHardPeakERCV8(
                feat_sz=feat_sz,
                hidden_dim=erc_hidden_dim,
                regional_hidden_dim=erc_regional_hidden_dim,
                drop_rate=erc_drop_rate,
                max_alpha=erc_max_alpha,
                global_alpha_scale=erc_global_alpha_scale,
                region_alpha_scale=erc_region_alpha_scale,
                eps=erc_eps,
                log_ratio_clip=erc_log_ratio_clip,
                detach_features=erc_detach_features,
                apply_vl_tracking_only=erc_apply_vl_tracking_only,
                gate_prior=erc_gate_prior,
                detach_reliability_for_alpha=(
                    erc_detach_reliability_for_alpha
                ),
                force_off=erc_force_off,
                inference_mode=erc_inference_mode,
                response_topk=erc_response_topk,
            )
        elif self.erc_design == ResponseHannGateERCV8C.DESIGN:
            self.erc = ResponseHannGateERCV8C(
                feat_sz=feat_sz,
                hidden_dim=erc_hidden_dim,
                regional_hidden_dim=erc_regional_hidden_dim,
                drop_rate=erc_drop_rate,
                max_alpha=erc_max_alpha,
                global_alpha_scale=erc_global_alpha_scale,
                region_alpha_scale=erc_region_alpha_scale,
                eps=erc_eps,
                log_ratio_clip=erc_log_ratio_clip,
                detach_features=erc_detach_features,
                apply_vl_tracking_only=erc_apply_vl_tracking_only,
                gate_prior=erc_gate_prior,
                detach_reliability_for_alpha=(
                    erc_detach_reliability_for_alpha
                ),
                force_off=erc_force_off,
                inference_mode=erc_inference_mode,
                response_topk=erc_response_topk,
            )
        elif self.erc_design == ResponseRegionalEffectiveERCV8D.DESIGN:
            self.erc = ResponseRegionalEffectiveERCV8D(
                feat_sz=feat_sz,
                hidden_dim=erc_hidden_dim,
                regional_hidden_dim=erc_regional_hidden_dim,
                drop_rate=erc_drop_rate,
                max_alpha=erc_max_alpha,
                global_alpha_scale=erc_global_alpha_scale,
                region_alpha_scale=erc_region_alpha_scale,
                eps=erc_eps,
                log_ratio_clip=erc_log_ratio_clip,
                detach_features=erc_detach_features,
                apply_vl_tracking_only=erc_apply_vl_tracking_only,
                gate_prior=erc_gate_prior,
                detach_reliability_for_alpha=(
                    erc_detach_reliability_for_alpha
                ),
                force_off=erc_force_off,
                inference_mode=erc_inference_mode,
                response_topk=erc_response_topk,
            )
        elif self.erc_design == ResponsePrecisionIncrementalERCV9.DESIGN:
            self.erc = ResponsePrecisionIncrementalERCV9(
                feat_sz=feat_sz,
                hidden_dim=erc_hidden_dim,
                regional_hidden_dim=erc_regional_hidden_dim,
                drop_rate=erc_drop_rate,
                max_alpha=erc_max_alpha,
                global_alpha_scale=erc_global_alpha_scale,
                region_alpha_scale=erc_region_alpha_scale,
                eps=erc_eps,
                log_ratio_clip=erc_log_ratio_clip,
                detach_features=erc_detach_features,
                apply_vl_tracking_only=erc_apply_vl_tracking_only,
                gate_prior=erc_gate_prior,
                regional_gate_prior=erc_regional_gate_prior,
                detach_reliability_for_alpha=(
                    erc_detach_reliability_for_alpha
                ),
                force_off=erc_force_off,
                inference_mode=erc_inference_mode,
                response_topk=erc_response_topk,
            )
        elif self.erc_design == ResponseCompetitionPairERCV10.DESIGN:
            self.erc = ResponseCompetitionPairERCV10(
                feat_sz=feat_sz,
                hidden_dim=erc_hidden_dim,
                regional_hidden_dim=erc_regional_hidden_dim,
                drop_rate=erc_drop_rate,
                max_alpha=erc_max_alpha,
                global_alpha_scale=erc_global_alpha_scale,
                region_alpha_scale=erc_region_alpha_scale,
                eps=erc_eps,
                log_ratio_clip=erc_log_ratio_clip,
                detach_features=erc_detach_features,
                apply_vl_tracking_only=erc_apply_vl_tracking_only,
                gate_prior=erc_gate_prior,
                regional_gate_prior=erc_regional_gate_prior,
                detach_reliability_for_alpha=(
                    erc_detach_reliability_for_alpha
                ),
                force_off=erc_force_off,
                inference_mode=erc_inference_mode,
                response_topk=erc_response_topk,
                pair_beta_scale=erc_pair_beta_scale,
                pair_exclusion_radius=erc_pair_exclusion_radius,
                pair_mask_sigma=erc_pair_mask_sigma,
            )
        elif self.erc_design == ResponseLocalSpatialERCV11.DESIGN:
            self.erc = ResponseLocalSpatialERCV11(
                feat_sz=feat_sz,
                hidden_dim=erc_hidden_dim,
                local_hidden_dim=erc_local_hidden_dim,
                local_window=erc_local_window,
                delta_max=erc_local_delta_max,
                drop_rate=erc_drop_rate,
                eps=erc_eps,
                log_ratio_clip=erc_log_ratio_clip,
                detach_features=erc_detach_features,
                apply_vl_tracking_only=erc_apply_vl_tracking_only,
                force_off=erc_force_off,
                inference_mode=erc_inference_mode,
            )
        elif self.erc_design == ResponseLocalReliabilityERCV12.DESIGN:
            self.erc = ResponseLocalReliabilityERCV12(
                feat_sz=feat_sz,
                hidden_dim=erc_hidden_dim,
                local_hidden_dim=erc_local_hidden_dim,
                local_window=erc_local_window,
                delta_max=erc_local_delta_max,
                drop_rate=erc_drop_rate,
                eps=erc_eps,
                log_ratio_clip=erc_log_ratio_clip,
                detach_features=erc_detach_features,
                apply_vl_tracking_only=erc_apply_vl_tracking_only,
                force_off=erc_force_off,
                inference_mode=erc_inference_mode,
                reliability_prior=erc_local_reliability_prior,
            )
        elif self.erc_design == ResponseBranchSpatialGateERCV13.DESIGN:
            self.erc = ResponseBranchSpatialGateERCV13(
                feat_sz=feat_sz,
                hidden_dim=erc_hidden_dim,
                spatial_hidden_dim=erc_local_hidden_dim,
                drop_rate=erc_drop_rate,
                beta_max=erc_max_alpha,
                d_max=erc_v13_d_max,
                sigmas=erc_v13_sigmas,
                eps=erc_eps,
                log_c_min=erc_v13_log_c_min,
                log_t_min=erc_v13_log_t_min,
                log_s0_min=erc_v13_log_s0_min,
                detach_features=erc_detach_features,
                apply_vl_tracking_only=erc_apply_vl_tracking_only,
                force_off=erc_force_off,
                inference_mode=erc_inference_mode,
            )
        elif self.erc_design == ResponseInterventionGateERCV14.DESIGN:
            self.erc = ResponseInterventionGateERCV14(
                feat_sz=feat_sz,
                hidden_dim=erc_hidden_dim,
                spatial_hidden_dim=erc_local_hidden_dim,
                drop_rate=erc_drop_rate,
                beta_max=erc_max_alpha,
                d_max=erc_v13_d_max,
                sigmas=erc_v13_sigmas,
                eps=erc_eps,
                log_c_min=erc_v13_log_c_min,
                log_t_min=erc_v13_log_t_min,
                log_s0_min=erc_v13_log_s0_min,
                detach_features=erc_detach_features,
                apply_vl_tracking_only=erc_apply_vl_tracking_only,
                force_off=erc_force_off,
                inference_mode=erc_inference_mode,
                intervention_prior=erc_v14_gate_prior,
            )
        elif self.erc_design == ResponseWorthinessGateERCV15.DESIGN:
            self.erc = ResponseWorthinessGateERCV15(
                feat_sz=feat_sz,
                hidden_dim=erc_hidden_dim,
                spatial_hidden_dim=erc_local_hidden_dim,
                drop_rate=erc_drop_rate,
                beta_max=erc_max_alpha,
                d_max=erc_v13_d_max,
                sigmas=erc_v13_sigmas,
                eps=erc_eps,
                log_c_min=erc_v13_log_c_min,
                log_t_min=erc_v13_log_t_min,
                log_s0_min=erc_v13_log_s0_min,
                detach_features=erc_detach_features,
                apply_vl_tracking_only=erc_apply_vl_tracking_only,
                force_off=erc_force_off,
                inference_mode=erc_inference_mode,
                intervention_prior=erc_v14_gate_prior,
            )
        elif self.erc_design == ResponseAsymmetricWorthinessGateERCV16.DESIGN:
            self.erc = ResponseAsymmetricWorthinessGateERCV16(
                feat_sz=feat_sz,
                hidden_dim=erc_hidden_dim,
                spatial_hidden_dim=erc_local_hidden_dim,
                drop_rate=erc_drop_rate,
                beta_max=erc_max_alpha,
                d_max=erc_v13_d_max,
                sigmas=erc_v13_sigmas,
                eps=erc_eps,
                log_c_min=erc_v13_log_c_min,
                log_t_min=erc_v13_log_t_min,
                log_s0_min=erc_v13_log_s0_min,
                detach_features=erc_detach_features,
                apply_vl_tracking_only=erc_apply_vl_tracking_only,
                force_off=erc_force_off,
                inference_mode=erc_inference_mode,
                intervention_prior=erc_v14_gate_prior,
            )
        elif self.erc_design == ResponseSelectiveAbstentionWorthinessGateERCV17.DESIGN:
            self.erc = ResponseSelectiveAbstentionWorthinessGateERCV17(
                feat_sz=feat_sz,
                hidden_dim=erc_hidden_dim,
                spatial_hidden_dim=erc_local_hidden_dim,
                drop_rate=erc_drop_rate,
                beta_max=erc_max_alpha,
                d_max=erc_v13_d_max,
                sigmas=erc_v13_sigmas,
                eps=erc_eps,
                log_c_min=erc_v13_log_c_min,
                log_t_min=erc_v13_log_t_min,
                log_s0_min=erc_v13_log_s0_min,
                detach_features=erc_detach_features,
                apply_vl_tracking_only=erc_apply_vl_tracking_only,
                force_off=erc_force_off,
                inference_mode=erc_inference_mode,
                intervention_prior=erc_v14_gate_prior,
            )
        elif self.erc_design == ResponsePreservationVetoGateERCV18.DESIGN:
            self.erc = ResponsePreservationVetoGateERCV18(
                feat_sz=feat_sz,
                hidden_dim=erc_hidden_dim,
                spatial_hidden_dim=erc_local_hidden_dim,
                drop_rate=erc_drop_rate,
                beta_max=erc_max_alpha,
                d_max=erc_v13_d_max,
                sigmas=erc_v13_sigmas,
                eps=erc_eps,
                log_c_min=erc_v13_log_c_min,
                log_t_min=erc_v13_log_t_min,
                log_s0_min=erc_v13_log_s0_min,
                detach_features=erc_detach_features,
                apply_vl_tracking_only=erc_apply_vl_tracking_only,
                force_off=erc_force_off,
                inference_mode=erc_inference_mode,
                intervention_prior=erc_v14_gate_prior,
            )
        else:
            # Preserve the complete v1-v6 implementation and state-dict layout.
            self.erc = EvidenceReliabilityCalibration(
                feat_sz=feat_sz,
                hidden_dim=erc_hidden_dim,
                drop_rate=erc_drop_rate,
                max_alpha=erc_max_alpha,
                eps=erc_eps,
                log_ratio_clip=erc_log_ratio_clip,
                detach_features=erc_detach_features,
                reliability_gated_alpha=erc_reliability_gated_alpha,
                detach_reliability_gate=erc_detach_reliability_gate,
                reliability_gate_floor=erc_reliability_gate_floor,
                apply_vl_tracking_only=erc_apply_vl_tracking_only,
                design=erc_design,
                direction_scale=erc_direction_scale,
                delta_scale=erc_delta_scale,
                gate_prior=erc_gate_prior,
                detach_reliability_for_alpha=(
                    erc_detach_reliability_for_alpha
                ),
                force_off=erc_force_off,
            )
        x, y = torch.arange(0, self.feat_sz), torch.arange(0, self.feat_sz)
        x, y = torch.meshgrid(x, y)
        if offset_sigmoid:
            coodinate = torch.cat([y.reshape(-1)[None, :], x.reshape(-1)[None, :]])[None] # b, 2, s
        else:
            coodinate = torch.cat([y.reshape(-1)[None, :], x.reshape(-1)[None, :]])[None] + 0.5
        self.register_buffer("coodinate", coodinate)

    def forward(self, out_dict, test=False):
        if self.erc is not None and self.erc_design == ForegroundReferenceERC.DESIGN:
            return forward_fr1(self, out_dict)
        # top-left branch
        flag = out_dict['flag']
        vis_token = out_dict['vis_token']
        txt_token = out_dict['txt_token']
        feature_erc_output = None
        head_input = out_dict
        if self.erc is not None and self.erc_design in (
            ReferenceReliabilityFeatureERC.DESIGN,
            ReferenceReliabilityFeatureERCV2.DESIGN,
        ):
            if self.erc_design == ReferenceReliabilityFeatureERC.DESIGN:
                feature_erc_output = self.erc(
                    out_dict['search'], out_dict['template'],
                    out_dict.get('template_mask'), txt_token, flag,
                )
            else:
                feature_erc_output = self.erc(
                    out_dict['search'], vis_token, txt_token, flag,
                )
            head_input = dict(out_dict)
            head_input['search'] = feature_erc_output['adapted_search']
        token_group = torch.cat([vis_token, txt_token, (vis_token+txt_token)/2], dim=1)
        bid = torch.arange(flag.shape[0])
        token = token_group[bid, flag, ..., None, None]
        
        cont_score, prompts = self.contractive_learning(head_input)
        b = head_input['search'].shape[0]
        x = head_input['search'].transpose(-2, -1).reshape(b, -1, self.feat_sz, self.feat_sz).contiguous()
        cls_map = self.conv_cls(x*token).sigmoid().squeeze(1) if self.cls_tokenize else self.conv_cls(x).sigmoid().squeeze(1)
        offset_map = self.conv_offset(x).sigmoid() if self.offset_sigmoid else self.conv_offset(x)
        size_map_tr = self.conv_bbox(x).sigmoid().unsqueeze(1)
        size_map_gr = self.conv_bbox_grounding(x).sigmoid().unsqueeze(1)
        
        flag = out_dict['flag']
        size_map_group = torch.cat([size_map_tr, size_map_gr, size_map_tr], dim=1)
        bid = torch.arange(flag.shape[0])
        size_map = size_map_group[bid, flag]
        
        cont_score_2d = cont_score.softmax(-1)[..., 0].reshape(-1, *cls_map.shape[1:])
        # With SOFTMAX_ONE, deployed tracking includes an additional fixed-zero
        # logit.  Keep the native 2-logit training tensor/loss unchanged, but
        # feed FER-v5 the exact test-time probability.
        erc_cont_score_2d = cont_score_2d
        if self.erc_design in (
                BranchAlignedTriStateMultiActionERCV5.DESIGN,
                BranchAlignedFactorizedReachabilityERCV6.DESIGN,
                BranchAlignedIoUUtilityERCV7.DESIGN,
                BranchAlignedActionValueERCAV1.DESIGN,
                BranchAlignedActionValueERCAV1B.DESIGN,
                BranchAlignedTemporalConsensusERCAV2.DESIGN,
                CrossDomainGlobalRegionalERCV1.DESIGN,
                CrossModalConsensusResidualERCAV3.DESIGN,
                CrossDomainNormalizedConsensusERCAV4.DESIGN,
                CrossDomainContinuousOracleERCAV5.DESIGN,
                PeakQualityCalibratedERCPQ1.DESIGN,
                PeakQualityDecisionAlignedERCPQ2.DESIGN,
                PeakQualityIndependentBranchERCPQ3.DESIGN,
                PeakAmbiguityTargetVerifierERCPQ4.DESIGN,
                TargetReferenceDirectDenseERCDR1.DESIGN,
                TargetReferenceTrajectorySafeDenseERCDR2.DESIGN,
                TargetReferenceRealizedActionDenseERCDR3.DESIGN):
            deploy_logits = cont_score
            if deploy_logits.shape[-1] == 2:
                deploy_logits = torch.cat([
                    deploy_logits,
                    torch.zeros_like(deploy_logits[..., :1]),
                ], dim=-1)
            elif deploy_logits.shape[-1] != 3:
                raise ValueError(
                    'FER response calibration expects 2 or 3 contrastive logits')
            erc_cont_score_2d = deploy_logits.softmax(-1)[..., 0].reshape(
                -1, *cls_map.shape[1:])
        base_fusion_score = cls_map.flatten(1) * cont_score_2d.flatten(1)
        if feature_erc_output is not None:
            batch_size, num_candidates = base_fusion_score.shape
            zeros = base_fusion_score.new_zeros(batch_size, 1)
            ones = base_fusion_score.new_ones(batch_size, num_candidates)
            reliability = feature_erc_output['reliability']
            active = feature_erc_output['active']
            erc_output = {
                'features': feature_erc_output['feature_delta'].transpose(
                    1, 2).reshape(batch_size, -1, self.feat_sz, self.feat_sz),
                'cls_response': cls_map.flatten(1),
                'cont_response': cont_score_2d.flatten(1),
                'windowed_cls_score': cls_map.flatten(1),
                'windowed_cont_score': cont_score_2d.flatten(1),
                'alpha': zeros,
                'raw_alpha': zeros,
                'z_alpha': reliability[:, :1] - reliability[:, 1:],
                'reliability': reliability.max(-1, keepdim=True).values,
                'reliability_gate': active[:, None].to(base_fusion_score),
                'reliability_cls': reliability[:, :1],
                'reliability_cont': reliability[:, 1:],
                'reliability_base': 1.0 - reliability.max(
                    -1, keepdim=True).values,
                'intervention_gate': active[:, None].to(base_fusion_score),
                'delta': feature_erc_output['feature_delta'].float().norm(
                    dim=-1).mean(-1, keepdim=True).to(base_fusion_score),
                'log_ratio': base_fusion_score.new_zeros(
                    batch_size, num_candidates),
                'factor': ones,
                'base_score': base_fusion_score,
                'windowed_base_score': base_fusion_score,
                'calibrated_score': base_fusion_score,
                'windowed_calibrated_score': base_fusion_score,
                'global_calibrated_score': base_fusion_score,
                'windowed_global_calibrated_score': base_fusion_score,
                'active_mask': active,
            }
            selection_score = base_fusion_score
        elif self.erc is not None:
            if self.erc_design in (
                    TargetReferenceDirectDenseERCDR1.DESIGN,
                    TargetReferenceTrajectorySafeDenseERCDR2.DESIGN,
                    TargetReferenceRealizedActionDenseERCDR3.DESIGN):
                erc_output = self.erc(
                    cls_map, erc_cont_score_2d,
                    search=out_dict['search'],
                    visual_reference=vis_token,
                    language_reference=txt_token,
                    prompts=prompts,
                    flag=flag,
                )
            elif self.erc_design in (
                ReferenceReliabilityDenseResponseERCV3.DESIGN,
                BranchAlignedDenseResponseERCV4.DESIGN,
                BranchAlignedTriStateMultiActionERCV5.DESIGN,
                BranchAlignedFactorizedReachabilityERCV6.DESIGN,
                BranchAlignedIoUUtilityERCV7.DESIGN,
                BranchAlignedActionValueERCAV1.DESIGN,
                BranchAlignedActionValueERCAV1B.DESIGN,
                BranchAlignedTemporalConsensusERCAV2.DESIGN,
                CrossDomainGlobalRegionalERCV1.DESIGN,
                CrossModalConsensusResidualERCAV3.DESIGN,
                CrossDomainNormalizedConsensusERCAV4.DESIGN,
                CrossDomainContinuousOracleERCAV5.DESIGN,
                PeakQualityCalibratedERCPQ1.DESIGN,
                PeakQualityDecisionAlignedERCPQ2.DESIGN,
                PeakQualityIndependentBranchERCPQ3.DESIGN,
                PeakAmbiguityTargetVerifierERCPQ4.DESIGN,
            ):
                erc_output = self.erc(
                    cls_map, erc_cont_score_2d,
                    search=out_dict['search'],
                    visual_reference=vis_token,
                    language_reference=txt_token,
                    flag=flag,
                )
            elif self.erc_design in (
                TargetReferenceERC.DESIGN, TargetReferenceERCV2.DESIGN,
                TargetReferenceERCV3.DESIGN, TargetReferenceERCV4.DESIGN,
                TargetReferenceERCV4B.DESIGN, TargetReferenceERCV5.DESIGN,
                TargetReferenceERCV5B.DESIGN,
                TargetReferenceERCV5C.DESIGN,
                TargetReferenceERCV5D.DESIGN,
                TargetReferenceERCV5F.DESIGN,
                TargetReferenceERCV5G.DESIGN,
                TargetReferenceERCV5H.DESIGN,
                TargetReferenceERCV6.DESIGN,
                TargetReferenceERCV6B.DESIGN,
            ):
                erc_output = self.erc(
                    cls_map, cont_score_2d, search=out_dict['search'],
                    prompts=prompts, template=out_dict['template'],
                    template_mask=out_dict['template_mask'],
                    logit_scale=self.logit_scale, flag=flag)
            else:
                erc_output = self.erc(cls_map, cont_score_2d, flag=flag)
            erc_selection_score = (
                erc_output['windowed_calibrated_score']
                if self.erc_match_test_hann
                else erc_output['calibrated_score']
            )
            if self.erc_apply_vl_tracking_only:
                selection_score = torch.where(
                    erc_output['active_mask'][:, None],
                    erc_selection_score,
                    erc_output['base_score'],
                )
            else:
                selection_score = erc_selection_score
        else:
            batch_size, num_candidates = base_fusion_score.shape
            erc_output = {
                'features': None,
                'cls_response': base_fusion_score,
                'cont_response': base_fusion_score.new_ones(
                    batch_size, num_candidates
                ),
                'windowed_cls_score': base_fusion_score,
                'windowed_cont_score': base_fusion_score,
                'alpha': base_fusion_score.new_zeros(batch_size, 1),
                'raw_alpha': base_fusion_score.new_zeros(batch_size, 1),
                'z_alpha': base_fusion_score.new_zeros(batch_size, 1),
                'reliability': base_fusion_score.new_ones(batch_size, 1),
                'reliability_gate': base_fusion_score.new_zeros(batch_size, 1),
                'reliability_cls': base_fusion_score.new_ones(batch_size, 1),
                'reliability_cont': base_fusion_score.new_ones(batch_size, 1),
                'reliability_base': base_fusion_score.new_ones(batch_size, 1),
                'intervention_gate': base_fusion_score.new_zeros(batch_size, 1),
                'gate_logit': base_fusion_score.new_zeros(batch_size, 1),
                'regional_intervention_gate': base_fusion_score.new_zeros(
                    batch_size, 1
                ),
                'regional_gate_logit': base_fusion_score.new_zeros(
                    batch_size, 1
                ),
                'gate_decision_features': base_fusion_score.new_zeros(
                    batch_size, 0
                ),
                'delta': base_fusion_score.new_zeros(batch_size, 1),
                'log_ratio': base_fusion_score.new_zeros(batch_size, num_candidates),
                'factor': base_fusion_score.new_ones(batch_size, num_candidates),
                'base_score': base_fusion_score,
                'windowed_base_score': base_fusion_score,
                'calibrated_score': base_fusion_score,
                'windowed_calibrated_score': base_fusion_score,
                'global_calibrated_score': base_fusion_score,
                'windowed_global_calibrated_score': base_fusion_score,
                'alpha_global': base_fusion_score.new_zeros(batch_size, 1),
                'alpha_region': base_fusion_score.new_zeros(
                    batch_size, self.feat_sz, self.feat_sz
                ),
                'alpha_final': base_fusion_score.new_zeros(
                    batch_size, self.feat_sz, self.feat_sz
                ),
                'active_mask': torch.ones(
                    batch_size, dtype=torch.bool, device=base_fusion_score.device
                ),
            }
            selection_score = base_fusion_score
        bbox_map, selected_bbox = self.convert2bbox(
            cls_map, offset_map, size_map, selection_score
        )
        if self.erc_design in (
            TargetReferenceERC.DESIGN, TargetReferenceERCV2.DESIGN,
            TargetReferenceERCV3.DESIGN, TargetReferenceERCV4.DESIGN,
            TargetReferenceERCV4B.DESIGN, TargetReferenceERCV5.DESIGN,
            TargetReferenceERCV5B.DESIGN,
            TargetReferenceERCV5C.DESIGN,
            TargetReferenceERCV5D.DESIGN,
            TargetReferenceERCV5F.DESIGN,
            TargetReferenceERCV5G.DESIGN,
            TargetReferenceERCV5H.DESIGN,
            TargetReferenceERCV6.DESIGN,
            TargetReferenceERCV6B.DESIGN,
            ReferenceReliabilityFeatureERC.DESIGN,
            ReferenceReliabilityFeatureERCV2.DESIGN,
            ReferenceReliabilityDenseResponseERCV3.DESIGN,
            BranchAlignedDenseResponseERCV4.DESIGN,
            BranchAlignedTriStateMultiActionERCV5.DESIGN,
            BranchAlignedFactorizedReachabilityERCV6.DESIGN,
            BranchAlignedIoUUtilityERCV7.DESIGN,
            BranchAlignedActionValueERCAV1.DESIGN,
            BranchAlignedActionValueERCAV1B.DESIGN,
            BranchAlignedTemporalConsensusERCAV2.DESIGN,
            CrossDomainGlobalRegionalERCV1.DESIGN,
            CrossModalConsensusResidualERCAV3.DESIGN,
            CrossDomainNormalizedConsensusERCAV4.DESIGN,
            CrossDomainContinuousOracleERCAV5.DESIGN,
            PeakQualityCalibratedERCPQ1.DESIGN,
            PeakQualityDecisionAlignedERCPQ2.DESIGN,
            PeakQualityIndependentBranchERCPQ3.DESIGN,
            PeakAmbiguityTargetVerifierERCPQ4.DESIGN,
            TargetReferenceDirectDenseERCDR1.DESIGN,
            TargetReferenceTrajectorySafeDenseERCDR2.DESIGN,
            TargetReferenceRealizedActionDenseERCDR3.DESIGN,
            'free_gate', ResponseGlobalRegionalERCV7.DESIGN,
            ResponseHannHardPeakERCV8.DESIGN,
            ResponseHannGateERCV8C.DESIGN,
            ResponseRegionalEffectiveERCV8D.DESIGN,
            ResponsePrecisionIncrementalERCV9.DESIGN,
            ResponseCompetitionPairERCV10.DESIGN,
            ResponseLocalSpatialERCV11.DESIGN,
            ResponseLocalReliabilityERCV12.DESIGN,
            ResponseBranchSpatialGateERCV13.DESIGN,
            ResponseInterventionGateERCV14.DESIGN,
            ResponseWorthinessGateERCV15.DESIGN,
            ResponseAsymmetricWorthinessGateERCV16.DESIGN,
            ResponseSelectiveAbstentionWorthinessGateERCV17.DESIGN,
            ResponsePreservationVetoGateERCV18.DESIGN,
        ):
            # The official training-time regression path selects from C*T
            # without a Hann window.  Keep that hard selection independent of
            # ERC while exposing ERC's selected box for diagnostics.  Test-time
            # tracking already ranks bbox_map explicitly in the tracker.
            _, baseline_bbox = self.convert2bbox(
                cls_map, offset_map, size_map, base_fusion_score
            )
            bbox = baseline_bbox
            erc_bbox = selected_bbox
        else:
            # Preserve the exact v1-v5 behavior for old configurations and
            # checkpoints.
            bbox = selected_bbox
            erc_bbox = selected_bbox
        out_dict.update({
            "cls_score": (cls_map * cont_score_2d) if self.joint_cls else cls_map,
            "bbox_map": bbox_map,
            "pred_boxes": bbox,
            "erc_pred_boxes": erc_bbox,
            "cont_score": cont_score,
            "prompts": prompts,
            "cls_score_test": cls_map,
            "base_fusion_score": erc_output['base_score'],
            "erc_cls_response": erc_output['cls_response'],
            "erc_cont_response": erc_output['cont_response'],
            "erc_windowed_cls_score": erc_output['windowed_cls_score'],
            "erc_windowed_cont_score": erc_output['windowed_cont_score'],
            "erc_windowed_base_score": erc_output['windowed_base_score'],
            "erc_score": erc_output['calibrated_score'],
            "erc_windowed_score": erc_output['windowed_calibrated_score'],
            "erc_global_score": erc_output.get(
                'global_calibrated_score', erc_output['calibrated_score']
            ),
            "erc_windowed_global_score": erc_output.get(
                'windowed_global_calibrated_score',
                erc_output['windowed_calibrated_score'],
            ),
            "erc_factor": erc_output['factor'],
            "erc_log_ratio": erc_output['log_ratio'],
            "erc_alpha": erc_output['alpha'],
            "erc_raw_alpha": erc_output['raw_alpha'],
            "erc_z_alpha": erc_output['z_alpha'],
            "erc_reliability": erc_output['reliability'],
            "erc_reliability_gate": erc_output['reliability_gate'],
            "erc_reliability_cls": erc_output['reliability_cls'],
            "erc_reliability_cont": erc_output['reliability_cont'],
            "erc_reliability_base": erc_output['reliability_base'],
            "erc_intervention_gate": erc_output['intervention_gate'],
            "erc_gate_logit": erc_output.get(
                'gate_logit', torch.zeros_like(
                    erc_output['intervention_gate']
                )
            ),
            "erc_regional_intervention_gate": erc_output.get(
                'regional_intervention_gate',
                erc_output['intervention_gate'],
            ),
            "erc_regional_gate_logit": erc_output.get(
                'regional_gate_logit',
                erc_output.get(
                    'gate_logit', torch.zeros_like(
                        erc_output['intervention_gate']
                    )
                ),
            ),
            "erc_delta": erc_output['delta'],
            "erc_alpha_global": erc_output.get(
                'alpha_global', erc_output['raw_alpha']
            ),
            "erc_alpha_region": erc_output.get(
                'alpha_region', torch.zeros_like(cls_map)
            ),
            "erc_alpha_final": erc_output.get(
                'alpha_final', erc_output['alpha']
            ),
            "erc_features": erc_output['features'],
            "erc_active_mask": erc_output['active_mask'],
        })
        for source_name, output_name in (
            ('pair_top_index', 'erc_pair_top_index'),
            ('pair_competitor_index', 'erc_pair_competitor_index'),
            ('pair_beta', 'erc_pair_beta'),
            ('pair_beta_logit', 'erc_pair_beta_logit'),
            ('pair_achieved_delta', 'erc_pair_achieved_delta'),
            ('pair_base_log_margin', 'erc_pair_base_log_margin'),
            ('pair_distance', 'erc_pair_distance'),
            ('local_top_index', 'erc_local_top_index'),
            ('local_indices', 'erc_local_indices'),
            ('local_valid_mask', 'erc_local_valid_mask'),
            ('local_logits', 'erc_local_logits'),
            ('local_centered_logits', 'erc_local_centered_logits'),
            ('local_predicted_delta_patch', 'erc_local_predicted_delta_patch'),
            ('local_delta_patch', 'erc_local_delta_patch'),
            ('local_delta_map', 'erc_local_delta_map'),
            ('local_reliability', 'erc_local_reliability'),
            ('local_reliability_logit', 'erc_local_reliability_logit'),
            ('local_effective_reliability', 'erc_local_effective_reliability'),
            ('v13_input_channels', 'erc_v13_input_channels'),
            ('v13_encoder_input', 'erc_v13_encoder_input'),
            ('v13_spatial_features', 'erc_v13_spatial_features'),
            ('v13_d_map', 'erc_v13_d_map'),
            ('v13_beta', 'erc_v13_beta'),
            ('v13_effective_beta', 'erc_v13_effective_beta'),
            ('v13_beta_logit', 'erc_v13_beta_logit'),
            ('v13_center_logits', 'erc_v13_center_logits'),
            ('v13_center_probability', 'erc_v13_center_probability'),
            ('v13_center_x', 'erc_v13_center_x'),
            ('v13_center_y', 'erc_v13_center_y'),
            ('v13_sigma_logits', 'erc_v13_sigma_logits'),
            ('v13_sigma_probability', 'erc_v13_sigma_probability'),
            ('v13_sigma', 'erc_v13_sigma'),
            ('v13_spatial_gate', 'erc_v13_spatial_gate'),
            ('v13_correction_map', 'erc_v13_correction_map'),
            ('v14_gate_logit', 'erc_v14_gate_logit'),
            ('v14_intervention_gate', 'erc_v14_intervention_gate'),
            ('v14_effective_gate', 'erc_v14_effective_gate'),
            ('v14_effective_beta', 'erc_v14_effective_beta'),
            ('v14_anchor_beta_logit', 'erc_v14_anchor_beta_logit'),
            ('v14_anchor_sigma_logits', 'erc_v14_anchor_sigma_logits'),
            ('v15_gate_logit', 'erc_v15_gate_logit'),
            ('v15_worthiness_gate', 'erc_v15_worthiness_gate'),
            ('v15_effective_gate', 'erc_v15_effective_gate'),
            ('v15_effective_beta', 'erc_v15_effective_beta'),
            ('v15_anchor_beta_logit', 'erc_v15_anchor_beta_logit'),
            ('v15_anchor_sigma_logits', 'erc_v15_anchor_sigma_logits'),
            ('v16_gate_logit', 'erc_v16_gate_logit'),
            ('v16_worthiness_gate', 'erc_v16_worthiness_gate'),
            ('v16_effective_gate', 'erc_v16_effective_gate'),
            ('v16_effective_beta', 'erc_v16_effective_beta'),
            ('v16_anchor_beta_logit', 'erc_v16_anchor_beta_logit'),
            ('v16_anchor_sigma_logits', 'erc_v16_anchor_sigma_logits'),
            ('v17_permission_logit', 'erc_v17_permission_logit'),
            ('v17_permission_probability', 'erc_v17_permission_probability'),
            ('v17_permission_hard', 'erc_v17_permission_hard'),
            ('v17_permission_gate', 'erc_v17_permission_gate'),
            ('v17_worthiness_logit', 'erc_v17_worthiness_logit'),
            ('v17_worthiness_gate', 'erc_v17_worthiness_gate'),
            ('v17_effective_worthiness', 'erc_v17_effective_worthiness'),
            ('v17_effective_gate', 'erc_v17_effective_gate'),
            ('v17_effective_beta', 'erc_v17_effective_beta'),
            ('v17_anchor_beta_logit', 'erc_v17_anchor_beta_logit'),
            ('v17_anchor_sigma_logits', 'erc_v17_anchor_sigma_logits'),
            ('v18_preservation_logit', 'erc_v18_preservation_logit'),
            ('v18_preservation_probability', 'erc_v18_preservation_probability'),
            ('v18_preservation_hard', 'erc_v18_preservation_hard'),
            ('v18_preservation_gate', 'erc_v18_preservation_gate'),
            ('v18_allow_after_veto', 'erc_v18_allow_after_veto'),
            ('v18_effective_gate', 'erc_v18_effective_gate'),
            ('v18_effective_beta', 'erc_v18_effective_beta'),
            ('v3_visual_response', 'erc_v3_visual_response'),
            ('v3_language_response', 'erc_v3_language_response'),
            ('v3_reference_response', 'erc_v3_reference_response'),
            ('v3_reference_statistics', 'erc_v3_reference_statistics'),
            ('v3_reliability_logits', 'erc_v3_reliability_logits'),
            ('v3_reliability', 'erc_v3_reliability'),
            ('v3_input_channels', 'erc_v3_input_channels'),
            ('v3_encoder_input', 'erc_v3_encoder_input'),
            ('v3_spatial_features', 'erc_v3_spatial_features'),
            ('v3_alpha_logits', 'erc_v3_alpha_logits'),
            ('v3_predicted_alpha', 'erc_v3_predicted_alpha'),
            ('v3_effective_alpha', 'erc_v3_effective_alpha'),
            ('v3_difference_map', 'erc_v3_difference_map'),
            ('v3_correction_basis', 'erc_v3_correction_basis'),
            ('v3_correction_map', 'erc_v3_correction_map'),
            ('v4_visual_response', 'erc_v4_visual_response'),
            ('v4_language_response', 'erc_v4_language_response'),
            ('v4_reference_context', 'erc_v4_reference_context'),
            ('v4_branch_statistics', 'erc_v4_branch_statistics'),
            ('v4_reliability_input', 'erc_v4_reliability_input'),
            ('v4_reliability_logits', 'erc_v4_reliability_logits'),
            ('v4_reliability', 'erc_v4_reliability'),
            ('v4_reliability_direction', 'erc_v4_reliability_direction'),
            ('v4_input_channels', 'erc_v4_input_channels'),
            ('v4_encoder_input', 'erc_v4_encoder_input'),
            ('v4_spatial_features', 'erc_v4_spatial_features'),
            ('v4_spatial_strength_logits', 'erc_v4_spatial_strength_logits'),
            ('v4_spatial_strength', 'erc_v4_spatial_strength'),
            ('v4_predicted_alpha', 'erc_v4_predicted_alpha'),
            ('v4_effective_alpha', 'erc_v4_effective_alpha'),
            ('v4_difference_map', 'erc_v4_difference_map'),
            ('v4_correction_basis', 'erc_v4_correction_basis'),
            ('v4_correction_map', 'erc_v4_correction_map'),
            ('v4_alpha_abs', 'erc_v4_alpha_abs'),
            ('v5_action_values', 'erc_v5_action_values'),
            ('v5_action_class_ema', 'erc_v5_action_class_ema'),
            ('v5_action_logits', 'erc_v5_action_logits'),
            ('v5_action_probability', 'erc_v5_action_probability'),
            ('v5_raw_hard_action', 'erc_v5_raw_hard_action'),
            ('v5_hard_action', 'erc_v5_hard_action'),
            ('v5_hard_policy', 'erc_v5_hard_policy'),
            ('v5_action_policy', 'erc_v5_action_policy'),
            ('v5_predicted_alpha_scalar', 'erc_v5_predicted_alpha_scalar'),
            ('v5_effective_alpha_scalar', 'erc_v5_effective_alpha_scalar'),
            ('v5_difference_map', 'erc_v5_difference_map'),
            ('v5_correction_basis', 'erc_v5_correction_basis'),
            ('v5_correction_map', 'erc_v5_correction_map'),
            ('v5_support_mask', 'erc_v5_support_mask'),
            ('v5_semantic_probability', 'erc_v5_semantic_probability'),
            ('v5_permission_logit', 'erc_v5_permission_logit'),
            ('v5_direction_logits', 'erc_v5_direction_logits'),
            ('v5_log_base', 'erc_v5_log_base'),
            ('v6_permission_logit', 'erc_v6_permission_logit'),
            ('v6_permission_probability', 'erc_v6_permission_probability'),
            ('v6_direction_logits', 'erc_v6_direction_logits'),
            ('v6_direction_probability', 'erc_v6_direction_probability'),
            ('v6_scope_logits', 'erc_v6_scope_logits'),
            ('v6_scope_probability', 'erc_v6_scope_probability'),
            ('v6_magnitude_logits', 'erc_v6_magnitude_logits'),
            ('v6_magnitude_probability', 'erc_v6_magnitude_probability'),
            ('v6_joint_probability', 'erc_v6_joint_probability'),
            ('v6_hard_action', 'erc_v6_hard_action'),
            ('v6_raw_hard_action', 'erc_v6_raw_hard_action'),
            ('v6_hard_policy', 'erc_v6_hard_policy'),
            ('v6_action_policy', 'erc_v6_action_policy'),
            ('v6_action_correction_maps', 'erc_v6_action_correction_maps'),
            ('v6_predicted_correction_map', 'erc_v6_predicted_correction_map'),
            ('v6_effective_correction_map', 'erc_v6_effective_correction_map'),
            ('v6_correction_basis', 'erc_v6_correction_basis'),
            ('v6_support_mask', 'erc_v6_support_mask'),
            ('v6_log_base', 'erc_v6_log_base'),
            ('v6_regional_t_mask', 'erc_v6_regional_t_mask'),
            ('v6_action_direction', 'erc_v6_action_direction'),
            ('v6_action_scope', 'erc_v6_action_scope'),
            ('v6_action_magnitude_index', 'erc_v6_action_magnitude_index'),
            ('v6_action_signed_alpha', 'erc_v6_action_signed_alpha'),
            ('v6_magnitudes', 'erc_v6_magnitudes'),
            ('v6_action_class_ema', 'erc_v6_action_class_ema'),
            ('v6_semantic_probability', 'erc_v6_semantic_probability'),
            ('av1_action_utility', 'erc_av1_action_utility'),
            ('av1_nonkeep_utility', 'erc_av1_nonkeep_utility'),
            ('av1_action_probability', 'erc_av1_action_probability'),
            ('av1_raw_hard_action', 'erc_av1_raw_hard_action'),
            ('av1_hard_action', 'erc_av1_hard_action'),
            ('av1_hard_policy', 'erc_av1_hard_policy'),
            ('av1_best_utility', 'erc_av1_best_utility'),
            ('av1_utility_gate', 'erc_av1_utility_gate'),
            ('av1_effective_gate', 'erc_av1_effective_gate'),
            ('av1_action_correction_maps', 'erc_av1_action_correction_maps'),
            ('av1_selected_full_correction', 'erc_av1_selected_full_correction'),
            ('av1_predicted_correction_map', 'erc_av1_predicted_correction_map'),
            ('av1_effective_correction_map', 'erc_av1_effective_correction_map'),
            ('av1_support_mask', 'erc_av1_support_mask'),
            ('av1_log_base', 'erc_av1_log_base'),
            ('av1_action_direction', 'erc_av1_action_direction'),
            ('av1_action_scope', 'erc_av1_action_scope'),
            ('av1_action_magnitude_index', 'erc_av1_action_magnitude_index'),
            ('av1_action_signed_alpha', 'erc_av1_action_signed_alpha'),
            ('av1_magnitudes', 'erc_av1_magnitudes'),
            ('av1_spatial_features', 'erc_av1_spatial_features'),
            ('av1_branch_statistics', 'erc_av1_branch_statistics'),
            ('av1_semantic_probability', 'erc_av1_semantic_probability'),
            ('av2_current_action_utility', 'erc_av2_current_action_utility'),
            ('av2_temporal_action_utility', 'erc_av2_temporal_action_utility'),
            ('av2_memory_used', 'erc_av2_memory_used'),
            ('av2_current_hard_action', 'erc_av2_current_hard_action'),
            ('av2_temporal_action_changed', 'erc_av2_temporal_action_changed'),
            ('av2_current_support', 'erc_av2_current_support'),
            ('av2_current_supported', 'erc_av2_current_supported'),
            ('av2_group_confirmed', 'erc_av2_group_confirmed'),
            ('av2_policy_confirmed', 'erc_av2_policy_confirmed'),
            ('av2_gate_utility', 'erc_av2_gate_utility'),
            ('av3_action_family', 'erc_av3_action_family'),
            ('av3_action_strength', 'erc_av3_action_strength'),
            ('av3_family_probability', 'erc_av3_family_probability'),
            ('av3_selected_family', 'erc_av3_selected_family'),
            ('av3_selected_strength', 'erc_av3_selected_strength'),
            ('av3_effective_strength', 'erc_av3_effective_strength'),
            ('av3_global_basis', 'erc_av3_global_basis'),
            ('av3_regional_basis', 'erc_av3_regional_basis'),
            ('av3_veto_basis', 'erc_av3_veto_basis'),
            ('av3_regional_mask', 'erc_av3_regional_mask'),
            ('av3_reference_agreement', 'erc_av3_reference_agreement'),
            ('av3_branch_consensus', 'erc_av3_branch_consensus'),
            ('av3_reference_consensus', 'erc_av3_reference_consensus'),
            ('av4_current_support', 'erc_av4_current_support'),
            ('av4_current_supported', 'erc_av4_current_supported'),
            ('av4_group_confirmed', 'erc_av4_group_confirmed'),
            ('av4_policy_confirmed', 'erc_av4_policy_confirmed'),
            ('av4_gate_utility', 'erc_av4_gate_utility'),
            ('cd1_reliability_logits', 'erc_cd1_reliability_logits'),
            ('cd1_raw_controls', 'erc_cd1_raw_controls'),
            ('cd1_temporal_controls', 'erc_cd1_temporal_controls'),
            ('cd1_raw_gate', 'erc_cd1_raw_gate'),
            ('cd1_temporal_gate', 'erc_cd1_temporal_gate'),
            ('cd1_strength', 'erc_cd1_strength'),
            ('cd1_regional_mix', 'erc_cd1_regional_mix'),
            ('cd1_direction', 'erc_cd1_direction'),
            ('cd1_alpha_total', 'erc_cd1_alpha_total'),
            ('cd1_alpha_global', 'erc_cd1_alpha_global'),
            ('cd1_alpha_regional', 'erc_cd1_alpha_regional'),
            ('cd1_difference', 'erc_cd1_difference'),
            ('cd1_regional_mask', 'erc_cd1_regional_mask'),
            ('cd1_raw_correction', 'erc_cd1_raw_correction'),
            ('cd1_effective_correction', 'erc_cd1_effective_correction'),
            ('cd1_support_mask', 'erc_cd1_support_mask'),
            ('cd1_memory_used', 'erc_cd1_memory_used'),
            ('cd5_reliability_logits', 'erc_cd5_reliability_logits'),
            ('cd5_reliability', 'erc_cd5_reliability'),
            ('cd5_raw_control_logits', 'erc_cd5_raw_control_logits'),
            ('cd5_raw_controls', 'erc_cd5_raw_controls'),
            ('cd5_temporal_controls', 'erc_cd5_temporal_controls'),
            ('cd5_raw_gate', 'erc_cd5_raw_gate'),
            ('cd5_gate', 'erc_cd5_gate'),
            ('cd5_raw_alpha_global', 'erc_cd5_raw_alpha_global'),
            ('cd5_raw_alpha_regional', 'erc_cd5_raw_alpha_regional'),
            ('cd5_alpha_global', 'erc_cd5_alpha_global'),
            ('cd5_alpha_regional', 'erc_cd5_alpha_regional'),
            ('cd5_effective_alpha_global',
             'erc_cd5_effective_alpha_global'),
            ('cd5_effective_alpha_regional',
             'erc_cd5_effective_alpha_regional'),
            ('cd5_difference', 'erc_cd5_difference'),
            ('cd5_regional_mask', 'erc_cd5_regional_mask'),
            ('cd5_coefficient', 'erc_cd5_coefficient'),
            ('cd5_raw_correction', 'erc_cd5_raw_correction'),
            ('cd5_effective_correction', 'erc_cd5_effective_correction'),
            ('cd5_support_mask', 'erc_cd5_support_mask'),
            ('cd5_memory_used', 'erc_cd5_memory_used'),
            ('cd5_spatial_input', 'erc_cd5_spatial_input'),
            ('pq_reliability_logits', 'erc_pq_reliability_logits'),
            ('pq_reliability', 'erc_pq_reliability'),
            ('pq_cls_peak_index', 'erc_pq_cls_peak_index'),
            ('pq_cont_peak_index', 'erc_pq_cont_peak_index'),
            ('pq_cls_peak_xy', 'erc_pq_cls_peak_xy'),
            ('pq_cont_peak_xy', 'erc_pq_cont_peak_xy'),
            ('pq_peak_distance', 'erc_pq_peak_distance'),
            ('pq_distribution_disagreement',
             'erc_pq_distribution_disagreement'),
            ('pq_disagreement', 'erc_pq_disagreement'),
            ('pq_reliability_gap', 'erc_pq_reliability_gap'),
            ('pq_raw_alpha', 'erc_pq_raw_alpha'),
            ('pq_alpha', 'erc_pq_alpha'),
            ('pq_difference', 'erc_pq_difference'),
            ('pq_raw_correction', 'erc_pq_raw_correction'),
            ('pq_correction', 'erc_pq_correction'),
            ('pq_support_mask', 'erc_pq_support_mask'),
            ('pq_spatial_input', 'erc_pq_spatial_input'),
            ('pq2_gate_logit', 'erc_pq2_gate_logit'),
            ('pq2_gate', 'erc_pq2_gate'),
            ('pq2_direction', 'erc_pq2_direction'),
            ('pq2_raw_alpha', 'erc_pq2_raw_alpha'),
            ('pq2_alpha', 'erc_pq2_alpha'),
            ('pq3_control_logits', 'erc_pq3_control_logits'),
            ('pq3_raw_delta', 'erc_pq3_raw_delta'),
            ('pq3_delta', 'erc_pq3_delta'),
            ('pq3_centered_log_c', 'erc_pq3_centered_log_c'),
            ('pq3_centered_log_t', 'erc_pq3_centered_log_t'),
            ('pq3_raw_correction', 'erc_pq3_raw_correction'),
            ('pq3_correction', 'erc_pq3_correction'),
            ('pq4_candidate_indices', 'erc_pq4_candidate_indices'),
            ('pq4_candidate_valid', 'erc_pq4_candidate_valid'),
            ('pq4_cls_ambiguity', 'erc_pq4_cls_ambiguity'),
            ('pq4_verifier_logits', 'erc_pq4_verifier_logits'),
            ('pq4_raw_peak_correction', 'erc_pq4_raw_peak_correction'),
            ('pq4_peak_correction', 'erc_pq4_peak_correction'),
            ('pq4_correction', 'erc_pq4_correction'),
            ('dr1_spatial_input', 'erc_dr1_spatial_input'),
            ('dr1_spatial_features', 'erc_dr1_spatial_features'),
            ('dr1_target_cosine', 'erc_dr1_target_cosine'),
            ('dr1_distractor_cosine', 'erc_dr1_distractor_cosine'),
            ('dr1_background_cosine', 'erc_dr1_background_cosine'),
            ('dr1_target_margin', 'erc_dr1_target_margin'),
            ('dr1_visual_response', 'erc_dr1_visual_response'),
            ('dr1_language_response', 'erc_dr1_language_response'),
            ('dr1_residual_logits', 'erc_dr1_residual_logits'),
            ('dr1_raw_correction', 'erc_dr1_raw_correction'),
            ('dr1_correction', 'erc_dr1_correction'),
            ('dr1_support_mask', 'erc_dr1_support_mask'),
            ('dr2_gate_features', 'erc_dr2_gate_features'),
            ('dr2_gate_logit', 'erc_dr2_gate_logit'),
            ('dr2_current_gate', 'erc_dr2_current_gate'),
            ('dr2_temporal_gate', 'erc_dr2_temporal_gate'),
            ('dr2_effective_gate', 'erc_dr2_effective_gate'),
            ('dr2_memory_used', 'erc_dr2_memory_used'),
            ('dr2_raw_correction', 'erc_dr2_raw_correction'),
            ('dr2_gated_correction', 'erc_dr2_gated_correction'),
            ('dr2_correction', 'erc_dr2_correction'),
            ('dr2_baseline_margin', 'erc_dr2_baseline_margin'),
            ('dr2_base_entropy', 'erc_dr2_base_entropy'),
            ('dr2_branch_peak_distance', 'erc_dr2_branch_peak_distance'),
            ('dr2_proposal_distance', 'erc_dr2_proposal_distance'),
            ('dr2_baseline_index', 'erc_dr2_baseline_index'),
            ('dr2_proposal_index', 'erc_dr2_proposal_index'),
            ('dr2_scalar_features', 'erc_dr2_scalar_features'),
            ('dr2_correction_strength', 'erc_dr2_correction_strength'),
            ('dr3_current_gate', 'erc_dr3_current_gate'),
            ('dr3_temporal_gate', 'erc_dr3_temporal_gate'),
            ('dr3_effective_gate', 'erc_dr3_effective_gate'),
            ('dr3_raw_correction', 'erc_dr3_raw_correction'),
            ('dr3_gated_correction', 'erc_dr3_gated_correction'),
            ('dr3_correction', 'erc_dr3_correction'),
            ('dr3_proposal_distance', 'erc_dr3_proposal_distance'),
            ('dr3_baseline_index', 'erc_dr3_baseline_index'),
            ('dr3_proposal_index', 'erc_dr3_proposal_index'),
        ):
            if source_name in erc_output:
                out_dict[output_name] = erc_output[source_name]
        if 'trc_output' in erc_output:
            out_dict['erc_trc_output'] = erc_output['trc_output']
        if feature_erc_output is not None:
            out_dict['erc_feature_output'] = feature_erc_output
        return out_dict
    
    def forward_prompt(self, out_dict):
        flag = out_dict['flag']
        vis_token = out_dict['vis_token']
        txt_token = out_dict['txt_token']
        token_group = torch.cat([vis_token, txt_token, (vis_token+txt_token)/2], dim=1)
        bid = torch.arange(flag.shape[0])
        token = token_group[bid, flag]
        prompt = self.prompter(out_dict['template'], out_dict['template_mask'], 
                                out_dict['search'], out_dict['context_mask'], 
                                token, out_dict['flag'])
        return prompt
        
    def convert2bbox(self, cls_map, offset_map, size_map, selection_score):
        b = cls_map.shape[0]
        b_idx = torch.arange(0, b)
        s_idx = torch.argmax(selection_score, dim=-1)
        offset_map = offset_map.reshape(b, 2, -1)
        size_map   = size_map.reshape(b, 2, -1)
        ctr_map = self.coodinate.repeat(b, 1, 1) # b, 2, s
        ctr_map = (ctr_map+offset_map) / self.feat_sz
        bbox_map = torch.cat([ctr_map, size_map], dim=1).transpose(-2, -1) # b, s, 4
        bbox = bbox_map[b_idx, s_idx]
        return bbox_map, bbox.unsqueeze(1)

    def contractive_learning(self, out_dict):
        # for training
        if out_dict.get('prompt', None) is None:
            flag = out_dict['flag']
            vis_token = out_dict['vis_token']
            txt_token = out_dict['txt_token']
            token_group = torch.cat([vis_token, txt_token, (vis_token+txt_token)/2], dim=1)
            bid = torch.arange(flag.shape[0])
            token = token_group[bid, flag]
            search = out_dict['search']
            context = torch.cat([search[bid.shape[0]//2:], search[:bid.shape[0]//2]], dim=0)
            prompt = self.prompter(out_dict['template'], out_dict['template_mask'], context, out_dict['context_mask'], token, out_dict['flag']) # b, 2, c
            cont_score = self.logit_scale.exp() * (F.normalize(out_dict['search'], dim=-1) @ F.normalize(prompt, dim=-1).transpose(-2,-1))
            if self.softmax_one:
                ext_one = torch.zeros_like(cont_score[:, :, :1])
                cont_score = torch.cat([cont_score[:, :, :1], torch.cat([cont_score[:, :, 1:], ext_one], dim=-1).max(dim=-1, keepdim=True).values], dim=-1)
            else:
                cont_score = torch.cat([cont_score[:, :, :1], cont_score[:, :, 1:].max(dim=-1, keepdim=True).values], dim=-1)
        # for testing
        else:
            prompt = out_dict['prompt']
            cont_score = self.logit_scale.exp() * (F.normalize(out_dict['search'], dim=-1) @ F.normalize(prompt, dim=-1).transpose(-2,-1))
            
            if self.softmax_one:
                ext_one = torch.zeros_like(cont_score[:, :, :1])
                cont_score = torch.cat([cont_score[:, :, :1], torch.cat([cont_score[:, :, 1:], ext_one], dim=-1).max(dim=-1, keepdim=True).values, ext_one], dim=-1)
            else:
                cont_score = torch.cat([cont_score[:, :, :1], cont_score[:, :, 1:].max(dim=-1, keepdim=True).values], dim=-1)

        return cont_score, prompt

from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xywh_to_cxcywh, box_xywh_to_cxcywh_scale
import torch
from lib import registry
from lib.utils.box_ops import giou_loss, GaussWeightedLoss
from torch.nn.functional import l1_loss
from lib.utils.misc import NestedTensor
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
from lib.models.heads.response_worthiness_gate_erc_v15 import (
    continuous_worthiness_target,
)
from lib.models.heads.response_selective_abstention_worthiness_gate_erc_v17 import (
    selective_permission_target,
)
from lib.train.actors.erc_trc_loss import compute_trc_loss_and_metrics
from lib.train.actors.erc_trc_v2_loss import compute_trc_v2_loss_and_metrics
from lib.train.actors.erc_trc_v3_loss import compute_trc_v3_loss_and_metrics
from lib.train.actors.erc_trc_v4_loss import compute_trc_v4_loss_and_metrics
from lib.train.actors.erc_trc_v4b_loss import compute_trc_v4b_loss_and_metrics
from lib.train.actors.erc_trc_v5_loss import compute_trc_v5_loss_and_metrics
from lib.train.actors.erc_trc_v5b_loss import compute_trc_v5b_loss_and_metrics
from lib.train.actors.erc_trc_v5c_loss import compute_trc_v5c_loss_and_metrics
from lib.train.actors.erc_trc_v5d_loss import compute_trc_v5d_loss_and_metrics
from lib.train.actors.erc_trc_v5g_loss import compute_trc_v5g_loss_and_metrics
from lib.train.actors.erc_trc_v5h_loss import compute_trc_v5h_loss_and_metrics
from lib.train.actors.erc_trc_v6_loss import compute_trc_v6_loss_and_metrics
from lib.train.actors.erc_trc_v6b_loss import compute_trc_v6b_loss_and_metrics
from lib.train.actors.erc_feature_reliability_loss import (
    compute_feature_reliability_loss_and_metrics,
)
from lib.train.actors.erc_dense_response_reliability_loss import (
    compute_dense_response_reliability_loss_and_metrics,
)
from lib.train.actors.erc_branch_aligned_dense_response_loss import (
    compute_branch_aligned_dense_response_loss_and_metrics,
)
from lib.train.actors.erc_branch_aligned_tristate_multiaction_loss import (
    compute_branch_aligned_tristate_multiaction_loss_and_metrics,
)
from lib.train.actors.erc_branch_aligned_factorized_reachability_loss import (
    compute_factorized_reachability_loss_and_metrics,
)
from lib.train.actors.erc_branch_aligned_iou_utility_loss import (
    compute_iou_utility_loss_and_metrics,
)
from lib.train.actors.erc_branch_aligned_action_value_loss import (
    compute_action_value_loss_and_metrics,
)
from lib.train.actors.erc_branch_aligned_action_value_balanced_loss import (
    compute_balanced_action_value_loss_and_metrics,
)
from lib.train.actors.erc_temporal_consensus_action_value_loss import (
    compute_temporal_consensus_action_value_loss_and_metrics,
)
from lib.train.actors.erc_cross_domain_reliability_loss import (
    compute_cross_domain_reliability_loss_and_metrics,
)
from lib.train.actors.erc_cross_domain_continuous_oracle_loss import (
    compute_cross_domain_continuous_oracle_loss_and_metrics,
)
from lib.train.actors.erc_peak_quality_calibration_loss import (
    compute_peak_quality_calibration_loss_and_metrics,
)
from lib.train.actors.erc_peak_quality_decision_loss import (
    compute_peak_quality_decision_loss_and_metrics,
)
from lib.train.actors.erc_peak_quality_independent_branch_loss import (
    compute_peak_quality_independent_branch_loss_and_metrics,
)
from lib.train.actors.erc_peak_ambiguity_target_verifier_loss import (
    compute_peak_ambiguity_target_verifier_loss_and_metrics,
)
from lib.train.actors.erc_target_reference_direct_dense_loss import (
    compute_target_reference_direct_dense_loss_and_metrics,
)
from lib.train.actors.erc_target_reference_trajectory_safe_dense_loss import (
    compute_target_reference_trajectory_safe_dense_loss_and_metrics,
)
from lib.train.actors.erc_target_reference_realized_action_dense_loss import (
    compute_target_reference_realized_action_dense_loss_and_metrics,
)

@registry.ACTORS.register("uvltrack")
class UVLTrackActor(BaseActor):
    """ Actor for training the TSP_online and TSP_cls_online"""
    def __init__(self, net, cfg):
        super().__init__(net)
        self.cfg = cfg
        self.build_loss(cfg)
        self._response_erc_design = getattr(
            cfg.MODEL.HEAD.ERC, 'DESIGN', 'legacy'
        )
        self._is_response_v7 = self._response_erc_design in (
            'foreground_reference_evidence_erc_fr1',
            'target_reference_erc_trc_v1',
            'target_reference_erc_trc_v2_decision_aligned',
            'target_reference_erc_trc_v3_decision_selective',
            'target_reference_erc_trc_v4_hard_decision',
            'target_reference_erc_trc_v4b_factorized',
            'target_reference_erc_trc_v5_calibrated_selective',
            'target_reference_erc_trc_v5b_action_calibrated',
            'target_reference_erc_trc_v5c_deployment_aligned',
            'target_reference_erc_trc_v5d_decoupled_balanced',
            'target_reference_erc_trc_v5f_direction_calibrated',
            'target_reference_erc_trc_v5g_branch_conditional_permission',
            'target_reference_erc_trc_v5h_positive_anchored_permission',
            'target_reference_erc_trc_v6_joint_action_utility',
            'target_reference_erc_trc_v6b_balanced_joint_action',
            'reference_reliability_feature_erc_v1',
            'reference_reliability_feature_erc_v2_aligned_tokens',
            'reference_reliability_dense_response_erc_v3',
            'branch_aligned_dense_response_erc_v4',
            'branch_aligned_tristate_multiaction_erc_v5',
            'branch_aligned_factorized_reachability_erc_v6',
            'branch_aligned_iou_utility_erc_v7',
            'branch_aligned_action_value_erc_av1',
            'branch_aligned_action_value_erc_av1b_balanced',
            'branch_aligned_temporal_consensus_erc_av2',
            'cross_modal_consensus_residual_erc_av3',
            'cross_domain_normalized_consensus_erc_av4',
            'cross_domain_global_regional_reliability_erc_v1',
            'cross_domain_continuous_oracle_erc_av5',
            'peak_quality_calibrated_erc_pq1',
            'peak_quality_decision_aligned_erc_pq2',
            'peak_quality_independent_branch_erc_pq3',
            'peak_ambiguity_target_verifier_erc_pq4',
            'target_reference_direct_dense_erc_dr1',
            'target_reference_trajectory_safe_dense_erc_dr2',
            'target_reference_realized_action_dense_erc_dr3',
            'response_global_regional_v7',
            'response_hann_hardpeak_v8',
            'response_hann_gate_v8c',
            'response_regional_effective_v8d',
            'response_precision_incremental_v9',
            'response_competition_pair_v10',
            'response_local_spatial_v11',
            'response_local_reliability_v12',
            'response_branch_spatial_gate_v13',
            'response_intervention_gate_v14',
            'response_continuous_worthiness_gate_v15',
            'response_asymmetric_worthiness_gate_v16',
            'response_selective_abstention_worthiness_gate_v17',
            'response_preservation_veto_gate_v18',
        )
        self._head_param_reference = {}
        self._head_bn_reference = {}
        self._head_teacher = None
        if self._is_response_v7:
            self._initialize_head_reference()
            if self.cfg.TRAIN.HEAD_RESPONSE_DISTILL_ENABLE:
                self._initialize_head_teacher()

    def build_loss(self, cfg):
        weight = torch.tensor([self.cfg.DATA.SEARCH.FACTOR**2, self.cfg.TRAIN.CTR_RATIO**2]).cuda()
        weight = weight / weight.sum()
        self.objective = {'giou': giou_loss, 'l1': l1_loss, 
                          'cls': GaussWeightedLoss(reduction=cfg.TRAIN.REDUCTION),
                          'aux': CrossEntropyLoss(), 
                          'cib': CrossEntropyLoss(ignore_index=-1), 
                          'cont': CrossEntropyLoss(ignore_index=-1, weight=weight)}
        self.loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 
                            'cls': 1, 'aux': cfg.TRAIN.AUX_WEIGHT, 
                            'cib': cfg.TRAIN.CIB_WEIGHT, 'cont': cfg.TRAIN.CONT_WEIGHT,
                            'erc_reliability': cfg.TRAIN.ERC_RELIABILITY_WEIGHT,
                            'erc_reliability_rank': cfg.TRAIN.ERC_RELIABILITY_RANK_WEIGHT,
                            'erc_rank': cfg.TRAIN.ERC_RANK_WEIGHT,
                            'erc_alpha_reg': cfg.TRAIN.ERC_ALPHA_REG_WEIGHT,
                            'erc_alpha_sup': cfg.TRAIN.ERC_ALPHA_SUP_WEIGHT,
                            'erc_decision': cfg.TRAIN.ERC_DECISION_WEIGHT,
                            'erc_gate': cfg.TRAIN.ERC_GATE_WEIGHT,
                            'erc_noharm': cfg.TRAIN.ERC_NOHARM_WEIGHT,
                            'erc_delta': cfg.TRAIN.ERC_DELTA_WEIGHT,
                            'erc_harm': cfg.TRAIN.ERC_HARM_WEIGHT,
                            'erc_geo': cfg.TRAIN.ERC_GEO_WEIGHT,
                            'erc_response_map': cfg.TRAIN.ERC_RESPONSE_MAP_WEIGHT,
                            'erc_response_rank': cfg.TRAIN.ERC_RESPONSE_RANK_WEIGHT,
                            'erc_response_rel': cfg.TRAIN.ERC_RESPONSE_REL_WEIGHT,
                            'erc_response_gate': cfg.TRAIN.ERC_RESPONSE_GATE_WEIGHT,
                            'erc_response_pref': cfg.TRAIN.ERC_RESPONSE_PREF_WEIGHT,
                            'erc_response_noharm': cfg.TRAIN.ERC_RESPONSE_NOHARM_WEIGHT,
                            'erc_response_identity': cfg.TRAIN.ERC_RESPONSE_IDENTITY_WEIGHT,
                            'erc_response_smooth': cfg.TRAIN.ERC_RESPONSE_SMOOTH_WEIGHT,
                            'erc_v8_hard_correct': cfg.TRAIN.ERC_V8_HARD_CORRECT_WEIGHT,
                            'erc_v8_map': cfg.TRAIN.ERC_V8_MAP_WEIGHT,
                            'erc_v8_rel': cfg.TRAIN.ERC_V8_REL_WEIGHT,
                            'erc_v8_gate': cfg.TRAIN.ERC_V8_GATE_WEIGHT,
                            'erc_v8_pref': cfg.TRAIN.ERC_V8_PREF_WEIGHT,
                            'erc_v8_hard_preserve': cfg.TRAIN.ERC_V8_HARD_PRESERVE_WEIGHT,
                            'erc_v8_margin_noharm': cfg.TRAIN.ERC_V8_MARGIN_NOHARM_WEIGHT,
                            'erc_v8_smooth': cfg.TRAIN.ERC_V8_SMOOTH_WEIGHT,
                            'erc_v8_gate_separation': cfg.TRAIN.ERC_V8_GATE_SEPARATION_WEIGHT,
                            'erc_v8_gate_pairwise': cfg.TRAIN.ERC_V8_GATE_PAIRWISE_WEIGHT,
                            'erc_v8_negative_saturation': cfg.TRAIN.ERC_V8_NEGATIVE_SATURATION_WEIGHT,
                            'erc_v8_regional_effective': cfg.TRAIN.ERC_V8_REGIONAL_EFFECTIVE_WEIGHT,
                            'erc_v8_regional_noharm': cfg.TRAIN.ERC_V8_REGIONAL_NOHARM_WEIGHT,
                            'erc_v9_global_oracle': cfg.TRAIN.ERC_V9_GLOBAL_ORACLE_WEIGHT,
                            'erc_v9_regional_gate': cfg.TRAIN.ERC_V9_REGIONAL_GATE_WEIGHT,
                            'erc_v9_regional_identity': cfg.TRAIN.ERC_V9_REGIONAL_IDENTITY_WEIGHT,
                            'erc_v10_pair_correct': cfg.TRAIN.ERC_V10_PAIR_CORRECT_WEIGHT,
                            'erc_v10_pair_gate': cfg.TRAIN.ERC_V10_PAIR_GATE_WEIGHT,
                            'erc_v10_pair_preserve': cfg.TRAIN.ERC_V10_PAIR_PRESERVE_WEIGHT,
                            'erc_v10_pair_identity': cfg.TRAIN.ERC_V10_PAIR_IDENTITY_WEIGHT,
                            'erc_v11_local': cfg.TRAIN.ERC_V11_LOCAL_WEIGHT,
                            'erc_v11_hard': cfg.TRAIN.ERC_V11_HARD_WEIGHT,
                            'erc_v11_gap': cfg.TRAIN.ERC_V11_GAP_WEIGHT,
                            'erc_v11_outside': cfg.TRAIN.ERC_V11_OUTSIDE_WEIGHT,
                            'erc_v11_reg': cfg.TRAIN.ERC_V11_REG_WEIGHT,
                            'erc_v12_local': cfg.TRAIN.ERC_V12_LOCAL_WEIGHT,
                            'erc_v12_hard': cfg.TRAIN.ERC_V12_HARD_WEIGHT,
                            'erc_v12_gap': cfg.TRAIN.ERC_V12_GAP_WEIGHT,
                            'erc_v12_outside': cfg.TRAIN.ERC_V12_OUTSIDE_WEIGHT,
                            'erc_v12_reliability': cfg.TRAIN.ERC_V12_RELIABILITY_WEIGHT,
                            'erc_v12_preserve': cfg.TRAIN.ERC_V12_PRESERVE_WEIGHT,
                            'erc_v12_identity': cfg.TRAIN.ERC_V12_IDENTITY_WEIGHT,
                            'erc_v12_reg': cfg.TRAIN.ERC_V12_REG_WEIGHT,
                            'erc_v13_center': cfg.TRAIN.ERC_V13_CENTER_WEIGHT,
                            'erc_v13_sigma': cfg.TRAIN.ERC_V13_SIGMA_WEIGHT,
                            'erc_v13_beta': cfg.TRAIN.ERC_V13_BETA_WEIGHT,
                            'erc_v13_sign': cfg.TRAIN.ERC_V13_SIGN_WEIGHT,
                            'erc_v13_rank': cfg.TRAIN.ERC_V13_RANK_WEIGHT,
                            'erc_v13_preserve': cfg.TRAIN.ERC_V13_PRESERVE_WEIGHT,
                            'erc_v13_noharm': cfg.TRAIN.ERC_V13_NOHARM_WEIGHT,
                            'erc_v13_reg': cfg.TRAIN.ERC_V13_REG_WEIGHT,
                            'erc_v14_gate': cfg.TRAIN.ERC_V14_GATE_WEIGHT,
                            'erc_v14_gate_separation': cfg.TRAIN.ERC_V14_GATE_SEPARATION_WEIGHT,
                            'erc_v15_gate_soft': cfg.TRAIN.ERC_V15_GATE_SOFT_WEIGHT,
                            'erc_v15_gate_zero': cfg.TRAIN.ERC_V15_GATE_ZERO_WEIGHT,
                            'erc_v15_gate_rank': cfg.TRAIN.ERC_V15_GATE_RANK_WEIGHT,
                            'erc_v16_gate_calibration': cfg.TRAIN.ERC_V16_GATE_CALIBRATION_WEIGHT,
                            'erc_v16_gate_keep_logit': cfg.TRAIN.ERC_V16_GATE_KEEP_LOGIT_WEIGHT,
                            'erc_v16_gate_rank': cfg.TRAIN.ERC_V16_GATE_RANK_WEIGHT,
                            'erc_v17_permission': cfg.TRAIN.ERC_V17_PERMISSION_WEIGHT,
                            'erc_v17_worthiness': cfg.TRAIN.ERC_V17_WORTHINESS_WEIGHT,
                            'erc_v17_rank': cfg.TRAIN.ERC_V17_RANK_WEIGHT,
                            'erc_v18_preservation': cfg.TRAIN.ERC_V18_PRESERVATION_WEIGHT,
                            'erc_v18_preservation_rank': cfg.TRAIN.ERC_V18_PRESERVATION_RANK_WEIGHT}

    def _model(self):
        return self.net.module if hasattr(self.net, 'module') else self.net

    def _initialize_head_reference(self):
        """Snapshot the official head immediately after checkpoint loading."""
        model = self._model()
        with torch.no_grad():
            if self.cfg.TRAIN.HEAD_ANCHOR_ENABLE:
                self._head_param_reference = {
                    name: parameter.detach().clone()
                    for name, parameter in model.box_head.named_parameters()
                    if not name.startswith('erc.') and parameter.requires_grad
                }
            self._head_bn_reference = {
                name: (
                    module.running_mean.detach().clone(),
                    module.running_var.detach().clone(),
                    module.num_batches_tracked.detach().clone()
                    if module.num_batches_tracked is not None else None,
                )
                for name, module in model.box_head.named_modules()
                if isinstance(module, nn.modules.batchnorm._BatchNorm)
            }

    def _initialize_head_teacher(self):
        """Freeze an official-checkpoint Head for response distillation.

        The actor is constructed after INIT_CHECKPOINT is loaded, so this copy
        is the official Head even when a later pilot checkpoint is resumed.
        ERC is removed from the teacher because only the original C/T response
        maps are required.
        """
        self._head_teacher = deepcopy(self._model().box_head)
        self._head_teacher.erc = None
        self._head_teacher.eval()
        for parameter in self._head_teacher.parameters():
            parameter.requires_grad = False

    def _head_anchor_terms(self):
        """Return L2-SP and drift diagnostics without including ERC params."""
        reference_tensor = next(self._model().box_head.parameters())
        zero = reference_tensor.sum() * 0.0
        squared_delta = zero
        squared_reference = zero.detach()
        parameter_count = 0
        if self._head_param_reference:
            current_parameters = dict(
                self._model().box_head.named_parameters()
            )
            for name, reference in self._head_param_reference.items():
                current = current_parameters[name]
                squared_delta = squared_delta + (current - reference).square().sum()
                squared_reference = (
                    squared_reference + reference.square().sum()
                )
                parameter_count += current.numel()
        if not parameter_count:
            anchor_loss = zero
        elif getattr(self.cfg.TRAIN, 'HEAD_ANCHOR_RELATIVE', False):
            anchor_loss = squared_delta / squared_reference.clamp_min(1e-12)
        else:
            anchor_loss = squared_delta / max(parameter_count, 1)
        delta_norm = squared_delta.detach().clamp_min(0.0).sqrt()
        relative_delta = delta_norm / (
            squared_reference.clamp_min(0.0).sqrt() + 1e-12
        )

        bn_mean_squared = zero.detach()
        bn_var_squared = zero.detach()
        bn_batches_delta = zero.detach()
        with torch.no_grad():
            for name, module in self._model().box_head.named_modules():
                if name not in self._head_bn_reference:
                    continue
                mean_reference, var_reference, batches_reference = (
                    self._head_bn_reference[name]
                )
                bn_mean_squared = bn_mean_squared + (
                    module.running_mean - mean_reference
                ).square().sum()
                bn_var_squared = bn_var_squared + (
                    module.running_var - var_reference
                ).square().sum()
                if batches_reference is not None:
                    bn_batches_delta = bn_batches_delta + (
                        module.num_batches_tracked.float()
                        - batches_reference.float()
                    ).abs()
        return anchor_loss, {
            'Loss/head_anchor': anchor_loss.detach(),
            'Head/param_delta_norm': delta_norm,
            'Head/param_relative_delta': relative_delta,
            'Head/bn_running_mean_delta': bn_mean_squared.sqrt(),
            'Head/bn_running_var_delta': bn_var_squared.sqrt(),
            'Head/bn_num_batches_tracked_delta': bn_batches_delta,
        }

    def train(self, mode=True):
        """Keep the frozen baseline deterministic during ERC-only training."""
        self.net.train(mode)
        if mode and self.cfg.TRAIN.ERC_ONLY:
            model = self.net.module if hasattr(self.net, 'module') else self.net
            model.backbone.eval()
            model.box_head.eval()
            model.box_head.erc.train(True)
        elif mode and self.cfg.TRAIN.FREEZE_BACKBONE:
            model = self.net.module if hasattr(self.net, 'module') else self.net
            model.backbone.eval()
        if mode and getattr(self.cfg.TRAIN, 'FREEZE_HEAD', False):
            model = self._model()
            model.box_head.eval()
        if mode and getattr(self.cfg.TRAIN, 'FREEZE_BBOX_GEOMETRY', False):
            model = self._model()
            for module_name in (
                    'conv_offset', 'conv_bbox', 'conv_bbox_grounding'):
                module = getattr(model.box_head, module_name, None)
                if not isinstance(module, nn.Module):
                    raise TypeError(
                        "TRAIN.FREEZE_BBOX_GEOMETRY requires "
                        "box_head.%s to be an nn.Module" % module_name
                    )
                module.eval()
        if mode and getattr(self.cfg.TRAIN, 'FREEZE_HEAD_BN_STATS', False):
            model = self._model()
            for module in model.box_head.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    # Keep affine weight/bias trainable while freezing running
                    # mean/variance updates from model.train().
                    module.eval()
        if self._head_teacher is not None:
            self._head_teacher.eval()

    def _gate_curriculum_lambda(self, epoch, training):
        """Return the training-only Gate-to-Alpha curriculum coefficient."""
        if not training or not self.cfg.TRAIN.ERC_GATE_ALPHA_CURRICULUM:
            return 1.0
        values = list(self.cfg.TRAIN.ERC_GATE_ALPHA_CURRICULUM_VALUES)
        if not values:
            return 1.0
        epoch_index = max(int(epoch or 1) - 1, 0)
        return float(values[min(epoch_index, len(values) - 1)])

    def _safety_curriculum_weights(self, epoch, training):
        """Return response no-harm/identity weights for the current epoch."""
        final_noharm = float(self.cfg.TRAIN.ERC_RESPONSE_NOHARM_WEIGHT)
        final_identity = float(self.cfg.TRAIN.ERC_RESPONSE_IDENTITY_WEIGHT)
        if not training or not self.cfg.TRAIN.ERC_SAFETY_CURRICULUM:
            return final_noharm, final_identity
        epoch = int(epoch or 1)
        if epoch <= 1:
            return (
                float(self.cfg.TRAIN.ERC_RESPONSE_NOHARM_WEIGHT_EPOCH1),
                float(self.cfg.TRAIN.ERC_RESPONSE_IDENTITY_WEIGHT_EPOCH1),
            )
        if epoch == 2:
            return (
                float(self.cfg.TRAIN.ERC_RESPONSE_NOHARM_WEIGHT_EPOCH2),
                float(self.cfg.TRAIN.ERC_RESPONSE_IDENTITY_WEIGHT_EPOCH2),
            )
        return final_noharm, final_identity

    def _v8_safety_curriculum_weights(self, epoch, training):
        """Return v8 hard-preserve and ordering-no-harm weights."""
        final_preserve = float(
            self.cfg.TRAIN.ERC_V8_HARD_PRESERVE_WEIGHT
        )
        final_noharm = float(
            self.cfg.TRAIN.ERC_V8_MARGIN_NOHARM_WEIGHT
        )
        if not training or not self.cfg.TRAIN.ERC_SAFETY_CURRICULUM:
            return final_preserve, final_noharm
        epoch = int(epoch or 1)
        if epoch <= 1:
            return (
                float(self.cfg.TRAIN.ERC_V8_HARD_PRESERVE_WEIGHT_EPOCH1),
                float(self.cfg.TRAIN.ERC_V8_MARGIN_NOHARM_WEIGHT_EPOCH1),
            )
        if epoch == 2:
            return (
                float(self.cfg.TRAIN.ERC_V8_HARD_PRESERVE_WEIGHT_EPOCH2),
                float(self.cfg.TRAIN.ERC_V8_MARGIN_NOHARM_WEIGHT_EPOCH2),
            )
        return final_preserve, final_noharm

    @staticmethod
    def _gradient_norm(parameters, reference):
        squared = reference.new_zeros(())
        for parameter in parameters:
            if parameter.grad is not None:
                squared = squared + parameter.grad.detach().float().square().sum()
        return squared.sqrt()

    def gradient_diagnostics(self, status=None):
        """Collect post-backward v7 branch gradients for the training log."""
        if not self._is_response_v7:
            return {}
        model = self._model()
        erc = model.box_head.erc
        reference = next(erc.parameters())
        if getattr(erc, 'design', '') == 'foreground_reference_evidence_erc_fr1':
            return {
                'Gradient/erc_total_norm': self._gradient_norm(erc.parameters(), reference),
                'Gradient/head_norm': self._gradient_norm((p for n,p in model.box_head.named_parameters() if not n.startswith('erc.')), reference),
                'Gradient/backbone_norm': self._gradient_norm(model.backbone.parameters(), reference),
            }
        if getattr(erc, 'design', '') in (
            'reference_reliability_feature_erc_v1',
            'reference_reliability_feature_erc_v2_aligned_tokens',
        ):
            return {
                'Gradient/reliability_head_norm': self._gradient_norm(
                    erc.reliability_head.parameters(), reference),
                'Gradient/feature_adapter_down_norm': self._gradient_norm(
                    list(erc.search_down.parameters())
                    + list(erc.reference_down.parameters()), reference),
                'Gradient/feature_adapter_up_norm': self._gradient_norm(
                    erc.adapter_up.parameters(), reference),
                'Gradient/erc_total_norm': self._gradient_norm(
                    erc.parameters(), reference),
                'Gradient/head_norm': self._gradient_norm(
                    (p for n, p in model.box_head.named_parameters()
                     if not n.startswith('erc.')), reference),
                'Gradient/backbone_norm': self._gradient_norm(
                    model.backbone.parameters(), reference),
            }
        if getattr(erc, 'design', '') == 'reference_reliability_dense_response_erc_v3':
            return {
                'Gradient/reliability_head_norm': self._gradient_norm(
                    erc.reliability_head.parameters(), reference),
                'Gradient/dense_encoder_norm': self._gradient_norm(
                    erc.encoder.parameters(), reference),
                'Gradient/dense_alpha_head_norm': self._gradient_norm(
                    erc.alpha_head.parameters(), reference),
                'Gradient/erc_total_norm': self._gradient_norm(
                    erc.parameters(), reference),
                'Gradient/head_norm': self._gradient_norm(
                    (p for n, p in model.box_head.named_parameters()
                     if not n.startswith('erc.')), reference),
                'Gradient/backbone_norm': self._gradient_norm(
                    model.backbone.parameters(), reference),
            }
        if getattr(erc, 'design', '') == 'target_reference_direct_dense_erc_dr1':
            return {
                'Gradient/dense_encoder_norm': self._gradient_norm(
                    erc.encoder.parameters(), reference),
                'Gradient/dense_residual_head_norm': self._gradient_norm(
                    erc.residual_head.parameters(), reference),
                'Gradient/erc_total_norm': self._gradient_norm(
                    erc.parameters(), reference),
                'Gradient/head_norm': self._gradient_norm(
                    (p for n, p in model.box_head.named_parameters()
                     if not n.startswith('erc.')), reference),
                'Gradient/backbone_norm': self._gradient_norm(
                    model.backbone.parameters(), reference),
            }
        if getattr(erc, 'design', '') in (
                'target_reference_trajectory_safe_dense_erc_dr2',
                'target_reference_realized_action_dense_erc_dr3'):
            return {
                'Gradient/dense_encoder_norm': self._gradient_norm(
                    erc.encoder.parameters(), reference),
                'Gradient/dense_residual_head_norm': self._gradient_norm(
                    erc.residual_head.parameters(), reference),
                'Gradient/safety_gate_head_norm': self._gradient_norm(
                    erc.safety_head.parameters(), reference),
                'Gradient/erc_total_norm': self._gradient_norm(
                    erc.parameters(), reference),
                'Gradient/head_norm': self._gradient_norm(
                    (p for n, p in model.box_head.named_parameters()
                     if not n.startswith('erc.')), reference),
                'Gradient/backbone_norm': self._gradient_norm(
                    model.backbone.parameters(), reference),
            }
        if getattr(erc, 'design', '') == 'branch_aligned_dense_response_erc_v4':
            return {
                'Gradient/reliability_head_norm': self._gradient_norm(
                    erc.reliability_head.parameters(), reference),
                'Gradient/dense_encoder_norm': self._gradient_norm(
                    erc.encoder.parameters(), reference),
                'Gradient/dense_spatial_strength_head_norm': self._gradient_norm(
                    erc.spatial_strength_head.parameters(), reference),
                'Gradient/erc_total_norm': self._gradient_norm(
                    erc.parameters(), reference),
                'Gradient/head_norm': self._gradient_norm(
                    (p for n, p in model.box_head.named_parameters()
                     if not n.startswith('erc.')), reference),
                'Gradient/backbone_norm': self._gradient_norm(
                    model.backbone.parameters(), reference),
            }
        if getattr(erc, 'design', '') == 'branch_aligned_tristate_multiaction_erc_v5':
            return {
                'Gradient/action_head_norm': self._gradient_norm(
                    erc.action_head.parameters(), reference),
                'Gradient/dense_encoder_norm': self._gradient_norm(
                    erc.encoder.parameters(), reference),
                'Gradient/erc_total_norm': self._gradient_norm(
                    erc.parameters(), reference),
                'Gradient/head_norm': self._gradient_norm(
                    (p for n, p in model.box_head.named_parameters()
                     if not n.startswith('erc.')), reference),
                'Gradient/backbone_norm': self._gradient_norm(
                    model.backbone.parameters(), reference),
            }
        if getattr(erc, 'design', '') in (
            'branch_aligned_factorized_reachability_erc_v6',
            'branch_aligned_iou_utility_erc_v7',
        ):
            return {
                'Gradient/permission_head_norm': self._gradient_norm(
                    erc.permission_head.parameters(), reference),
                'Gradient/direction_head_norm': self._gradient_norm(
                    erc.direction_head.parameters(), reference),
                'Gradient/scope_head_norm': self._gradient_norm(
                    erc.scope_head.parameters(), reference),
                'Gradient/magnitude_head_norm': self._gradient_norm(
                    erc.magnitude_head.parameters(), reference),
                'Gradient/dense_encoder_norm': self._gradient_norm(
                    erc.encoder.parameters(), reference),
                'Gradient/erc_total_norm': self._gradient_norm(
                    erc.parameters(), reference),
                'Gradient/head_norm': self._gradient_norm(
                    (p for n, p in model.box_head.named_parameters()
                     if not n.startswith('erc.')), reference),
                'Gradient/backbone_norm': self._gradient_norm(
                    model.backbone.parameters(), reference),
            }
        if getattr(erc, 'design', '') in (
                'branch_aligned_action_value_erc_av1',
                'branch_aligned_action_value_erc_av1b_balanced',
                'branch_aligned_temporal_consensus_erc_av2',
                'cross_modal_consensus_residual_erc_av3',
                'cross_domain_normalized_consensus_erc_av4'):
            return {
                'Gradient/utility_head_norm': self._gradient_norm(
                    erc.utility_head.parameters(), reference),
                'Gradient/dense_encoder_norm': self._gradient_norm(
                    erc.encoder.parameters(), reference),
                'Gradient/erc_total_norm': self._gradient_norm(
                    erc.parameters(), reference),
                'Gradient/head_norm': self._gradient_norm(
                    (p for n, p in model.box_head.named_parameters()
                     if not n.startswith('erc.')), reference),
                'Gradient/backbone_norm': self._gradient_norm(
                    model.backbone.parameters(), reference),
            }
        if getattr(erc, 'design', '') == (
                'cross_domain_global_regional_reliability_erc_v1'):
            return {
                'Gradient/reliability_head_norm': self._gradient_norm(
                    erc.reliability_head.parameters(), reference),
                'Gradient/gate_head_norm': self._gradient_norm(
                    erc.gate_head.parameters(), reference),
                'Gradient/strength_head_norm': self._gradient_norm(
                    erc.strength_head.parameters(), reference),
                'Gradient/regional_mix_head_norm': self._gradient_norm(
                    erc.regional_mix_head.parameters(), reference),
                'Gradient/dense_encoder_norm': self._gradient_norm(
                    erc.encoder.parameters(), reference),
                'Gradient/erc_total_norm': self._gradient_norm(
                    erc.parameters(), reference),
                'Gradient/head_norm': self._gradient_norm(
                    (p for n, p in model.box_head.named_parameters()
                     if not n.startswith('erc.')), reference),
                'Gradient/backbone_norm': self._gradient_norm(
                    model.backbone.parameters(), reference),
            }
        if getattr(erc, 'design', '') == (
                'cross_domain_continuous_oracle_erc_av5'):
            return {
                'Gradient/reliability_head_norm': self._gradient_norm(
                    erc.reliability_head.parameters(), reference),
                'Gradient/control_head_norm': self._gradient_norm(
                    erc.control_head.parameters(), reference),
                'Gradient/dense_encoder_norm': self._gradient_norm(
                    erc.encoder.parameters(), reference),
                'Gradient/erc_total_norm': self._gradient_norm(
                    erc.parameters(), reference),
                'Gradient/head_norm': self._gradient_norm(
                    (p for n, p in model.box_head.named_parameters()
                     if not n.startswith('erc.')), reference),
                'Gradient/backbone_norm': self._gradient_norm(
                    model.backbone.parameters(), reference),
            }
        if getattr(erc, 'design', '') in (
            'target_reference_erc_trc_v1',
            'target_reference_erc_trc_v2_decision_aligned',
        ):
            return {
                'Gradient/shared_encoder_norm': self._gradient_norm(erc.encoder.parameters(), reference),
                'Gradient/reliability_head_norm': self._gradient_norm(erc.reliability.parameters(), reference),
                'Gradient/erc_total_norm': self._gradient_norm(erc.parameters(), reference),
                'Gradient/head_norm': self._gradient_norm(
                    (p for n, p in model.box_head.named_parameters() if not n.startswith('erc.')), reference),
                'Gradient/backbone_norm': self._gradient_norm(model.backbone.parameters(), reference),
            }
        if getattr(erc, 'design', '') == 'target_reference_erc_trc_v3_decision_selective':
            return {
                'Gradient/shared_encoder_norm': self._gradient_norm(erc.encoder.parameters(), reference),
                'Gradient/action_head_norm': self._gradient_norm(erc.action_head.parameters(), reference),
                'Gradient/spatial_gate_norm': self._gradient_norm(erc.spatial_gate.parameters(), reference),
                'Gradient/erc_total_norm': self._gradient_norm(erc.parameters(), reference),
                'Gradient/head_norm': self._gradient_norm(
                    (p for n, p in model.box_head.named_parameters() if not n.startswith('erc.')), reference),
                'Gradient/backbone_norm': self._gradient_norm(model.backbone.parameters(), reference),
            }
        if getattr(erc, 'design', '') == 'target_reference_erc_trc_v4_hard_decision':
            return {
                'Gradient/shared_encoder_norm': self._gradient_norm(erc.encoder.parameters(), reference),
                'Gradient/action_head_norm': self._gradient_norm(erc.action_head.parameters(), reference),
                'Gradient/strength_head_norm': self._gradient_norm(erc.strength_head.parameters(), reference),
                'Gradient/erc_total_norm': self._gradient_norm(erc.parameters(), reference),
                'Gradient/head_norm': self._gradient_norm(
                    (p for n, p in model.box_head.named_parameters() if not n.startswith('erc.')), reference),
                'Gradient/backbone_norm': self._gradient_norm(model.backbone.parameters(), reference),
            }
        if getattr(erc, 'design', '') == 'target_reference_erc_trc_v4b_factorized':
            return {
                'Gradient/shared_encoder_norm': self._gradient_norm(erc.encoder.parameters(), reference),
                'Gradient/permission_head_norm': self._gradient_norm(erc.permission_head.parameters(), reference),
                'Gradient/direction_head_norm': self._gradient_norm(erc.direction_head.parameters(), reference),
                'Gradient/strength_head_norm': self._gradient_norm(erc.strength_head.parameters(), reference),
                'Gradient/erc_total_norm': self._gradient_norm(erc.parameters(), reference),
                'Gradient/head_norm': self._gradient_norm(
                    (p for n, p in model.box_head.named_parameters() if not n.startswith('erc.')), reference),
                'Gradient/backbone_norm': self._gradient_norm(model.backbone.parameters(), reference),
            }
        if getattr(erc, 'design', '') in (
            'target_reference_erc_trc_v6_joint_action_utility',
            'target_reference_erc_trc_v6b_balanced_joint_action',
        ):
            return {
                'Gradient/shared_encoder_norm': self._gradient_norm(erc.encoder.parameters(), reference),
                'Gradient/utility_head_norm': self._gradient_norm(erc.utility_head.parameters(), reference),
                'Gradient/branch_strength_head_norm': self._gradient_norm(
                    erc.branch_strength_head.parameters(), reference),
                'Gradient/erc_total_norm': self._gradient_norm(erc.parameters(), reference),
                'Gradient/head_norm': self._gradient_norm(
                    (p for n, p in model.box_head.named_parameters() if not n.startswith('erc.')), reference),
                'Gradient/backbone_norm': self._gradient_norm(model.backbone.parameters(), reference),
            }
        if getattr(erc, 'design', '') in (
            'target_reference_erc_trc_v5_calibrated_selective',
            'target_reference_erc_trc_v5b_action_calibrated',
            'target_reference_erc_trc_v5c_deployment_aligned',
            'target_reference_erc_trc_v5d_decoupled_balanced',
            'target_reference_erc_trc_v5f_direction_calibrated',
            'target_reference_erc_trc_v5g_branch_conditional_permission',
            'target_reference_erc_trc_v5h_positive_anchored_permission',
        ):
            return {
                'Gradient/shared_encoder_norm': self._gradient_norm(erc.encoder.parameters(), reference),
                'Gradient/permission_head_norm': self._gradient_norm(erc.permission_head.parameters(), reference),
                'Gradient/direction_head_norm': self._gradient_norm(erc.direction_head.parameters(), reference),
                'Gradient/strength_head_norm': self._gradient_norm(erc.strength_head.parameters(), reference),
                **({
                    'Gradient/branch_permission_head_norm': self._gradient_norm(
                        erc.branch_permission_head.parameters(), reference
                    )
                } if hasattr(erc, 'branch_permission_head') else {}),
                'Gradient/erc_total_norm': self._gradient_norm(erc.parameters(), reference),
                'Gradient/head_norm': self._gradient_norm(
                    (p for n, p in model.box_head.named_parameters() if not n.startswith('erc.')), reference),
                'Gradient/backbone_norm': self._gradient_norm(model.backbone.parameters(), reference),
            }
        if getattr(erc, 'design', '') in (
            'response_branch_spatial_gate_v13',
            'response_intervention_gate_v14',
            'response_continuous_worthiness_gate_v15',
            'response_asymmetric_worthiness_gate_v16',
            'response_selective_abstention_worthiness_gate_v17',
            'response_preservation_veto_gate_v18',
        ):
            diagnostics = {
                'Gradient/shared_encoder_norm': self._gradient_norm(
                    list(erc.encoder.parameters())
                    + list(erc.global_projection.parameters())
                    + list(erc.decision_projection.parameters()), reference
                ),
                'Gradient/global_alpha_head_norm': self._gradient_norm(
                    erc.beta_head.parameters(), reference
                ),
                'Gradient/regional_alpha_head_norm': self._gradient_norm(
                    erc.center_head.parameters(), reference
                ),
                'Gradient/gate_head_norm': self._gradient_norm(
                    erc.sigma_head.parameters(), reference
                ),
                'Gradient/head_norm': self._gradient_norm(
                    (
                        parameter
                        for name, parameter in model.box_head.named_parameters()
                        if not name.startswith('erc.')
                    ),
                    reference,
                ),
                'Gradient/erc_total_norm': self._gradient_norm(
                    erc.parameters(), reference
                ),
                'Gradient/backbone_norm': self._gradient_norm(
                    model.backbone.parameters(), reference
                ),
            }
            if hasattr(erc, 'intervention_head'):
                diagnostics['Gradient/intervention_head_norm'] = (
                    self._gradient_norm(
                        erc.intervention_head.parameters(), reference
                    )
                )
            if hasattr(erc, 'permission_head'):
                diagnostics['Gradient/permission_head_norm'] = (
                    self._gradient_norm(
                        erc.permission_head.parameters(), reference
                    )
                )
            if hasattr(erc, 'worthiness_head'):
                diagnostics['Gradient/worthiness_head_norm'] = (
                    self._gradient_norm(
                        erc.worthiness_head.parameters(), reference
                    )
                )
            if hasattr(erc, 'preservation_head'):
                diagnostics['Gradient/preservation_head_norm'] = (
                    self._gradient_norm(
                        erc.preservation_head.parameters(), reference
                    )
                )
            return diagnostics
        if getattr(erc, 'design', '') in (
            'response_local_spatial_v11',
            'response_local_reliability_v12',
        ):
            zero = reference.new_zeros(())
            diagnostics = {
                'Gradient/shared_encoder_norm': self._gradient_norm(
                    list(erc.local_encoder.parameters())
                    + list(erc.local_projection.parameters()), reference
                ),
                'Gradient/global_alpha_head_norm': zero,
                'Gradient/regional_alpha_head_norm': self._gradient_norm(
                    erc.local_residual_head.parameters(), reference
                ),
                'Gradient/gate_head_norm': self._gradient_norm(
                    erc.local_reliability_head.parameters(), reference
                ) if hasattr(erc, 'local_reliability_head') else zero,
                'Gradient/head_norm': self._gradient_norm(
                    (
                        parameter
                        for name, parameter in model.box_head.named_parameters()
                        if not name.startswith('erc.')
                    ),
                    reference,
                ),
                'Gradient/erc_total_norm': self._gradient_norm(
                    erc.parameters(), reference
                ),
                'Gradient/backbone_norm': self._gradient_norm(
                    model.backbone.parameters(), reference
                ),
            }
            return diagnostics
        if getattr(erc, 'design', '') == (
                'peak_quality_independent_branch_erc_pq3'):
            return {
                'Gradient/shared_encoder_norm': self._gradient_norm(
                    erc.encoder.parameters(), reference
                ),
                'Gradient/quality_head_norm': reference.new_zeros(()),
                'Gradient/global_alpha_head_norm': self._gradient_norm(
                    erc.control_head.parameters(), reference
                ),
                'Gradient/regional_alpha_head_norm': reference.new_zeros(()),
                'Gradient/gate_head_norm': reference.new_zeros(()),
                'Gradient/head_norm': self._gradient_norm(
                    (
                        parameter
                        for name, parameter in model.box_head.named_parameters()
                        if not name.startswith('erc.')
                    ),
                    reference,
                ),
                'Gradient/erc_total_norm': self._gradient_norm(
                    erc.parameters(), reference
                ),
                'Gradient/backbone_norm': self._gradient_norm(
                    model.backbone.parameters(), reference
                ),
            }
        if getattr(erc, 'design', '') == (
                'peak_ambiguity_target_verifier_erc_pq4'):
            return {
                'Gradient/shared_encoder_norm': self._gradient_norm(
                    erc.encoder.parameters(), reference
                ),
                'Gradient/quality_head_norm': reference.new_zeros(()),
                'Gradient/global_alpha_head_norm': self._gradient_norm(
                    erc.verifier_head.parameters(), reference
                ),
                'Gradient/regional_alpha_head_norm': reference.new_zeros(()),
                'Gradient/gate_head_norm': reference.new_zeros(()),
                'Gradient/head_norm': self._gradient_norm(
                    (
                        parameter
                        for name, parameter in model.box_head.named_parameters()
                        if not name.startswith('erc.')
                    ),
                    reference,
                ),
                'Gradient/erc_total_norm': self._gradient_norm(
                    erc.parameters(), reference
                ),
                'Gradient/backbone_norm': self._gradient_norm(
                    model.backbone.parameters(), reference
                ),
            }
        if getattr(erc, 'design', '') in (
                'peak_quality_calibrated_erc_pq1',
                'peak_quality_decision_aligned_erc_pq2'):
            gate_parameters = (
                erc.gate_head.parameters()
                if hasattr(erc, 'gate_head') else [])
            return {
                'Gradient/shared_encoder_norm': self._gradient_norm(
                    erc.encoder.parameters(), reference
                ),
                'Gradient/quality_head_norm': self._gradient_norm(
                    erc.quality_head.parameters(), reference
                ),
                'Gradient/global_alpha_head_norm': self._gradient_norm(
                    erc.quality_head.parameters(), reference
                ),
                'Gradient/regional_alpha_head_norm': reference.new_zeros(()),
                'Gradient/gate_head_norm': self._gradient_norm(
                    gate_parameters, reference),
                'Gradient/head_norm': self._gradient_norm(
                    (
                        parameter
                        for name, parameter in model.box_head.named_parameters()
                        if not name.startswith('erc.')
                    ),
                    reference,
                ),
                'Gradient/erc_total_norm': self._gradient_norm(
                    erc.parameters(), reference
                ),
                'Gradient/backbone_norm': self._gradient_norm(
                    model.backbone.parameters(), reference
                ),
            }
        if getattr(erc, 'design', '') == 'response_competition_pair_v10':
            regional_parameters = list(erc.pair_encoder.parameters()) + list(
                erc.pair_beta_head.parameters()
            )
        else:
            regional_parameters = list(
                erc.regional_encoder.parameters()
            ) + list(erc.regional_output.parameters())
        diagnostics = {
            'Gradient/shared_encoder_norm': self._gradient_norm(
                erc.global_encoder.parameters(), reference
            ),
            'Gradient/global_alpha_head_norm': self._gradient_norm(
                erc.global_alpha_head.parameters(), reference
            ),
            'Gradient/regional_alpha_head_norm': self._gradient_norm(
                regional_parameters, reference
            ),
            'Gradient/gate_head_norm': self._gradient_norm(
                erc.gate_head.parameters(), reference
            ),
            'Gradient/head_norm': self._gradient_norm(
                (
                    parameter
                    for name, parameter in model.box_head.named_parameters()
                    if not name.startswith('erc.')
                ),
                reference,
            ),
            'Gradient/erc_total_norm': self._gradient_norm(
                erc.parameters(), reference
            ),
            'Gradient/backbone_norm': self._gradient_norm(
                model.backbone.parameters(), reference
            ),
        }
        if getattr(erc, 'design', '') == 'response_competition_pair_v10':
            diagnostics['Gradient/regional_gate_head_norm'] = (
                self._gradient_norm(
                    erc.pair_gate_head.parameters(), reference
                )
            )
        elif hasattr(erc, 'regional_gate_head'):
            diagnostics['Gradient/regional_gate_head_norm'] = (
                self._gradient_norm(
                    erc.regional_gate_head.parameters(), reference
                )
            )
        if status is not None:
            gate_mean = status.get(
                'ERC/regional_gate_mean', status.get('ERC/gate_mean')
            )
            if gate_mean is not None:
                gate_value = float(gate_mean.detach())
                regional_value = float(
                    diagnostics['Gradient/regional_alpha_head_norm']
                )
                if gate_value < 0.1 and regional_value <= 1e-12:
                    print(
                        'WARNING: Gate is unexpectedly suppressing regional '
                        'alpha learning.'
                    )
        return diagnostics

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # process the groundtruth
        n, b, hc, wc = data['search_cls'].shape
        gt_bboxes = data['search_anno'].reshape(n*b, -1)  # (Ns, batch, 4) (x1,y1,w,h)
        gt_cls = data['search_cls'].reshape(n*b, hc, wc)  # (Ns, batch, 4) (x1,y1,w,h)
        
        # compute losses
        size = data['search_images'].shape[-1]//16
        loss, status = self.compute_losses(
            out_dict,
            gt_bboxes,
            gt_cls,
            self.cont_gt(gt_bboxes, size),
            epoch=data.get('epoch', 1),
            num_search_frames=n,
            dataset_ids=(
                data['dataset_id'].repeat(n, 1, 1).reshape(n * b)
                if 'dataset_id' in data else None
            ),
        )
        if self.cfg.MODEL.RQC.ENABLED:
            rqc_loss, rqc_status = self.compute_rqc_loss(
                out_dict, gt_bboxes, gt_cls
            )
            loss = loss + self.cfg.TRAIN.RQC_LOSS_WEIGHT * rqc_loss
            status['Loss/total'] = loss
            status.update(rqc_status)

        return loss, status

    def compute_rqc_loss(self, pred_dict, gt_bboxes, gt_cls):
        """Train RQC on VL tracking proposals; Head keeps its original losses."""
        from lib.models.heads.candidate_quality_v3 import candidates
        if self.cfg.MODEL.RQC.DESIGN == 'candidate_quality_jq':
            from lib.models.heads.candidate_quality_jq import candidates, training_loss
        elif self.cfg.MODEL.RQC.DESIGN == 'candidate_quality_bq':
            from lib.models.heads.candidate_quality_bq import candidates, training_loss
        elif self.cfg.MODEL.RQC.DESIGN == 'candidate_quality_pairgain':
            from lib.models.heads.candidate_quality_pairgain import candidates, training_loss
        elif self.cfg.MODEL.RQC.DESIGN == 'candidate_quality_pr':
            from lib.models.heads.candidate_quality_pr import candidates, training_loss
        elif self.cfg.MODEL.RQC.DESIGN == 'candidate_quality_pd':
            from lib.models.heads.candidate_quality_pd import candidates, training_loss
        elif self.cfg.MODEL.RQC.DESIGN == 'candidate_quality_pc':
            from lib.models.heads.candidate_quality_pc import candidates, training_loss
        elif self.cfg.MODEL.RQC.DESIGN == 'candidate_quality_jc':
            from lib.models.heads.candidate_quality_jc import candidates, training_loss
        elif self.cfg.MODEL.RQC.DESIGN == 'candidate_quality_v5_advantage':
            from lib.models.heads.candidate_quality_v5_advantage import training_loss
        elif self.cfg.MODEL.RQC.DESIGN == 'candidate_quality_v5':
            from lib.models.heads.candidate_quality_v5 import training_loss
        elif self.cfg.MODEL.RQC.DESIGN == 'candidate_quality_r1':
            from lib.models.heads.candidate_quality_r1 import training_loss
        elif self.cfg.MODEL.RQC.DESIGN == 'candidate_quality_v4':
            from lib.models.heads.candidate_quality_v4 import training_loss
        else:
            from lib.models.heads.candidate_quality_v3 import training_loss
        model = self.net.module if hasattr(self.net, 'module') else self.net
        active = pred_dict['flag'].reshape(-1).eq(2)
        zero = pred_dict['pred_boxes'].sum() * 0.0
        status = {'Loss/rqc_total': zero.detach(),
                  'Loss/rqc_quality': zero.detach(),
                  'Loss/rqc_decision': zero.detach(),
                  'RQC/active_samples': active.sum().detach().float(),
                  'RQC/selection_change_rate': zero.detach()}
        if not active.any():
            return zero, status
        fields = ('search', 'template', 'txt_token', 'prompts',
                  'cls_score_test', 'cont_score', 'bbox_map')
        subset = {key: pred_dict[key][active] for key in fields}
        proposals = candidates(subset, pred_dict['template_mask'][active],
                               k=self.cfg.MODEL.RQC.TOPK)
        prediction = model.rqc(proposals)
        with torch.no_grad():
            target = self.candidate_iou(proposals['boxes'], gt_bboxes[active])
        if self.cfg.MODEL.RQC.DESIGN == 'candidate_quality_v5':
            with torch.no_grad():
                identity_target = gt_cls[active].flatten(1).gather(
                    1, proposals['indices']
                )
            rqc_loss, parts = training_loss(
                prediction, target, identity_target,
                temperature=self.cfg.TRAIN.RQC_TEMPERATURE,
            )
        else:
            rqc_loss, parts = training_loss(
                prediction, target, temperature=self.cfg.TRAIN.RQC_TEMPERATURE)
        status.update({
            'Loss/rqc_total': rqc_loss.detach(),
            'Loss/rqc_quality': parts['quality'],
            'Loss/rqc_decision': parts['decision'],
            'RQC/selection_change_rate':
                prediction['decision_score'].argmax(-1).ne(0).float().mean().detach(),
        })
        if self.cfg.MODEL.RQC.DESIGN in (
                'candidate_quality_v4', 'candidate_quality_v5',
                'candidate_quality_v5_advantage'):
            status.update({
                'Loss/rqc_permission': parts['permission'],
                'Loss/rqc_hard_harm': parts['hard_harm'],
                'RQC/selected_iou_gain': parts['selected_iou_gain'],
                'RQC/harmful_change_rate': parts['harmful_change_rate'],
                'RQC/beneficial_change_rate': parts['beneficial_change_rate'],
                'RQC/permission_mean': parts['permission_mean'],
            })
        if self.cfg.MODEL.RQC.DESIGN == 'candidate_quality_v5':
            status.update({
                'Loss/erc_identity': parts['identity'],
                'Loss/erc_identity_bce': parts['identity_bce'],
                'Loss/erc_identity_rank': parts['identity_rank'],
                'Loss/erc_decision': parts['erc_decision'],
                'Loss/erc_hard_harm': parts['erc_hard_harm'],
                'ERC/selection_change_rate': parts['erc_change_rate'],
                'ERC/incremental_iou_gain': parts['erc_incremental_iou_gain'],
                'ERC/beneficial_change_rate':
                    parts['erc_beneficial_change_rate'],
                'ERC/harmful_change_rate': parts['erc_harmful_change_rate'],
                'ERC/final_selected_iou_gain':
                    parts['final_selected_iou_gain'],
                'ERC/reliability_mean': parts['identity_reliability_mean'],
                'ERC/residual_abs': parts['erc_residual_abs'],
            })
        if self.cfg.MODEL.RQC.DESIGN == 'candidate_quality_v5_advantage':
            status.update({
                'Loss/rqc_advantage': parts['advantage'],
                'Loss/rqc_advantage_regression': parts['advantage_regression'],
                'Loss/rqc_advantage_direction': parts['advantage_direction'],
                'Loss/rqc_final_decision': parts['final_decision'],
                'Loss/rqc_final_hard_harm': parts['final_hard_harm'],
                'RQC/advantage_mae': parts['advantage_mae'],
                'RQC/advantage_target_abs': parts['advantage_target_abs'],
                'RQC/advantage_residual_abs': parts['advantage_residual_abs'],
                'RQC/final_selected_iou_gain': parts['final_selected_iou_gain'],
                'RQC/final_changed_iou_gain': parts['final_changed_iou_gain'],
                'RQC/final_beneficial_change_rate':
                    parts['final_beneficial_change_rate'],
                'RQC/final_harmful_change_rate':
                    parts['final_harmful_change_rate'],
            })
        if self.cfg.MODEL.RQC.DESIGN == 'candidate_quality_pr':
            for key in ('selected_iou_gain', 'changed_iou_gain', 'harmful_change_rate',
                        'beneficial_change_rate', 'severe_harm_rate', 'positive_gain',
                        'negative_gain', 'score_abs', 'selected_regret', 'oracle_gain',
                        'margin_violation_rate', 'residual_abs', 'prior_abs',
                        'selected_prior_gap', 'selected_residual'):
                status['RQC/' + key] = parts[key]
        if self.cfg.MODEL.RQC.DESIGN == 'candidate_quality_pairgain':
            for key in ('gain_mae', 'selected_iou_gain', 'harmful_change_rate',
                        'beneficial_change_rate', 'severe_harm_rate', 'predicted_gain_abs'):
                status['RQC/' + key] = parts[key]
        if self.cfg.MODEL.RQC.DESIGN in ('candidate_quality_bq', 'candidate_quality_jq'):
            for key in ('quality_mae', 'selected_iou_gain', 'harmful_change_rate',
                        'beneficial_change_rate', 'severe_harm_rate', 'boundary_valid_rate'):
                status['RQC/' + key] = parts[key]
        if self.cfg.MODEL.RQC.DESIGN == 'candidate_quality_pd':
            for key in ('selected_regret', 'oracle_gain', 'margin_violation_rate'):
                status['RQC/' + key] = parts[key]
        if self.cfg.MODEL.RQC.DESIGN in ('candidate_quality_pc', 'candidate_quality_pd'):
            for key in ('selected_iou_gain', 'changed_iou_gain', 'harmful_change_rate',
                        'beneficial_change_rate', 'severe_harm_rate', 'positive_gain',
                        'negative_gain', 'score_abs', 'saturation_rate'):
                status['RQC/' + key] = parts[key]
        if self.cfg.MODEL.RQC.DESIGN == 'candidate_quality_jc':
            for key in ('selected_iou_gain','changed_iou_gain','harmful_change_rate',
                        'beneficial_change_rate','severe_harm_rate','quality_mae','residual_abs'):
                status['RQC/' + key] = parts[key]
        if self.cfg.MODEL.RQC.DESIGN == 'candidate_quality_r1':
            for key in ('utility', 'hard_harm'):
                status['Loss/rqc_' + key] = parts[key]
            for key in ('selected_iou_gain', 'changed_iou_gain', 'harmful_change_rate',
                        'beneficial_change_rate', 'severe_harm_rate', 'utility_mae', 'utility_abs'):
                status['RQC/' + key] = parts[key]
        if self.cfg.TRAIN.RQC_HEAD_ALIGNMENT:
            if self.cfg.MODEL.RQC.DESIGN != 'candidate_quality_v4':
                raise ValueError('Head quality alignment requires RQC-v4')
            from lib.train.actors.rqc_head_quality_alignment import auxiliary_loss
            auxiliary, metrics = auxiliary_loss(
                model, subset, pred_dict['template_mask'][active],
                gt_cls[active], gt_bboxes[active], prediction, target,
                self.candidate_iou, self.cfg, expand=model.training)
            rqc_loss = rqc_loss + auxiliary
            status.update(metrics)
            status['Loss/rqc_total'] = rqc_loss.detach()
        return rqc_loss, status

    def forward_pass(self, data):
        _, b, _, ht, wt = data['template_images'].shape
        n, b, _, hs, ws = data['search_images'].shape
        template_images = data['template_images'].repeat(n, 1, 1, 1, 1).reshape(n*b, 3, ht, wt)
        template_anno = data['template_anno'].repeat(n, 1, 1).reshape(n*b, -1)
        search_images = data['search_images'].reshape(n*b, 3, hs, ws)
        search_anno = data['search_anno'].reshape(n*b, -1)
        text = data['text'].reshape(n*b, -1)
        text_mask = data['text_mask'].reshape(n*b, -1)
        text = NestedTensor(text, text_mask)
        template_mask = self.anno2mask(template_anno, wt//16)
        context_mask = self.anno2mask(search_anno, ws//16, reverse=True)
        flag = data['flag'].repeat(n, 1, 1).reshape(n*b, -1)
        out_dict = self.net(template_images, search_images, text, template_mask, context_mask, flag)
        if self._head_teacher is not None:
            # Reuse the frozen-backbone tensors and run only the small official
            # Head copy.  The teacher dictionary is isolated because Head.forward
            # updates its input mapping in-place.
            with torch.no_grad():
                teacher_dict = self._head_teacher(dict(out_dict))
                teacher_cls = teacher_dict['cls_score_test'].detach()
                teacher_cont_logits = teacher_dict['cont_score']
                if self._response_erc_design in (
                        'foreground_reference_evidence_erc_fr1',
                        'branch_aligned_tristate_multiaction_erc_v5',
                        'branch_aligned_factorized_reachability_erc_v6',
                        'branch_aligned_iou_utility_erc_v7',
                        'branch_aligned_action_value_erc_av1',
                        'branch_aligned_action_value_erc_av1b_balanced',
                        'branch_aligned_temporal_consensus_erc_av2',
                        'cross_modal_consensus_residual_erc_av3',
                        'cross_domain_normalized_consensus_erc_av4',
                        'cross_domain_global_regional_reliability_erc_v1',
                        'cross_domain_continuous_oracle_erc_av5',
                        'peak_quality_calibrated_erc_pq1',
                        'peak_quality_decision_aligned_erc_pq2',
                        'peak_quality_independent_branch_erc_pq3',
                        'peak_ambiguity_target_verifier_erc_pq4',
                        'target_reference_direct_dense_erc_dr1',
                        'target_reference_trajectory_safe_dense_erc_dr2',
                        'target_reference_realized_action_dense_erc_dr3'):
                    if teacher_cont_logits.shape[-1] == 2:
                        teacher_cont_logits = torch.cat([
                            teacher_cont_logits,
                            torch.zeros_like(teacher_cont_logits[..., :1]),
                        ], dim=-1)
                    elif teacher_cont_logits.shape[-1] != 3:
                        raise ValueError('FER-v5 teacher expects 2 or 3 contrastive logits')
                teacher_cont = teacher_cont_logits.softmax(-1)[..., 0].reshape_as(
                    teacher_cls).detach()
                teacher_base = (
                    teacher_cls.flatten(1) * teacher_cont.flatten(1)
                )
                hann = self._head_teacher.erc_match_test_hann
                if hann:
                    window = self._model().box_head.erc.hann_window
                    teacher_base = teacher_base * window
                    if self._response_erc_design == 'foreground_reference_evidence_erc_fr1':
                        teacher_base = teacher_cls.flatten(1).double() * window * teacher_cont.flatten(1).double()
                out_dict['head_teacher_cls_response'] = teacher_cls
                out_dict['head_teacher_cont_response'] = teacher_cont
                out_dict['head_teacher_windowed_base_score'] = teacher_base
        # out_dict: (B, N, C), outputs_coord: (1, B, N, C), target_query: (1, B, N, C)
        return out_dict

    def _head_response_distill_terms(self, pred_dict):
        """Keep the trainable Head's C/T responses near the official Head.

        This does not freeze the Head: the normal UVLTrack losses still update
        it.  The KL term only prevents a small parameter drift from producing a
        much larger hard-peak margin drift that drowns ERC at inference.
        """
        reference = pred_dict['pred_boxes']
        zero = reference.sum() * 0.0
        status = {
            'Loss/head_response_distill': zero.detach(),
            'Head/response_distill_cls': zero.detach(),
            'Head/response_distill_cont': zero.detach(),
            'Head/response_distill_base': zero.detach(),
        }
        if self._head_teacher is None:
            return zero, status

        active_mask = pred_dict['erc_active_mask'].reshape(-1).bool()
        if not active_mask.any():
            return zero, status

        temperature = max(
            float(self.cfg.TRAIN.HEAD_RESPONSE_DISTILL_TEMPERATURE), 1e-6
        )
        eps = max(float(self.cfg.MODEL.HEAD.ERC.EPS), 1e-12)

        def spatial_kl(student, teacher):
            student = student.reshape(student.shape[0], -1)[active_mask]
            teacher = teacher.reshape(teacher.shape[0], -1)[active_mask]
            student_logits = student.clamp_min(eps).log() / temperature
            teacher_logits = teacher.clamp_min(eps).log() / temperature
            teacher_probability = F.softmax(teacher_logits, dim=-1)
            return F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                teacher_probability,
                reduction='batchmean',
            ) * (temperature ** 2)

        cls_loss = spatial_kl(
            pred_dict['erc_cls_response'],
            pred_dict['head_teacher_cls_response'],
        )
        cont_loss = spatial_kl(
            pred_dict['erc_cont_response'],
            pred_dict['head_teacher_cont_response'],
        )
        base_loss = spatial_kl(
            pred_dict['erc_windowed_base_score'],
            pred_dict['head_teacher_windowed_base_score'],
        )
        loss = (cls_loss + cont_loss + base_loss) / 3.0
        status.update({
            'Loss/head_response_distill': loss.detach(),
            'Head/response_distill_cls': cls_loss.detach(),
            'Head/response_distill_cont': cont_loss.detach(),
            'Head/response_distill_base': base_loss.detach(),
        })
        return loss, status
    
    def cont_gt(self, gt_bboxes, size):
        bboxes = box_cxcywh_to_xyxy(box_xywh_to_cxcywh_scale(gt_bboxes, self.cfg.TRAIN.CTR_RATIO))*size
        cood = torch.arange(size).unsqueeze(0).repeat(gt_bboxes.shape[0], 1).cuda()+0.5 # b, sz
        x_mask = ((cood > bboxes[:, 0:1]) & (cood < bboxes[:, 2:3])).unsqueeze(1) # b, 1, w
        y_mask = ((cood > bboxes[:, 1:2]) & (cood < bboxes[:, 3:4])).unsqueeze(2) # b, h, 1
        mask_c = (x_mask & y_mask)

        cx = torch.floor((bboxes[:, 0]+bboxes[:, 2])/2).long()
        cy = torch.floor((bboxes[:, 1]+bboxes[:, 3])/2).long()
        bid = torch.arange(cx.shape[0]).to(cx)
        mask_c[bid, cy, cx] = True
        
        bboxes = box_cxcywh_to_xyxy(box_xywh_to_cxcywh(gt_bboxes))*size
        cood = torch.arange(size).unsqueeze(0).repeat(gt_bboxes.shape[0], 1).cuda()+0.5 # b, sz
        x_mask = ((cood > bboxes[:, 0:1]) & (cood < bboxes[:, 2:3])).unsqueeze(1) # b, 1, w
        y_mask = ((cood > bboxes[:, 1:2]) & (cood < bboxes[:, 3:4])).unsqueeze(2) # b, h, 1
        mask_t = 1-2*(x_mask & y_mask).long()
        mask_t[mask_c] = 0
        return mask_t.flatten(1)
        
    def anno2mask(self, gt_bboxes, size, reverse=False):
        bboxes = box_xywh_to_xyxy(gt_bboxes)*size # b, 4
        cood = torch.arange(size).unsqueeze(0).repeat(gt_bboxes.shape[0], 1).cuda()+0.5 # b, sz
        x_mask = ((cood > bboxes[:, 0:1]) & (cood < bboxes[:, 2:3])).unsqueeze(1) # b, 1, w
        y_mask = ((cood > bboxes[:, 1:2]) & (cood < bboxes[:, 3:4])).unsqueeze(2) # b, h, 1
        mask = (x_mask & y_mask)

        cx = torch.floor((bboxes[:, 0]+bboxes[:, 2])/2).long()
        cy = torch.floor((bboxes[:, 1]+bboxes[:, 3])/2).long()
        bid = torch.arange(cx.shape[0]).to(cx)
        mask[bid, cy, cx] = True
        
        if reverse:
            mask = torch.cat([mask[bid.shape[0]//2:], mask[:bid.shape[0]//2]], dim=0)
        return mask.flatten(1)
        
    def sample_negative(self, logits, gt_bboxes, size):
        bboxes = gt_bboxes # b, 4
        cood_1d = (torch.arange(size)+0.5) / size
        cood = cood_1d.unsqueeze(0).repeat(gt_bboxes.shape[0], 1).cuda() # b, sz
        x_mask = ((cood > bboxes[:, 0:1]) & (cood < bboxes[:, 2:3])).unsqueeze(1) # b, 1, w
        y_mask = ((cood > bboxes[:, 1:2]) & (cood < bboxes[:, 3:4])).unsqueeze(2) # b, h, 1
        mask = (x_mask & y_mask) # b, h, w
        mask = (mask.reshape(gt_bboxes.shape[0], -1))*(-1e9) # background == 1
        sample_logits = torch.sort(logits.reshape(gt_bboxes.shape[0], -1)+mask, descending=True, dim=-1).values[:, :9]
        return sample_logits
        
    def contractive_learning(self, logits, gt_bbox):  # b, n, sz, sz
        b, n, sz, sz = logits.shape
        logits = logits.reshape(-1, 1, sz, sz)
        gt_bbox = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, n, 1)).view(-1, 4).clamp(min=0.0, max=1.0)
        ctr = (gt_bbox[:, :2] + gt_bbox[:, 2:]).reshape(b*n, 1, 1, 2) / 2
        neg_logits = self.sample_negative(logits, gt_bbox, sz).to(logits)
        sample_points = ctr * 2 - 1
        pos_logits = F.grid_sample(logits, sample_points, padding_mode="border", align_corners=True).reshape(b*n, -1) # b, 1, 1, 10
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        target = torch.zeros(b*n).to(gt_bbox.device).long()
        return logits, target # check
        

    def compute_losses(
            self, pred_dict, gt_bbox, gt_cls, gt_cont, epoch=1,
            num_search_frames=1, dataset_ids=None):
        # Get boxes
        pred_boxes = pred_dict['pred_boxes'] # b, 4, s
        pred_cls = pred_dict['cls_score']
        pred_cont = pred_dict['cont_score']
        B = pred_cont.shape[0]
        aux_loss = pred_boxes.new_zeros(())
        if self.loss_weight['aux'] > 0:
            pred_logits, target = self.contractive_learning(pred_dict['logits'], gt_bbox)
            aux_loss = self.objective['aux'](pred_logits, target)
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        cls_loss = self.objective['cls'](pred_cls, gt_cls)
        cont_loss = self.objective['cont'](pred_cont.reshape(-1, 2), gt_cont.reshape(-1))
        erc_loss = pred_boxes.new_zeros(())
        erc_status = {}
        if self.cfg.MODEL.HEAD.ERC.ENABLED:
            erc_loss, erc_status = self.compute_erc_losses(
                pred_dict, gt_bbox, gt_cls, epoch=epoch,
                num_search_frames=num_search_frames,
                dataset_ids=dataset_ids,
            )
        head_anchor_loss = pred_boxes.new_zeros(())
        head_status = {}
        if self._is_response_v7:
            head_anchor_loss, head_status = self._head_anchor_terms()
        head_response_distill_loss, head_distill_status = (
            self._head_response_distill_terms(pred_dict)
            if self._is_response_v7
            else (pred_boxes.new_zeros(()), {})
        )
        head_peak_loss = pred_boxes.sum() * 0.0
        head_peak_status = {}
        if self.cfg.TRAIN.HEAD_PEAK_PRESERVE_WEIGHT > 0:
            from .erc_dr4a_continuous_supervision import teacher_peak_preservation
            head_peak_loss, head_peak_status = teacher_peak_preservation(
                pred_dict, gt_bbox, self.cfg)
        # weighted sum
        loss =  self.loss_weight['giou'] * giou_loss + \
                    self.loss_weight['l1'] * l1_loss + \
                        self.loss_weight['cls'] * cls_loss + \
                            self.loss_weight['aux'] * aux_loss + \
                        self.loss_weight['cont'] * cont_loss + erc_loss + \
                            float(self.cfg.TRAIN.HEAD_ANCHOR_WEIGHT) * head_anchor_loss + \
                            float(self.cfg.TRAIN.HEAD_RESPONSE_DISTILL_WEIGHT) * head_response_distill_loss + \
                            float(self.cfg.TRAIN.HEAD_PEAK_PRESERVE_WEIGHT) * head_peak_loss
        
        # status for log
        mean_iou = iou.detach().mean()
        status = {"Loss/total": loss,
                    "Loss/giou": giou_loss,
                    "Loss/l1": l1_loss,
                    "Loss/cls": cls_loss,
                    "Loss/aux": aux_loss,
                    "Loss/cont": cont_loss,
                    "IoU": mean_iou}
        status.update(erc_status)
        status.update(head_status)
        status.update(head_distill_status)
        status.update(head_peak_status)
        if not self.net.training:
            acc = torch.mean((iou > 0.5).float()).detach()
            status['Acc@0.5'] = acc
        return loss, status

    @staticmethod
    def candidate_iou(candidate_boxes, gt_bbox):
        """Compute aligned IoU targets for all candidate boxes without gradients."""
        candidate_xyxy = box_cxcywh_to_xyxy(candidate_boxes).clamp(0.0, 1.0)
        gt_xyxy = box_xywh_to_xyxy(gt_bbox).clamp(0.0, 1.0).unsqueeze(1)

        top_left = torch.maximum(candidate_xyxy[..., :2], gt_xyxy[..., :2])
        bottom_right = torch.minimum(candidate_xyxy[..., 2:], gt_xyxy[..., 2:])
        intersection_size = (bottom_right - top_left).clamp_min(0.0)
        intersection = intersection_size[..., 0] * intersection_size[..., 1]

        candidate_size = (candidate_xyxy[..., 2:] - candidate_xyxy[..., :2]).clamp_min(0.0)
        gt_size = (gt_xyxy[..., 2:] - gt_xyxy[..., :2]).clamp_min(0.0)
        candidate_area = candidate_size[..., 0] * candidate_size[..., 1]
        gt_area = gt_size[..., 0] * gt_size[..., 1]
        union = candidate_area + gt_area - intersection
        return intersection / union.clamp_min(1e-6)

    def compute_erc_losses(
            self, pred_dict, gt_bbox, gt_cls=None, epoch=1,
            num_search_frames=1, dataset_ids=None):
        design = getattr(self.cfg.MODEL.HEAD.ERC, 'DESIGN', 'legacy')
        if design == 'foreground_reference_evidence_erc_fr1':
            from .erc_fr1_loss import compute_fr1_loss
            return compute_fr1_loss(pred_dict, gt_bbox, self.cfg)
        if design == 'target_reference_erc_trc_v1':
            if gt_cls is None:
                raise ValueError('TRC requires the official GT center heatmap for its loss')
            return compute_trc_loss_and_metrics(pred_dict, gt_cls, gt_bbox, self.cfg)
        if design == 'target_reference_erc_trc_v2_decision_aligned':
            if gt_cls is None:
                raise ValueError('TRC-v2 requires the official GT center heatmap')
            return compute_trc_v2_loss_and_metrics(
                pred_dict, gt_cls, gt_bbox, self.cfg
            )
        if design == 'target_reference_erc_trc_v3_decision_selective':
            if gt_cls is None:
                raise ValueError('TRC-v3 requires the official GT center heatmap')
            return compute_trc_v3_loss_and_metrics(
                pred_dict, gt_cls, gt_bbox, self.cfg
            )
        if design == 'target_reference_erc_trc_v4_hard_decision':
            if gt_cls is None:
                raise ValueError('TRC-v4 requires the official GT center heatmap')
            return compute_trc_v4_loss_and_metrics(
                pred_dict, gt_cls, gt_bbox, self.cfg
            )
        if design == 'target_reference_erc_trc_v4b_factorized':
            if gt_cls is None:
                raise ValueError('TRC-v4b requires the official GT center heatmap')
            return compute_trc_v4b_loss_and_metrics(
                pred_dict, gt_cls, gt_bbox, self.cfg
            )
        if design == 'target_reference_erc_trc_v5_calibrated_selective':
            if gt_cls is None:
                raise ValueError('TRC-v5 requires the official GT center heatmap')
            return compute_trc_v5_loss_and_metrics(
                pred_dict, gt_cls, gt_bbox, self.cfg
            )
        if design == 'target_reference_erc_trc_v5b_action_calibrated':
            if gt_cls is None:
                raise ValueError('TRC-v5b requires the official GT center heatmap')
            return compute_trc_v5b_loss_and_metrics(
                pred_dict, gt_cls, gt_bbox, self.cfg
            )
        if design == 'target_reference_erc_trc_v5c_deployment_aligned':
            if gt_cls is None:
                raise ValueError('TRC-v5c requires the official GT center heatmap')
            return compute_trc_v5c_loss_and_metrics(
                pred_dict, gt_cls, gt_bbox, self.cfg
            )
        if design in (
            'target_reference_erc_trc_v5d_decoupled_balanced',
            'target_reference_erc_trc_v5f_direction_calibrated',
        ):
            if gt_cls is None:
                raise ValueError('TRC-v5d requires the official GT center heatmap')
            return compute_trc_v5d_loss_and_metrics(
                pred_dict, gt_cls, gt_bbox, self.cfg
            )
        if design == 'target_reference_erc_trc_v5g_branch_conditional_permission':
            if gt_cls is None:
                raise ValueError('TRC-v5g requires the official GT center heatmap')
            return compute_trc_v5g_loss_and_metrics(
                pred_dict, gt_cls, gt_bbox, self.cfg
            )
        if design == 'target_reference_erc_trc_v5h_positive_anchored_permission':
            if gt_cls is None:
                raise ValueError('TRC-v5h requires the official GT center heatmap')
            return compute_trc_v5h_loss_and_metrics(
                pred_dict, gt_cls, gt_bbox, self.cfg
            )
        if design == 'target_reference_erc_trc_v6_joint_action_utility':
            if gt_cls is None:
                raise ValueError('TRC-v6 requires the official GT center heatmap')
            return compute_trc_v6_loss_and_metrics(
                pred_dict, gt_cls, gt_bbox, self.cfg
            )
        if design == 'target_reference_erc_trc_v6b_balanced_joint_action':
            if gt_cls is None:
                raise ValueError('TRC-v6b requires the official GT center heatmap')
            return compute_trc_v6b_loss_and_metrics(
                pred_dict, gt_cls, gt_bbox, self.cfg
            )
        if design in (
            'reference_reliability_feature_erc_v1',
            'reference_reliability_feature_erc_v2_aligned_tokens',
        ):
            if gt_cls is None:
                raise ValueError('Feature ERC requires the official GT center heatmap')
            return compute_feature_reliability_loss_and_metrics(
                pred_dict, gt_cls, self.cfg
            )
        if design == 'reference_reliability_dense_response_erc_v3':
            if gt_cls is None:
                raise ValueError('FER-v3 requires the official GT center heatmap')
            return compute_dense_response_reliability_loss_and_metrics(
                pred_dict, gt_cls, self.cfg
            )
        if design == 'branch_aligned_dense_response_erc_v4':
            if gt_cls is None:
                raise ValueError('FER-v4 requires the official GT center heatmap')
            return compute_branch_aligned_dense_response_loss_and_metrics(
                pred_dict, gt_cls, self.cfg
            )
        if design == 'branch_aligned_tristate_multiaction_erc_v5':
            if gt_cls is None:
                raise ValueError('FER-v5 requires the official GT center heatmap')
            return compute_branch_aligned_tristate_multiaction_loss_and_metrics(
                pred_dict, gt_cls, self.cfg
            )
        if design == 'branch_aligned_factorized_reachability_erc_v6':
            if gt_cls is None:
                raise ValueError('FER-v6 requires the official GT center heatmap')
            return compute_factorized_reachability_loss_and_metrics(
                pred_dict, gt_cls, self.cfg
            )
        if design == 'branch_aligned_iou_utility_erc_v7':
            if gt_cls is None:
                raise ValueError('FER-v7 requires the official GT center heatmap')
            return compute_iou_utility_loss_and_metrics(
                pred_dict, gt_cls, gt_bbox, self.cfg
            )
        if design == 'branch_aligned_action_value_erc_av1':
            if gt_cls is None:
                raise ValueError('ERC-AV1 requires the official GT center heatmap')
            return compute_action_value_loss_and_metrics(
                pred_dict, gt_cls, gt_bbox, self.cfg
            )
        if design == 'branch_aligned_action_value_erc_av1b_balanced':
            if gt_cls is None:
                raise ValueError('ERC-AV1b requires the official GT center heatmap')
            return compute_balanced_action_value_loss_and_metrics(
                pred_dict, gt_cls, gt_bbox, self.cfg
            )
        if design in (
                'branch_aligned_temporal_consensus_erc_av2',
                'cross_modal_consensus_residual_erc_av3',
                'cross_domain_normalized_consensus_erc_av4'):
            if gt_cls is None:
                raise ValueError(
                    'temporal action-value ERC requires the official '
                    'GT center heatmap')
            return compute_temporal_consensus_action_value_loss_and_metrics(
                pred_dict, gt_cls, gt_bbox, self.cfg,
                num_search_frames=num_search_frames,
            )
        if design == 'cross_domain_global_regional_reliability_erc_v1':
            if gt_cls is None:
                raise ValueError('CD-ERC requires the official GT center heatmap')
            return compute_cross_domain_reliability_loss_and_metrics(
                pred_dict, gt_cls, self.cfg,
                num_search_frames=num_search_frames,
            )
        if design == 'cross_domain_continuous_oracle_erc_av5':
            if gt_cls is None:
                raise ValueError(
                    'ERC-AV5 requires the official GT center heatmap')
            return compute_cross_domain_continuous_oracle_loss_and_metrics(
                pred_dict, gt_cls, self.cfg,
                num_search_frames=num_search_frames,
                dataset_ids=dataset_ids,
            )
        if design == 'peak_quality_calibrated_erc_pq1':
            if gt_cls is None:
                raise ValueError(
                    'PQ-ERC requires the official GT center heatmap')
            return compute_peak_quality_calibration_loss_and_metrics(
                pred_dict, gt_cls, self.cfg, dataset_ids=dataset_ids)
        if design == 'peak_quality_decision_aligned_erc_pq2':
            if gt_cls is None:
                raise ValueError(
                    'PQ-ERC v2 requires the official GT center heatmap')
            return compute_peak_quality_decision_loss_and_metrics(
                pred_dict, gt_cls, self.cfg, dataset_ids=dataset_ids)
        if design == 'peak_quality_independent_branch_erc_pq3':
            if gt_cls is None:
                raise ValueError(
                    'PQ-ERC v3 requires the official GT center heatmap')
            return compute_peak_quality_independent_branch_loss_and_metrics(
                pred_dict, gt_cls, self.cfg, dataset_ids=dataset_ids)
        if design == 'peak_ambiguity_target_verifier_erc_pq4':
            if gt_cls is None:
                raise ValueError(
                    'PQ-ERC v4 requires the official GT center heatmap')
            return compute_peak_ambiguity_target_verifier_loss_and_metrics(
                pred_dict, gt_cls, self.cfg, dataset_ids=dataset_ids)
        if design == 'target_reference_direct_dense_erc_dr1':
            if gt_cls is None:
                raise ValueError(
                    'Direct dense ERC requires the official GT center heatmap')
            return compute_target_reference_direct_dense_loss_and_metrics(
                pred_dict, gt_cls, self.cfg, dataset_ids=dataset_ids)
        if design == 'target_reference_trajectory_safe_dense_erc_dr2':
            if gt_cls is None:
                raise ValueError(
                    'Trajectory-safe dense ERC requires the official '
                    'GT center heatmap')
            return (
                compute_target_reference_trajectory_safe_dense_loss_and_metrics(
                    pred_dict, gt_cls, self.cfg,
                    num_search_frames=num_search_frames,
                    dataset_ids=dataset_ids,
                )
            )
        if design == 'target_reference_realized_action_dense_erc_dr3':
            if gt_cls is None:
                raise ValueError(
                    'Realized-action dense ERC requires the official '
                    'GT center heatmap')
            if self.cfg.TRAIN.ERC_DR4_ENABLE:
                from .erc_deployed_hard_margin_loss_dr4 import compute_dr4_loss
                return compute_dr4_loss(
                    pred_dict, gt_cls, self.cfg,
                    num_search_frames=num_search_frames,
                    dataset_ids=dataset_ids, gt_bbox=gt_bbox)
            return (
                compute_target_reference_realized_action_dense_loss_and_metrics(
                    pred_dict, gt_cls, self.cfg,
                    num_search_frames=num_search_frames,
                    dataset_ids=dataset_ids,
                )
            )
        if design in (
            'response_branch_spatial_gate_v13',
            'response_intervention_gate_v14',
            'response_continuous_worthiness_gate_v15',
            'response_asymmetric_worthiness_gate_v16',
            'response_selective_abstention_worthiness_gate_v17',
            'response_preservation_veto_gate_v18',
        ):
            if gt_cls is None:
                raise ValueError('ERC-v13 requires the official gt_cls heatmap')
            return self.compute_erc_branch_spatial_gate_v13_losses(
                pred_dict, gt_cls, epoch=epoch
            )
        if design == 'response_local_reliability_v12':
            if gt_cls is None:
                raise ValueError('ERC-v12 requires the official gt_cls heatmap')
            return self.compute_erc_local_reliability_v12_losses(
                pred_dict, gt_cls
            )
        if design == 'response_local_spatial_v11':
            if gt_cls is None:
                raise ValueError('ERC-v11 requires the official gt_cls heatmap')
            return self.compute_erc_local_spatial_v11_losses(
                pred_dict, gt_cls
            )
        if design in (
            'response_hann_hardpeak_v8', 'response_hann_gate_v8c',
            'response_regional_effective_v8d',
            'response_precision_incremental_v9',
            'response_competition_pair_v10'
        ):
            if gt_cls is None:
                raise ValueError('ERC-v8 requires the official gt_cls heatmap')
            return self.compute_erc_response_v8_losses(
                pred_dict, gt_cls, epoch=epoch
            )
        if design == 'response_global_regional_v7':
            if gt_cls is None:
                raise ValueError('ERC-v7 requires the official gt_cls heatmap')
            return self.compute_erc_response_v7_losses(
                pred_dict, gt_cls, gt_bbox, epoch=epoch
            )
        if design == 'free_gate':
            return self.compute_erc_free_gate_losses(pred_dict, gt_bbox)
        if design == 'branch_gate':
            return self.compute_erc_branch_gate_losses(pred_dict, gt_bbox)
        return self.compute_erc_legacy_losses(pred_dict, gt_bbox)

    def compute_erc_branch_spatial_gate_v13_losses(
        self, pred_dict, gt_cls, epoch=1
    ):
        """Response-only supervision for branch-constrained ERC-v13.

        Response-only peaks from C/T/S0/+D/-D propose all possible Gaussian
        centers.  A detached GT-center response label only evaluates those
        proposals and chooses the smallest reachable |beta|, exactly matching
        the prospective Oracle audit.  Candidate boxes and IoU are never used.
        The differentiable ranking losses operate on
        Hann*C*T*exp(beta*g*D), exactly matching tracker inference.
        """
        active_mask = pred_dict['erc_active_mask'].reshape(-1).bool()
        design = getattr(self.cfg.MODEL.HEAD.ERC, 'DESIGN', 'legacy')
        is_v14 = design == 'response_intervention_gate_v14'
        is_v15 = design == 'response_continuous_worthiness_gate_v15'
        is_v16 = design == 'response_asymmetric_worthiness_gate_v16'
        is_v17 = (
            design == 'response_selective_abstention_worthiness_gate_v17'
        )
        is_v18 = design == 'response_preservation_veto_gate_v18'
        is_v17_family = is_v17 or is_v18
        is_intervention_gate = (
            is_v14 or is_v15 or is_v16 or is_v17_family
        )
        reference = pred_dict['erc_v13_beta']
        zero = reference.sum() * 0.0
        empty_status = {
            'Loss/erc_total': zero.detach(),
            'Loss/erc_v13_center': zero.detach(),
            'Loss/erc_v13_sigma': zero.detach(),
            'Loss/erc_v13_beta': zero.detach(),
            'Loss/erc_v13_sign': zero.detach(),
            'Loss/erc_v13_rank': zero.detach(),
            'Loss/erc_v13_preserve': zero.detach(),
            'Loss/erc_v13_noharm': zero.detach(),
            'Loss/erc_v13_reg': zero.detach(),
            'Loss/erc_v14_gate': zero.detach(),
            'Loss/erc_v14_gate_separation': zero.detach(),
            'ERC/v13_active_rate': active_mask.float().mean().detach(),
            'ERC/v13_correctable_rate': zero.detach(),
            'ERC/v13_selection_change_rate': zero.detach(),
            'ERC/v13_response_gain_mean': zero.detach(),
            'ERC/v14_gate_target_rate': zero.detach(),
            'ERC/v14_gate_mean': zero.detach(),
            'ERC/v14_gate_std': zero.detach(),
            'ERC/v14_gate_correctable_mean': zero.detach(),
            'ERC/v14_gate_keep_mean': zero.detach(),
            'ERC/v14_gate_mean_separation': zero.detach(),
            'ERC/v14_gate_balanced_accuracy': zero.detach(),
            'ERC/v14_gate_precision': zero.detach(),
            'ERC/v14_gate_recall': zero.detach(),
            'ERC/v14_effective_beta_abs_mean': zero.detach(),
            'ERC/candidate_iou_isolated': zero.detach().new_ones(()),
            'ERC/hard_masks_require_grad': zero.detach(),
        }
        if is_v15:
            empty_status.update({
                'Loss/erc_v15_gate_soft': zero.detach(),
                'Loss/erc_v15_gate_zero': zero.detach(),
                'Loss/erc_v15_gate_rank': zero.detach(),
                'ERC/v15_advantage_raw_mean': zero.detach(),
                'ERC/v15_advantage_raw_std': zero.detach(),
                'ERC/v15_advantage_effective_mean': zero.detach(),
                'ERC/v15_advantage_effective_std': zero.detach(),
                'ERC/v15_gate_target_mean': zero.detach(),
                'ERC/v15_gate_target_std': zero.detach(),
                'ERC/v15_gate_mean': zero.detach(),
                'ERC/v15_gate_std': zero.detach(),
                'ERC/v15_gate_advantage_pearson': zero.detach(),
                'ERC/v15_gate_advantage_spearman': zero.detach(),
                'ERC/v15_raw_nonpositive_rate': zero.detach(),
                'ERC/v15_preserve_region_rate': zero.detach(),
                'ERC/v15_rank_pair_rate': zero.detach(),
                'ERC/v15_effective_beta_abs_mean': zero.detach(),
            })
            for label in (
                'raw_le_005', 'raw_005_010', 'raw_010_020',
                'raw_020_040', 'raw_gt_040',
            ):
                empty_status.update({
                    'ERC/v15_bin_%s_rate' % label: zero.detach(),
                    'ERC/v15_bin_%s_gate_mean' % label: zero.detach(),
                    'ERC/v15_bin_%s_target_mean' % label: zero.detach(),
                })
        if is_v16:
            empty_status.update({
                'Loss/erc_v16_gate_calibration': zero.detach(),
                'Loss/erc_v16_gate_keep_logit': zero.detach(),
                'Loss/erc_v16_gate_rank': zero.detach(),
                'ERC/v16_advantage_raw_mean': zero.detach(),
                'ERC/v16_advantage_raw_std': zero.detach(),
                'ERC/v16_gate_target_mean': zero.detach(),
                'ERC/v16_gate_target_std': zero.detach(),
                'ERC/v16_gate_mean': zero.detach(),
                'ERC/v16_gate_std': zero.detach(),
                'ERC/v16_gate_target_mae': zero.detach(),
                'ERC/v16_gate_advantage_pearson': zero.detach(),
                'ERC/v16_gate_advantage_spearman': zero.detach(),
                'ERC/v16_preserve_region_rate': zero.detach(),
                'ERC/v16_preserve_gate_mean': zero.detach(),
                'ERC/v16_preserve_violation_rate': zero.detach(),
                'ERC/v16_strong_region_rate': zero.detach(),
                'ERC/v16_strong_gate_mean': zero.detach(),
                'ERC/v16_rank_pair_rate': zero.detach(),
                'ERC/v16_effective_beta_abs_mean': zero.detach(),
            })
        if is_v17_family:
            empty_status.update({
                'Loss/erc_v17_permission': zero.detach(),
                'Loss/erc_v17_worthiness': zero.detach(),
                'Loss/erc_v17_rank': zero.detach(),
                'ERC/v17_advantage_raw_mean': zero.detach(),
                'ERC/v17_actionable_rate': zero.detach(),
                'ERC/v17_permission_supervision_rate': zero.detach(),
                'ERC/v17_permission_ignore_rate': zero.detach(),
                'ERC/v17_permission_target_positive_rate': zero.detach(),
                'ERC/v17_permission_probability_mean': zero.detach(),
                'ERC/v17_permission_probability_std': zero.detach(),
                'ERC/v17_permission_positive_probability_mean': zero.detach(),
                'ERC/v17_permission_negative_probability_mean': zero.detach(),
                'ERC/v17_permission_probability_separation': zero.detach(),
                'ERC/v17_permission_activation_rate': zero.detach(),
                'ERC/v17_exact_abstention_rate': zero.detach(),
                'ERC/v17_permission_precision': zero.detach(),
                'ERC/v17_permission_recall': zero.detach(),
                'ERC/v17_permission_false_positive_rate': zero.detach(),
                'ERC/v17_oracle_positive_recall': zero.detach(),
                'ERC/v17_worthiness_target_mean': zero.detach(),
                'ERC/v17_worthiness_mean': zero.detach(),
                'ERC/v17_q_eff_mean': zero.detach(),
                'ERC/v17_effective_beta_abs_mean': zero.detach(),
                'ERC/v17_decision_change_given_activation': zero.detach(),
                'ERC/v17_decision_changes_per_1000_active_frames': zero.detach(),
                'ERC/v17_beneficial_given_activation': zero.detach(),
                'ERC/v17_harmful_given_activation': zero.detach(),
                'ERC/v17_intervention_precision': zero.detach(),
                'ERC/v17_rank_pair_rate': zero.detach(),
                'ERC/v17_all_abstain_batch': zero.detach(),
            })
        if is_v18:
            empty_status.update({
                'Loss/erc_v18_preservation': zero.detach(),
                'Loss/erc_v18_preservation_rank': zero.detach(),
                'ERC/v18_preservation_target_rate': zero.detach(),
                'ERC/v18_preservation_probability_mean': zero.detach(),
                'ERC/v18_preservation_probability_std': zero.detach(),
                'ERC/v18_preservation_positive_probability_mean': zero.detach(),
                'ERC/v18_preservation_negative_probability_mean': zero.detach(),
                'ERC/v18_preservation_probability_separation': zero.detach(),
                'ERC/v18_preservation_activation_rate': zero.detach(),
                'ERC/v18_preservation_precision': zero.detach(),
                'ERC/v18_preservation_recall': zero.detach(),
                'ERC/v18_preservation_false_positive_rate': zero.detach(),
                'ERC/v18_permission_block_rate': zero.detach(),
                'ERC/v18_oracle_permission_block_rate': zero.detach(),
                'ERC/v18_false_permission_block_rate': zero.detach(),
                'ERC/v18_final_allow_rate': zero.detach(),
                'ERC/v18_effective_beta_abs_mean': zero.detach(),
                'ERC/v18_rank_pair_rate': zero.detach(),
            })
        if not active_mask.any():
            return zero, empty_status

        eps = float(self.cfg.MODEL.HEAD.ERC.EPS)
        beta_max = float(self.cfg.MODEL.HEAD.ERC.MAX_ALPHA)
        d_max = float(self.cfg.MODEL.HEAD.ERC.V13_D_MAX)
        beta_steps = max(int(self.cfg.TRAIN.ERC_V13_BETA_STEPS), 3)
        oracle_margin = float(self.cfg.TRAIN.ERC_V13_ORACLE_MARGIN)
        rank_margin = float(self.cfg.TRAIN.ERC_V13_RANK_MARGIN)
        preserve_margin = float(self.cfg.TRAIN.ERC_V13_PRESERVE_MARGIN)
        keep_weight = float(self.cfg.TRAIN.ERC_V13_KEEP_WEIGHT)
        center_tau = max(float(self.cfg.TRAIN.ERC_V13_CENTER_TAU), eps)
        peaks_per_source = max(
            int(self.cfg.TRAIN.ERC_V13_PEAKS_PER_SOURCE), 1
        )
        max_anchors = max(int(self.cfg.TRAIN.ERC_V13_MAX_ANCHORS), 1)
        duplicate_radius = max(
            float(self.cfg.TRAIN.ERC_V13_DUPLICATE_RADIUS), 0.0
        )
        if is_v15:
            v15_keep_margin = float(self.cfg.TRAIN.ERC_V15_KEEP_MARGIN)
            v15_worthiness_tau = float(
                self.cfg.TRAIN.ERC_V15_WORTHINESS_TAU
            )
            v15_rank_target_gap = max(
                float(self.cfg.TRAIN.ERC_V15_GATE_RANK_TARGET_GAP), 0.0
            )
            v15_rank_margin = max(
                float(self.cfg.TRAIN.ERC_V15_GATE_RANK_MARGIN), 0.0
            )
        if is_v16:
            v16_keep_margin = float(self.cfg.TRAIN.ERC_V16_KEEP_MARGIN)
            v16_worthiness_tau = float(
                self.cfg.TRAIN.ERC_V16_WORTHINESS_TAU
            )
            v16_keep_ceiling = float(
                self.cfg.TRAIN.ERC_V16_KEEP_CEILING
            )
            if not 0.0 < v16_keep_ceiling < 1.0:
                raise ValueError('ERC-v16 keep ceiling must be in (0, 1)')
            v16_keep_logit_ceiling = float(np.log(
                v16_keep_ceiling / (1.0 - v16_keep_ceiling)
            ))
            v16_preserve_sample_weight = max(
                float(self.cfg.TRAIN.ERC_V16_PRESERVE_SAMPLE_WEIGHT), 0.0
            )
            v16_positive_sample_weight = max(
                float(self.cfg.TRAIN.ERC_V16_POSITIVE_SAMPLE_WEIGHT), 0.0
            )
            v16_rank_target_gap = max(
                float(self.cfg.TRAIN.ERC_V16_GATE_RANK_TARGET_GAP), 0.0
            )
            v16_rank_logit_margin = max(
                float(self.cfg.TRAIN.ERC_V16_GATE_RANK_LOGIT_MARGIN), 0.0
            )
        if is_v17_family:
            v17_keep_margin = float(self.cfg.TRAIN.ERC_V17_KEEP_MARGIN)
            v17_permission_margin = float(
                self.cfg.TRAIN.ERC_V17_PERMISSION_MARGIN
            )
            v17_permission_positive_weight = max(
                float(
                    self.cfg.TRAIN.ERC_V17_PERMISSION_POSITIVE_WEIGHT
                ),
                0.0,
            )
            v17_worthiness_tau = float(
                self.cfg.TRAIN.ERC_V17_WORTHINESS_TAU
            )
            v17_rank_target_gap = max(
                float(self.cfg.TRAIN.ERC_V17_RANK_TARGET_GAP), 0.0
            )
            v17_rank_logit_margin = max(
                float(self.cfg.TRAIN.ERC_V17_RANK_LOGIT_MARGIN), 0.0
            )
        if is_v18:
            v18_preservation_positive_weight = max(
                float(
                    self.cfg.TRAIN.ERC_V18_PRESERVATION_POSITIVE_WEIGHT
                ),
                0.0,
            )
            v18_preservation_rank_margin = max(
                float(self.cfg.TRAIN.ERC_V18_PRESERVATION_RANK_MARGIN),
                0.0,
            )

        pred_beta = pred_dict['erc_v13_beta'][active_mask].reshape(-1)
        pred_beta_logit = pred_dict[
            'erc_v13_beta_logit'
        ][active_mask].reshape(-1)
        center_logits = pred_dict[
            'erc_v13_center_logits'
        ][active_mask]
        sigma_logits = pred_dict[
            'erc_v13_sigma_logits'
        ][active_mask]
        correction_map = pred_dict[
            'erc_v13_correction_map'
        ][active_mask].flatten(1)
        d_map = pred_dict['erc_v13_d_map'][active_mask].detach().flatten(1)
        cls_response = pred_dict[
            'erc_cls_response'
        ][active_mask].detach().flatten(1)
        cont_response = pred_dict[
            'erc_cont_response'
        ][active_mask].detach().flatten(1)
        windowed_base = pred_dict[
            'erc_windowed_base_score'
        ][active_mask].detach()
        log_base = windowed_base.clamp_min(eps).log()
        gt_flat = gt_cls.reshape(gt_cls.shape[0], -1)[active_mask]
        gt_flat = gt_flat.detach().clamp_min(0.0)
        positive = gt_flat.gt(0.0)
        target_present = positive.any(dim=-1).detach()
        batch_size, candidate_count = log_base.shape
        feat_size = int(round(candidate_count ** 0.5))
        if feat_size * feat_size != candidate_count:
            raise ValueError('ERC-v13 requires a square response map')

        base_index = log_base.argmax(dim=-1).detach()
        baseline_correct = positive.gather(
            1, base_index[:, None]
        ).squeeze(1).detach()
        positive_base = log_base.masked_fill(~positive, float('-inf'))
        target_index = positive_base.argmax(dim=-1).detach()
        target_index = torch.where(target_present, target_index, base_index)
        baseline_positive_score = log_base.masked_fill(
            ~positive, float('-inf')
        ).max(dim=-1).values
        baseline_negative_score = log_base.masked_fill(
            positive, float('-inf')
        ).max(dim=-1).values
        valid_margin = target_present & (~positive.all(dim=-1))
        baseline_response_margin = torch.where(
            valid_margin,
            baseline_positive_score - baseline_negative_score,
            torch.zeros_like(baseline_positive_score),
        ).detach()

        sigma_values = torch.as_tensor(
            self.cfg.MODEL.HEAD.ERC.V13_SIGMAS,
            dtype=log_base.dtype,
            device=log_base.device,
        )
        coordinates = torch.arange(
            feat_size, dtype=log_base.dtype, device=log_base.device
        )
        grid_y, grid_x = torch.meshgrid(
            coordinates, coordinates, indexing='ij'
        )
        beta_values = torch.linspace(
            -beta_max, beta_max, beta_steps,
            dtype=log_base.dtype, device=log_base.device,
        )
        zero_beta_index = int(beta_values.abs().argmin())
        beta_values[zero_beta_index] = 0.0

        def local_peak_indices(values):
            spatial = values.reshape(1, 1, feat_size, feat_size)
            pooled = F.max_pool2d(
                spatial, kernel_size=3, stride=1, padding=1
            )
            is_peak = spatial.ge(pooled - 1e-12).reshape(-1)
            peak_values = values.masked_fill(~is_peak, float('-inf'))
            keep = min(peaks_per_source, int(is_peak.sum().item()))
            if keep <= 0:
                return [int(values.argmax())]
            return [int(index) for index in peak_values.topk(keep).indices]

        def merge_source_peaks(source_lists):
            selected = []
            maximum_rank = max(len(indices) for indices in source_lists)
            radius_sq = duplicate_radius ** 2
            for rank in range(maximum_rank):
                for indices in source_lists:
                    if rank >= len(indices):
                        continue
                    index = int(indices[rank])
                    row, col = divmod(index, feat_size)
                    duplicate = False
                    for previous in selected:
                        previous_row, previous_col = divmod(
                            previous, feat_size
                        )
                        if (
                            (row - previous_row) ** 2
                            + (col - previous_col) ** 2
                            <= radius_sq
                        ):
                            duplicate = True
                            break
                    if not duplicate:
                        selected.append(index)
                    if len(selected) >= max_anchors:
                        return selected
            if not selected:
                selected.append(int(source_lists[0][0]))
            return selected

        oracle_beta = log_base.new_zeros(batch_size)
        oracle_anchor_index = base_index.clone()
        oracle_selected_index = base_index.clone()
        best_sigma_index = torch.zeros(
            batch_size, dtype=torch.long, device=log_base.device
        )
        correctable = torch.zeros(
            batch_size, dtype=torch.bool, device=log_base.device
        )
        proposal_recall = torch.zeros_like(correctable)
        proposal_count = log_base.new_zeros(batch_size)
        oracle_response_margin = baseline_response_margin.clone()

        # All routing and Oracle selection are deliberately detached.  The
        # response maps propose anchors; GT only judges the resulting argmax.
        with torch.no_grad():
            for sample_index in range(batch_size):
                sources = (
                    cls_response[sample_index],
                    cont_response[sample_index],
                    windowed_base[sample_index],
                    d_map[sample_index].clamp_min(0.0),
                    (-d_map[sample_index]).clamp_min(0.0),
                )
                anchors = merge_source_peaks([
                    local_peak_indices(source) for source in sources
                ])
                anchor_tensor = torch.as_tensor(
                    anchors, dtype=torch.long, device=log_base.device
                )
                proposal_count[sample_index] = float(len(anchors))
                proposal_recall[sample_index] = positive[
                    sample_index, anchor_tensor
                ].any()
                if (
                    not bool(target_present[sample_index])
                    or bool(baseline_correct[sample_index])
                ):
                    continue

                anchor_y = torch.div(
                    anchor_tensor, feat_size, rounding_mode='floor'
                ).to(log_base.dtype)
                anchor_x = (anchor_tensor % feat_size).to(log_base.dtype)
                distance_sq = (
                    (grid_y[None] - anchor_y[:, None, None]).square()
                    + (grid_x[None] - anchor_x[:, None, None]).square()
                )
                gates = torch.exp(
                    -distance_sq[:, None]
                    / (
                        2.0
                        * sigma_values[None, :, None, None].square()
                    )
                ).flatten(2)
                gate_count = gates.shape[0] * gates.shape[1]
                flat_gates = gates.flatten(0, 1)
                gate_anchor = anchor_tensor[:, None].expand(
                    -1, sigma_values.numel()
                ).reshape(-1)
                gate_sigma = torch.arange(
                    sigma_values.numel(), device=log_base.device
                )[None, :].expand(len(anchors), -1).reshape(-1)

                oracle_scores = (
                    log_base[sample_index][None, None, :]
                    + beta_values[None, :, None]
                    * flat_gates[:, None, :]
                    * d_map[sample_index][None, None, :].clamp(
                        -d_max, d_max
                    )
                )
                flat_scores = oracle_scores.reshape(
                    gate_count * beta_steps, candidate_count
                )
                selected_indices = flat_scores.argmax(dim=-1)
                positive_score = flat_scores.masked_fill(
                    ~positive[sample_index][None, :], float('-inf')
                ).max(dim=-1).values
                negative_score = flat_scores.masked_fill(
                    positive[sample_index][None, :], float('-inf')
                ).max(dim=-1).values
                margins = positive_score - negative_score
                selected_positive = positive[
                    sample_index, selected_indices
                ]
                feasible = (
                    selected_positive & margins.ge(oracle_margin)
                ).reshape(gate_count, beta_steps)
                if not bool(feasible.any()):
                    continue

                # Exact lexicographic Oracle key used by the audit:
                # |beta|, then smaller sigma, then larger final margin.
                beta_abs = beta_values.abs()[None, :].expand(
                    gate_count, -1
                )
                minimum_beta = beta_abs.masked_fill(
                    ~feasible, float('inf')
                ).min()
                eligible = feasible & torch.isclose(
                    beta_abs, minimum_beta, atol=1e-8, rtol=0.0
                )
                sigma_cost = sigma_values[gate_sigma][:, None].expand(
                    -1, beta_steps
                )
                minimum_sigma = sigma_cost.masked_fill(
                    ~eligible, float('inf')
                ).min()
                eligible = eligible & torch.isclose(
                    sigma_cost, minimum_sigma, atol=1e-8, rtol=0.0
                )
                flat_margin = margins.reshape(gate_count, beta_steps)
                best_variant = flat_margin.masked_fill(
                    ~eligible, float('-inf')
                ).reshape(-1).argmax()
                gate_index = torch.div(
                    best_variant, beta_steps, rounding_mode='floor'
                )
                beta_index = best_variant % beta_steps
                correctable[sample_index] = True
                oracle_beta[sample_index] = beta_values[beta_index]
                oracle_anchor_index[sample_index] = gate_anchor[gate_index]
                best_sigma_index[sample_index] = gate_sigma[gate_index]
                oracle_selected_index[sample_index] = selected_indices[
                    best_variant
                ]
                oracle_response_margin[sample_index] = flat_margin[
                    gate_index, beta_index
                ]

        oracle_beta = oracle_beta.detach()
        oracle_anchor_index = oracle_anchor_index.detach()
        oracle_selected_index = oracle_selected_index.detach()
        best_sigma_index = best_sigma_index.detach()
        correctable = correctable.detach()
        baseline_response_margin = baseline_response_margin.detach()
        oracle_response_margin = oracle_response_margin.detach()
        unreachable = (
            target_present & (~baseline_correct) & (~correctable)
        ).detach()
        keep_mask = (~correctable).detach()

        epoch = int(epoch or 1)
        if is_intervention_gate:
            decision_values = list(
                self.cfg.TRAIN.ERC_V14_DECISION_SCALE_VALUES
            )
            teacher_values = list(
                self.cfg.TRAIN.ERC_V14_TEACHER_FORCING_VALUES
            )
            if not decision_values or not teacher_values:
                raise ValueError(
                    'ERC-v14/v15 curriculum lists must not be empty'
                )
            curriculum_index = max(epoch - 1, 0)
            decision_scale = float(
                decision_values[min(curriculum_index, len(decision_values) - 1)]
            )
            teacher_forcing_ratio = float(
                teacher_values[min(curriculum_index, len(teacher_values) - 1)]
            )
        elif epoch <= 1:
            decision_scale = float(
                self.cfg.TRAIN.ERC_V13_DECISION_SCALE_EPOCH1
            )
            teacher_forcing_ratio = float(
                self.cfg.TRAIN.ERC_V13_TEACHER_FORCING_EPOCH1
            )
        elif epoch == 2:
            decision_scale = float(
                self.cfg.TRAIN.ERC_V13_DECISION_SCALE_EPOCH2
            )
            teacher_forcing_ratio = float(
                self.cfg.TRAIN.ERC_V13_TEACHER_FORCING_EPOCH2
            )
        elif epoch == 3:
            decision_scale = float(
                self.cfg.TRAIN.ERC_V13_DECISION_SCALE_EPOCH3
            )
            teacher_forcing_ratio = float(
                self.cfg.TRAIN.ERC_V13_TEACHER_FORCING_EPOCH3
            )
        else:
            decision_scale = 1.0
            teacher_forcing_ratio = 0.0
        teacher_forcing_ratio = min(max(teacher_forcing_ratio, 0.0), 1.0)
        # v14 computes its all-anchor teacher predictions inside the model
        # forward so DDP owns every parameter use.  Evaluation/inference omits
        # those training-only tensors and therefore evaluates the predicted
        # center path without teacher forcing.
        anchor_prefix = (
            'erc_v17' if is_v17_family
            else (
                'erc_v16' if is_v16
                else ('erc_v15' if is_v15 else 'erc_v14')
            )
        )
        if (
            is_intervention_gate
            and anchor_prefix + '_anchor_beta_logit' not in pred_dict
        ):
            teacher_forcing_ratio = 0.0

        supervised_beta = pred_beta
        supervised_beta_logit = pred_beta_logit
        supervised_sigma_logits = sigma_logits
        if teacher_forcing_ratio > 0.0 and correctable.any():
            teacher_center_probability = F.one_hot(
                oracle_anchor_index, num_classes=candidate_count
            ).to(dtype=center_logits.dtype)
            if is_intervention_gate:
                active_anchor_beta_logit = pred_dict[
                    anchor_prefix + '_anchor_beta_logit'
                ][active_mask]
                active_anchor_sigma_logits = pred_dict[
                    anchor_prefix + '_anchor_sigma_logits'
                ][active_mask]
                active_batch_index = torch.arange(
                    batch_size, device=oracle_anchor_index.device
                )
                teacher_beta_logit = active_anchor_beta_logit[
                    active_batch_index, oracle_anchor_index
                ]
                teacher_sigma_logits = active_anchor_sigma_logits[
                    active_batch_index, oracle_anchor_index
                ]
            else:
                erc = self._model().box_head.erc
                teacher_decision = erc.decision_from_center_probability(
                    pred_dict['erc_v13_spatial_features'][active_mask],
                    pred_dict['erc_v13_encoder_input'][active_mask],
                    teacher_center_probability.detach(),
                )
                teacher_beta_logit = teacher_decision[
                    'beta_logit'
                ].reshape(-1)
                teacher_sigma_logits = teacher_decision['sigma_logits']
            teacher_beta = beta_max * torch.tanh(teacher_beta_logit)
            blended_beta = torch.lerp(
                pred_beta, teacher_beta, teacher_forcing_ratio
            )
            blended_beta_logit = torch.lerp(
                pred_beta_logit, teacher_beta_logit, teacher_forcing_ratio
            )
            blended_sigma_logits = torch.lerp(
                sigma_logits, teacher_sigma_logits,
                teacher_forcing_ratio,
            )
            supervised_beta = torch.where(
                correctable, blended_beta, pred_beta
            )
            supervised_beta_logit = torch.where(
                correctable, blended_beta_logit, pred_beta_logit
            )
            supervised_sigma_logits = torch.where(
                correctable[:, None], blended_sigma_logits, sigma_logits
            )

        if correctable.any():
            center_loss = F.cross_entropy(
                center_logits[correctable] / center_tau,
                oracle_anchor_index[correctable],
            )
            sigma_loss = F.cross_entropy(
                supervised_sigma_logits[correctable],
                best_sigma_index[correctable],
            )
        else:
            center_loss = zero
            sigma_loss = zero

        beta_element = F.smooth_l1_loss(
            supervised_beta, oracle_beta, reduction='none'
        )
        positive_beta_mask = (
            correctable & oracle_beta.gt(1e-8)
        ).detach()
        negative_beta_mask = (
            correctable & oracle_beta.lt(-1e-8)
        ).detach()
        directional_mask = (
            positive_beta_mask | negative_beta_mask
        ).detach()
        positive_beta_weight = max(
            float(self.cfg.TRAIN.ERC_V13_POSITIVE_BETA_WEIGHT), 0.0
        )
        direction_weights = torch.where(
            positive_beta_mask,
            beta_element.new_full(beta_element.shape, positive_beta_weight),
            torch.ones_like(beta_element),
        )
        if directional_mask.any():
            # Normalize by the number of directional samples, not by the
            # weight sum.  Dividing by the weight sum would cancel the class
            # weight inside each small batch and reintroduce the rare-positive
            # bias whenever a batch contains no positive-beta example.
            directional_count = directional_mask.float().sum().clamp_min(1.0)
            beta_loss = (
                beta_element[directional_mask]
                * direction_weights[directional_mask]
            ).sum() / directional_count
            sign_target = positive_beta_mask.to(
                dtype=supervised_beta_logit.dtype
            )
            sign_element = F.binary_cross_entropy_with_logits(
                supervised_beta_logit, sign_target, reduction='none'
            )
            sign_loss = (
                sign_element[directional_mask]
                * direction_weights[directional_mask]
            ).sum() / directional_count
        else:
            beta_loss = zero
            sign_loss = zero
        if keep_mask.any():
            beta_loss = beta_loss + keep_weight * beta_element[keep_mask].mean()

        final_log_score = log_base + correction_map
        final_positive = final_log_score.masked_fill(
            ~positive, float('-inf')
        ).max(dim=-1).values
        final_negative = final_log_score.masked_fill(
            positive, float('-inf')
        ).max(dim=-1).values
        rank_element = F.relu(
            rank_margin + final_negative - final_positive
        )
        rank_loss = (
            rank_element[correctable].mean()
            if correctable.any() else zero
        )

        final_base_score = final_log_score.gather(
            1, base_index[:, None]
        ).squeeze(1)
        other_mask = torch.ones_like(
            final_log_score, dtype=torch.bool
        )
        other_mask.scatter_(1, base_index[:, None], False)
        final_other_score = final_log_score.masked_fill(
            ~other_mask, float('-inf')
        ).max(dim=-1).values
        preserve_element = F.relu(
            preserve_margin + final_other_score - final_base_score
        )
        preserve_loss = (
            preserve_element[keep_mask].mean()
            if keep_mask.any() else zero
        )
        noharm_loss = (
            rank_element[baseline_correct].mean()
            if baseline_correct.any() else zero
        )
        reg_loss = pred_beta.abs().mean()

        gate_loss = zero
        gate_separation_loss = zero
        v15_gate_soft_loss = zero
        v15_gate_zero_loss = zero
        v15_gate_rank_loss = zero
        v15_advantage_raw = None
        v15_advantage_effective = None
        v15_gate_target = None
        v15_rank_pair_mask = None
        v16_gate_calibration_loss = zero
        v16_gate_keep_logit_loss = zero
        v16_gate_rank_loss = zero
        v16_advantage_raw = None
        v16_advantage_effective = None
        v16_gate_target = None
        v16_preserve_mask = None
        v16_strong_mask = None
        v16_rank_pair_mask = None
        v17_permission_loss = zero
        v17_worthiness_loss = zero
        v17_rank_loss = zero
        v17_advantage_raw = None
        v17_worthiness_target = None
        v17_permission_target = None
        v17_permission_supervision_mask = None
        v17_permission_positive = None
        v17_permission_negative = None
        v17_actionable = None
        v17_rank_pair_mask = None
        v17_permission_probability = None
        v17_permission_hard = None
        v17_worthiness = None
        v17_q_eff = None
        v18_preservation_loss = zero
        v18_preservation_rank_loss = zero
        v18_preservation_target = None
        v18_preservation_probability = None
        v18_preservation_hard = None
        v18_preservation_rank_pair_mask = None
        v18_allow_after_veto = None
        v18_effective_gate = None
        predicted_gate = None
        if is_v14:
            gate_logit = pred_dict[
                'erc_v14_gate_logit'
            ][active_mask].reshape(-1)
            predicted_gate = gate_logit.sigmoid()
            gate_target = correctable.to(dtype=gate_logit.dtype)
            gate_loss = F.binary_cross_entropy_with_logits(
                gate_logit, gate_target
            )
            if correctable.any() and keep_mask.any():
                gate_margin = float(
                    self.cfg.TRAIN.ERC_V14_GATE_SEPARATION_MARGIN
                )
                gate_separation_loss = F.relu(
                    gate_margin
                    - predicted_gate[correctable][:, None]
                    + predicted_gate[keep_mask][None, :]
                ).mean()
        elif is_v15:
            gate_logit = pred_dict[
                'erc_v15_gate_logit'
            ][active_mask].reshape(-1)
            predicted_gate = gate_logit.sigmoid()
            (
                v15_advantage_raw,
                v15_advantage_effective,
                v15_gate_target,
            ) = continuous_worthiness_target(
                oracle_response_margin - baseline_response_margin,
                keep_margin=v15_keep_margin,
                worthiness_tau=v15_worthiness_tau,
            )
            v15_gate_soft_loss = F.binary_cross_entropy_with_logits(
                gate_logit, v15_gate_target
            )
            v15_preserve_mask = v15_advantage_raw.le(
                v15_keep_margin
            ).detach()
            if v15_preserve_mask.any():
                v15_gate_zero_loss = predicted_gate[
                    v15_preserve_mask
                ].square().mean()

            target_difference = (
                v15_gate_target[:, None] - v15_gate_target[None, :]
            )
            v15_rank_pair_mask = (
                target_difference.gt(0.0)
                & target_difference.ge(v15_rank_target_gap)
            ).detach()
            if v15_rank_pair_mask.any():
                predicted_difference = (
                    predicted_gate[:, None] - predicted_gate[None, :]
                )
                v15_gate_rank_loss = F.relu(
                    v15_rank_margin - predicted_difference
                )[v15_rank_pair_mask].mean()
        elif is_v16:
            gate_logit = pred_dict[
                'erc_v16_gate_logit'
            ][active_mask].reshape(-1)
            predicted_gate = gate_logit.sigmoid()
            (
                v16_advantage_raw,
                v16_advantage_effective,
                v16_gate_target,
            ) = continuous_worthiness_target(
                oracle_response_margin - baseline_response_margin,
                keep_margin=v16_keep_margin,
                worthiness_tau=v16_worthiness_tau,
            )
            v16_preserve_mask = v16_advantage_raw.le(
                v16_keep_margin
            ).detach()
            v16_strong_mask = v16_gate_target.ge(0.5).detach()

            # False intervention is more costly in an online tracker because
            # it changes future crops.  Cost-sensitive calibration therefore
            # weights preserve samples and high-worthiness samples explicitly
            # while normalizing the batch scale.
            calibration_element = F.binary_cross_entropy_with_logits(
                gate_logit, v16_gate_target, reduction='none'
            )
            sample_weight = torch.ones_like(calibration_element)
            sample_weight = sample_weight + (
                v16_preserve_sample_weight
                * v16_preserve_mask.to(dtype=sample_weight.dtype)
            )
            sample_weight = sample_weight + (
                v16_positive_sample_weight * v16_gate_target
            )
            v16_gate_calibration_loss = (
                calibration_element * sample_weight
            ).sum() / sample_weight.sum().clamp_min(1.0)

            # A logit-space ceiling keeps gradients strong even when q is
            # already small.  It is continuous and does not introduce a hard
            # inference threshold.
            if v16_preserve_mask.any():
                v16_gate_keep_logit_loss = F.relu(
                    gate_logit[v16_preserve_mask]
                    - v16_keep_logit_ceiling
                ).square().mean()

            target_difference = (
                v16_gate_target[:, None] - v16_gate_target[None, :]
            )
            v16_rank_pair_mask = (
                target_difference.gt(0.0)
                & target_difference.ge(v16_rank_target_gap)
            ).detach()
            if v16_rank_pair_mask.any():
                logit_difference = (
                    gate_logit[:, None] - gate_logit[None, :]
                )
                v16_gate_rank_loss = F.relu(
                    v16_rank_logit_margin - logit_difference
                )[v16_rank_pair_mask].mean()
        elif is_v17_family:
            permission_logit = pred_dict[
                'erc_v17_permission_logit'
            ][active_mask].reshape(-1)
            worthiness_logit = pred_dict[
                'erc_v17_worthiness_logit'
            ][active_mask].reshape(-1)
            v17_permission_probability = permission_logit.sigmoid()
            v17_permission_hard = pred_dict[
                'erc_v17_permission_hard'
            ][active_mask].reshape(-1).detach()
            v17_worthiness = worthiness_logit.sigmoid()
            v17_q_eff = pred_dict[
                'erc_v17_effective_gate'
            ][active_mask].reshape(-1)

            v17_advantage_raw, _, v17_worthiness_target = (
                continuous_worthiness_target(
                    oracle_response_margin - baseline_response_margin,
                    keep_margin=v17_keep_margin,
                    worthiness_tau=v17_worthiness_tau,
                )
            )
            v17_actionable = oracle_selected_index.ne(base_index).detach()
            (
                v17_permission_target,
                v17_permission_supervision_mask,
                v17_permission_positive,
                v17_permission_negative,
            ) = selective_permission_target(
                v17_advantage_raw,
                v17_actionable,
                keep_margin=v17_keep_margin,
                permission_margin=v17_permission_margin,
            )
            if v17_permission_supervision_mask.any():
                v17_permission_loss = F.binary_cross_entropy_with_logits(
                    permission_logit[v17_permission_supervision_mask],
                    v17_permission_target[v17_permission_supervision_mask],
                    pos_weight=permission_logit.new_tensor(
                        v17_permission_positive_weight
                    ),
                )
            v17_worthiness_loss = F.binary_cross_entropy_with_logits(
                worthiness_logit, v17_worthiness_target
            )

            target_difference = (
                v17_worthiness_target[:, None]
                - v17_worthiness_target[None, :]
            )
            v17_rank_pair_mask = (
                target_difference.gt(0.0)
                & target_difference.ge(v17_rank_target_gap)
            ).detach()
            if v17_rank_pair_mask.any():
                logit_difference = (
                    worthiness_logit[:, None] - worthiness_logit[None, :]
                )
                v17_rank_loss = F.relu(
                    v17_rank_logit_margin - logit_difference
                )[v17_rank_pair_mask].mean()

            if is_v18:
                preservation_logit = pred_dict[
                    'erc_v18_preservation_logit'
                ][active_mask].reshape(-1)
                v18_preservation_probability = (
                    preservation_logit.sigmoid()
                )
                v18_preservation_hard = pred_dict[
                    'erc_v18_preservation_hard'
                ][active_mask].reshape(-1).detach()
                v18_allow_after_veto = pred_dict[
                    'erc_v18_allow_after_veto'
                ][active_mask].reshape(-1)
                v18_effective_gate = pred_dict[
                    'erc_v18_effective_gate'
                ][active_mask].reshape(-1)

                # The preservation target is intentionally response-only:
                # it asks whether the official Hann*C*T argmax already lies
                # inside the detached official target-response support.  No
                # candidate box, IoU, geometry, or future trajectory is used.
                v18_preservation_target = baseline_correct.to(
                    dtype=preservation_logit.dtype
                ).detach()
                v18_preservation_loss = (
                    F.binary_cross_entropy_with_logits(
                        preservation_logit,
                        v18_preservation_target,
                        pos_weight=preservation_logit.new_tensor(
                            v18_preservation_positive_weight
                        ),
                    )
                )

                preservation_target_difference = (
                    v18_preservation_target[:, None]
                    - v18_preservation_target[None, :]
                )
                v18_preservation_rank_pair_mask = (
                    preservation_target_difference.gt(0.0)
                ).detach()
                if v18_preservation_rank_pair_mask.any():
                    preservation_logit_difference = (
                        preservation_logit[:, None]
                        - preservation_logit[None, :]
                    )
                    v18_preservation_rank_loss = F.relu(
                        v18_preservation_rank_margin
                        - preservation_logit_difference
                    )[v18_preservation_rank_pair_mask].mean()

        total = (
            self.loss_weight['erc_v13_center'] * center_loss
            + self.loss_weight['erc_v13_sigma'] * sigma_loss
            + self.loss_weight['erc_v13_beta'] * beta_loss
            + self.loss_weight['erc_v13_sign'] * sign_loss
            + decision_scale * (
                self.loss_weight['erc_v13_rank'] * rank_loss
                + self.loss_weight['erc_v13_preserve'] * preserve_loss
                + self.loss_weight['erc_v13_noharm'] * noharm_loss
            )
            + self.loss_weight['erc_v13_reg'] * reg_loss
            + self.loss_weight['erc_v14_gate'] * gate_loss
            + self.loss_weight['erc_v14_gate_separation'] * gate_separation_loss
            + self.loss_weight['erc_v15_gate_soft'] * v15_gate_soft_loss
            + self.loss_weight['erc_v15_gate_zero'] * v15_gate_zero_loss
            + self.loss_weight['erc_v15_gate_rank'] * v15_gate_rank_loss
            + self.loss_weight['erc_v16_gate_calibration']
            * v16_gate_calibration_loss
            + self.loss_weight['erc_v16_gate_keep_logit']
            * v16_gate_keep_logit_loss
            + self.loss_weight['erc_v16_gate_rank'] * v16_gate_rank_loss
            + self.loss_weight['erc_v17_permission'] * v17_permission_loss
            + self.loss_weight['erc_v17_worthiness'] * v17_worthiness_loss
            + self.loss_weight['erc_v17_rank'] * v17_rank_loss
            + self.loss_weight['erc_v18_preservation']
            * v18_preservation_loss
            + self.loss_weight['erc_v18_preservation_rank']
            * v18_preservation_rank_loss
        )

        v15_status = {}
        v16_status = {}
        v17_status = {}
        v18_status = {}
        with torch.no_grad():
            final_index = final_log_score.argmax(dim=-1)
            selection_change = final_index.ne(base_index)
            base_quality = gt_flat.gather(
                1, base_index[:, None]
            ).squeeze(1)
            final_quality = gt_flat.gather(
                1, final_index[:, None]
            ).squeeze(1)
            response_gain = final_quality - base_quality
            changed_count = selection_change.float().sum().clamp_min(1.0)
            changed_gain = (
                response_gain * selection_change.float()
            ).sum() / changed_count
            beneficial = selection_change & response_gain.gt(0.0)
            harmful = selection_change & response_gain.lt(0.0)
            predicted_center = center_logits.argmax(dim=-1)
            center_hit = positive.gather(
                1, predicted_center[:, None]
            ).squeeze(1)
            center_target_match = predicted_center.eq(oracle_anchor_index)
            predicted_sigma_index = sigma_logits.argmax(dim=-1)
            beta_saturation = pred_beta.abs().ge(0.95 * beta_max)
            oracle_beta_positive = oracle_beta.gt(1e-8)
            oracle_beta_negative = oracle_beta.lt(-1e-8)
            predicted_beta_positive = pred_beta.gt(0.0)
            supervised_beta_positive = supervised_beta.gt(0.0)
            predicted_positive_accuracy = (
                predicted_beta_positive[positive_beta_mask].float().mean()
                if positive_beta_mask.any() else zero.detach()
            )
            predicted_negative_accuracy = (
                (~predicted_beta_positive[negative_beta_mask]).float().mean()
                if negative_beta_mask.any() else zero.detach()
            )
            supervised_positive_accuracy = (
                supervised_beta_positive[positive_beta_mask].float().mean()
                if positive_beta_mask.any() else zero.detach()
            )
            supervised_negative_accuracy = (
                (~supervised_beta_positive[negative_beta_mask]).float().mean()
                if negative_beta_mask.any() else zero.detach()
            )
            oracle_anchor_positive = positive.gather(
                1, oracle_anchor_index[:, None]
            ).squeeze(1)
            if is_v14:
                predicted_gate_positive = predicted_gate.ge(0.5)
                gate_true_positive = predicted_gate_positive & correctable
                gate_false_positive = predicted_gate_positive & keep_mask
                gate_true_negative = (~predicted_gate_positive) & keep_mask
                gate_false_negative = (~predicted_gate_positive) & correctable
                gate_positive_accuracy = (
                    gate_true_positive.float().sum()
                    / correctable.float().sum().clamp_min(1.0)
                )
                gate_negative_accuracy = (
                    gate_true_negative.float().sum()
                    / keep_mask.float().sum().clamp_min(1.0)
                )
                gate_precision = (
                    gate_true_positive.float().sum()
                    / (gate_true_positive | gate_false_positive)
                    .float().sum().clamp_min(1.0)
                )
                gate_recall = (
                    gate_true_positive.float().sum()
                    / (gate_true_positive | gate_false_negative)
                    .float().sum().clamp_min(1.0)
                )
                gate_correctable_mean = (
                    predicted_gate[correctable].mean()
                    if correctable.any() else predicted_gate.mean().detach()
                )
                gate_keep_mean = (
                    predicted_gate[keep_mask].mean()
                    if keep_mask.any() else predicted_gate.mean().detach()
                )
            elif is_v15 or is_v16 or is_v17_family:
                # Binary v14 metrics have no meaning for continuous
                # worthiness targets.  Keep the prediction for versioned
                # statistics but report no legacy binary numbers.
                gate_positive_accuracy = zero.detach()
                gate_negative_accuracy = zero.detach()
                gate_precision = zero.detach()
                gate_recall = zero.detach()
                gate_correctable_mean = zero.detach()
                gate_keep_mean = zero.detach()
            else:
                predicted_gate = pred_beta.new_zeros(pred_beta.shape)
                gate_positive_accuracy = zero.detach()
                gate_negative_accuracy = zero.detach()
                gate_precision = zero.detach()
                gate_recall = zero.detach()
                gate_correctable_mean = zero.detach()
                gate_keep_mean = zero.detach()

            if is_v15:
                def detached_pearson(first, second):
                    first = first.detach().float().reshape(-1)
                    second = second.detach().float().reshape(-1)
                    if first.numel() < 2:
                        return zero.detach()
                    first_centered = first - first.mean()
                    second_centered = second - second.mean()
                    denominator = (
                        first_centered.square().sum()
                        * second_centered.square().sum()
                    ).sqrt()
                    if float(denominator) <= 1e-12:
                        return zero.detach()
                    return (
                        first_centered * second_centered
                    ).sum() / denominator

                def average_rank(values):
                    values = values.detach().float().reshape(-1)
                    less_count = (
                        values[:, None] > values[None, :]
                    ).float().sum(dim=1)
                    equal_count = torch.isclose(
                        values[:, None], values[None, :],
                        atol=1e-12, rtol=0.0,
                    ).float().sum(dim=1)
                    return less_count + 0.5 * (equal_count - 1.0)

                raw = v15_advantage_raw.detach()
                effective = v15_advantage_effective.detach()
                target = v15_gate_target.detach()
                gate = predicted_gate.detach()
                rank_pair_rate = v15_rank_pair_mask.float().mean()
                v15_status = {
                    'Loss/erc_v15_gate_soft': v15_gate_soft_loss.detach(),
                    'Loss/erc_v15_gate_zero': v15_gate_zero_loss.detach(),
                    'Loss/erc_v15_gate_rank': v15_gate_rank_loss.detach(),
                    'ERC/v15_advantage_raw_mean': raw.mean(),
                    'ERC/v15_advantage_raw_std': raw.std(unbiased=False),
                    'ERC/v15_advantage_effective_mean': effective.mean(),
                    'ERC/v15_advantage_effective_std': effective.std(
                        unbiased=False
                    ),
                    'ERC/v15_gate_target_mean': target.mean(),
                    'ERC/v15_gate_target_std': target.std(unbiased=False),
                    'ERC/v15_gate_mean': gate.mean(),
                    'ERC/v15_gate_std': gate.std(unbiased=False),
                    'ERC/v15_gate_advantage_pearson': detached_pearson(
                        gate, raw
                    ),
                    'ERC/v15_gate_advantage_spearman': detached_pearson(
                        average_rank(gate), average_rank(raw)
                    ),
                    'ERC/v15_raw_nonpositive_rate': raw.le(0.0).float().mean(),
                    'ERC/v15_preserve_region_rate': raw.le(
                        v15_keep_margin
                    ).float().mean(),
                    'ERC/v15_rank_pair_rate': rank_pair_rate,
                    'ERC/v15_effective_beta_abs_mean': (
                        gate * pred_beta.detach()
                    ).abs().mean(),
                }
                v15_bins = (
                    ('raw_le_005', raw.le(0.05)),
                    ('raw_005_010', raw.gt(0.05) & raw.le(0.10)),
                    ('raw_010_020', raw.gt(0.10) & raw.le(0.20)),
                    ('raw_020_040', raw.gt(0.20) & raw.le(0.40)),
                    ('raw_gt_040', raw.gt(0.40)),
                )
                for label, bin_mask in v15_bins:
                    rate = bin_mask.float().mean()
                    gate_mean = (
                        gate[bin_mask].mean()
                        if bin_mask.any() else zero.detach()
                    )
                    target_mean = (
                        target[bin_mask].mean()
                        if bin_mask.any() else zero.detach()
                    )
                    v15_status.update({
                        'ERC/v15_bin_%s_rate' % label: rate,
                        'ERC/v15_bin_%s_gate_mean' % label: gate_mean,
                        'ERC/v15_bin_%s_target_mean' % label: target_mean,
                    })
            elif is_v16:
                def detached_pearson(first, second):
                    first = first.detach().float().reshape(-1)
                    second = second.detach().float().reshape(-1)
                    if first.numel() < 2:
                        return zero.detach()
                    first_centered = first - first.mean()
                    second_centered = second - second.mean()
                    denominator = (
                        first_centered.square().sum()
                        * second_centered.square().sum()
                    ).sqrt()
                    if float(denominator) <= 1e-12:
                        return zero.detach()
                    return (
                        first_centered * second_centered
                    ).sum() / denominator

                def average_rank(values):
                    values = values.detach().float().reshape(-1)
                    less_count = (
                        values[:, None] > values[None, :]
                    ).float().sum(dim=1)
                    equal_count = torch.isclose(
                        values[:, None], values[None, :],
                        atol=1e-12, rtol=0.0,
                    ).float().sum(dim=1)
                    return less_count + 0.5 * (equal_count - 1.0)

                raw = v16_advantage_raw.detach()
                target = v16_gate_target.detach()
                gate = predicted_gate.detach()
                preserve_mask = v16_preserve_mask.detach()
                strong_mask = v16_strong_mask.detach()
                rank_pair_rate = v16_rank_pair_mask.float().mean()
                v16_status = {
                    'Loss/erc_v16_gate_calibration': (
                        v16_gate_calibration_loss.detach()
                    ),
                    'Loss/erc_v16_gate_keep_logit': (
                        v16_gate_keep_logit_loss.detach()
                    ),
                    'Loss/erc_v16_gate_rank': v16_gate_rank_loss.detach(),
                    'ERC/v16_advantage_raw_mean': raw.mean(),
                    'ERC/v16_advantage_raw_std': raw.std(unbiased=False),
                    'ERC/v16_gate_target_mean': target.mean(),
                    'ERC/v16_gate_target_std': target.std(unbiased=False),
                    'ERC/v16_gate_mean': gate.mean(),
                    'ERC/v16_gate_std': gate.std(unbiased=False),
                    'ERC/v16_gate_target_mae': (gate - target).abs().mean(),
                    'ERC/v16_gate_advantage_pearson': detached_pearson(
                        gate, raw
                    ),
                    'ERC/v16_gate_advantage_spearman': detached_pearson(
                        average_rank(gate), average_rank(raw)
                    ),
                    'ERC/v16_preserve_region_rate': (
                        preserve_mask.float().mean()
                    ),
                    'ERC/v16_preserve_gate_mean': (
                        gate[preserve_mask].mean()
                        if preserve_mask.any() else zero.detach()
                    ),
                    'ERC/v16_preserve_violation_rate': (
                        gate[preserve_mask].gt(v16_keep_ceiling).float().mean()
                        if preserve_mask.any() else zero.detach()
                    ),
                    'ERC/v16_strong_region_rate': strong_mask.float().mean(),
                    'ERC/v16_strong_gate_mean': (
                        gate[strong_mask].mean()
                        if strong_mask.any() else zero.detach()
                    ),
                    'ERC/v16_rank_pair_rate': rank_pair_rate,
                    'ERC/v16_effective_beta_abs_mean': (
                        gate * pred_beta.detach()
                    ).abs().mean(),
                }
            elif is_v17_family:
                activation = v17_permission_hard.bool()
                positive_mask = v17_permission_positive.detach()
                negative_mask = v17_permission_negative.detach()
                supervised_mask = v17_permission_supervision_mask.detach()
                true_positive = activation & positive_mask
                false_positive = activation & negative_mask
                false_negative = (~activation) & positive_mask
                activated_change = activation & selection_change
                activated_beneficial = activation & beneficial
                activated_harmful = activation & harmful
                activation_count = activation.float().sum().clamp_min(1.0)
                changed_intervention_count = (
                    activated_beneficial | activated_harmful
                ).float().sum().clamp_min(1.0)
                positive_probability_mean = (
                    v17_permission_probability.detach()[positive_mask].mean()
                    if positive_mask.any() else zero.detach()
                )
                negative_probability_mean = (
                    v17_permission_probability.detach()[negative_mask].mean()
                    if negative_mask.any() else zero.detach()
                )
                v17_status = {
                    'Loss/erc_v17_permission': v17_permission_loss.detach(),
                    'Loss/erc_v17_worthiness': v17_worthiness_loss.detach(),
                    'Loss/erc_v17_rank': v17_rank_loss.detach(),
                    'ERC/v17_advantage_raw_mean': (
                        v17_advantage_raw.detach().mean()
                    ),
                    'ERC/v17_actionable_rate': (
                        v17_actionable.float().mean()
                    ),
                    'ERC/v17_permission_supervision_rate': (
                        supervised_mask.float().mean()
                    ),
                    'ERC/v17_permission_ignore_rate': (
                        (~supervised_mask).float().mean()
                    ),
                    'ERC/v17_permission_target_positive_rate': (
                        positive_mask.float().sum()
                        / supervised_mask.float().sum().clamp_min(1.0)
                    ),
                    'ERC/v17_permission_probability_mean': (
                        v17_permission_probability.detach().mean()
                    ),
                    'ERC/v17_permission_probability_std': (
                        v17_permission_probability.detach().std(
                            unbiased=False
                        )
                    ),
                    'ERC/v17_permission_positive_probability_mean': (
                        positive_probability_mean
                    ),
                    'ERC/v17_permission_negative_probability_mean': (
                        negative_probability_mean
                    ),
                    'ERC/v17_permission_probability_separation': (
                        positive_probability_mean - negative_probability_mean
                    ),
                    'ERC/v17_permission_activation_rate': (
                        activation.float().mean()
                    ),
                    'ERC/v17_exact_abstention_rate': (
                        (~activation).float().mean()
                    ),
                    'ERC/v17_permission_precision': (
                        true_positive.float().sum()
                        / (true_positive | false_positive)
                        .float().sum().clamp_min(1.0)
                    ),
                    'ERC/v17_permission_recall': (
                        true_positive.float().sum()
                        / (true_positive | false_negative)
                        .float().sum().clamp_min(1.0)
                    ),
                    'ERC/v17_permission_false_positive_rate': (
                        false_positive.float().sum()
                        / negative_mask.float().sum().clamp_min(1.0)
                    ),
                    'ERC/v17_oracle_positive_recall': (
                        true_positive.float().sum()
                        / positive_mask.float().sum().clamp_min(1.0)
                    ),
                    'ERC/v17_worthiness_target_mean': (
                        v17_worthiness_target.detach().mean()
                    ),
                    'ERC/v17_worthiness_mean': (
                        v17_worthiness.detach().mean()
                    ),
                    'ERC/v17_q_eff_mean': v17_q_eff.detach().mean(),
                    'ERC/v17_effective_beta_abs_mean': (
                        v17_q_eff.detach() * pred_beta.detach()
                    ).abs().mean(),
                    'ERC/v17_decision_change_given_activation': (
                        activated_change.float().sum() / activation_count
                    ),
                    'ERC/v17_decision_changes_per_1000_active_frames': (
                        selection_change.float().mean() * 1000.0
                    ),
                    'ERC/v17_beneficial_given_activation': (
                        activated_beneficial.float().sum() / activation_count
                    ),
                    'ERC/v17_harmful_given_activation': (
                        activated_harmful.float().sum() / activation_count
                    ),
                    'ERC/v17_intervention_precision': (
                        activated_beneficial.float().sum()
                        / changed_intervention_count
                    ),
                    'ERC/v17_rank_pair_rate': (
                        v17_rank_pair_mask.float().mean()
                    ),
                    'ERC/v17_all_abstain_batch': (
                        activation.float().sum().eq(0).float()
                    ),
                }
                if is_v18:
                    preservation_activation = (
                        v18_preservation_hard.bool()
                    )
                    preservation_positive = (
                        v18_preservation_target.bool()
                    )
                    preservation_negative = ~preservation_positive
                    preservation_true_positive = (
                        preservation_activation & preservation_positive
                    )
                    preservation_false_positive = (
                        preservation_activation & preservation_negative
                    )
                    preservation_false_negative = (
                        (~preservation_activation) & preservation_positive
                    )
                    blocked_permission = (
                        activation & preservation_activation
                    )
                    final_allow = activation & (~preservation_activation)
                    preservation_positive_probability_mean = (
                        v18_preservation_probability.detach()[
                            preservation_positive
                        ].mean()
                        if preservation_positive.any() else zero.detach()
                    )
                    preservation_negative_probability_mean = (
                        v18_preservation_probability.detach()[
                            preservation_negative
                        ].mean()
                        if preservation_negative.any() else zero.detach()
                    )
                    v18_status = {
                        'Loss/erc_v18_preservation': (
                            v18_preservation_loss.detach()
                        ),
                        'Loss/erc_v18_preservation_rank': (
                            v18_preservation_rank_loss.detach()
                        ),
                        'ERC/v18_preservation_target_rate': (
                            preservation_positive.float().mean()
                        ),
                        'ERC/v18_preservation_probability_mean': (
                            v18_preservation_probability.detach().mean()
                        ),
                        'ERC/v18_preservation_probability_std': (
                            v18_preservation_probability.detach().std(
                                unbiased=False
                            )
                        ),
                        'ERC/v18_preservation_positive_probability_mean': (
                            preservation_positive_probability_mean
                        ),
                        'ERC/v18_preservation_negative_probability_mean': (
                            preservation_negative_probability_mean
                        ),
                        'ERC/v18_preservation_probability_separation': (
                            preservation_positive_probability_mean
                            - preservation_negative_probability_mean
                        ),
                        'ERC/v18_preservation_activation_rate': (
                            preservation_activation.float().mean()
                        ),
                        'ERC/v18_preservation_precision': (
                            preservation_true_positive.float().sum()
                            / (
                                preservation_true_positive
                                | preservation_false_positive
                            ).float().sum().clamp_min(1.0)
                        ),
                        'ERC/v18_preservation_recall': (
                            preservation_true_positive.float().sum()
                            / (
                                preservation_true_positive
                                | preservation_false_negative
                            ).float().sum().clamp_min(1.0)
                        ),
                        'ERC/v18_preservation_false_positive_rate': (
                            preservation_false_positive.float().sum()
                            / preservation_negative.float().sum().clamp_min(1.0)
                        ),
                        'ERC/v18_permission_block_rate': (
                            blocked_permission.float().sum()
                            / activation.float().sum().clamp_min(1.0)
                        ),
                        'ERC/v18_oracle_permission_block_rate': (
                            (blocked_permission & positive_mask).float().sum()
                            / positive_mask.float().sum().clamp_min(1.0)
                        ),
                        'ERC/v18_false_permission_block_rate': (
                            (blocked_permission & negative_mask).float().sum()
                            / negative_mask.float().sum().clamp_min(1.0)
                        ),
                        'ERC/v18_final_allow_rate': final_allow.float().mean(),
                        'ERC/v18_effective_beta_abs_mean': (
                            v18_effective_gate.detach()
                            * pred_beta.detach()
                        ).abs().mean(),
                        'ERC/v18_rank_pair_rate': (
                            v18_preservation_rank_pair_mask.float().mean()
                        ),
                    }

        hard_tensors = (
            positive, target_present, base_index, baseline_correct,
            target_index, correctable, unreachable, oracle_beta,
            oracle_anchor_index, best_sigma_index, proposal_recall,
            oracle_selected_index, baseline_response_margin,
            oracle_response_margin,
        )
        if is_v15:
            hard_tensors = hard_tensors + (
                v15_advantage_raw, v15_advantage_effective,
                v15_gate_target, v15_rank_pair_mask,
            )
        if is_v16:
            hard_tensors = hard_tensors + (
                v16_advantage_raw, v16_advantage_effective,
                v16_gate_target, v16_preserve_mask, v16_strong_mask,
                v16_rank_pair_mask,
            )
        if is_v17_family:
            hard_tensors = hard_tensors + (
                v17_advantage_raw, v17_worthiness_target,
                v17_permission_target, v17_permission_supervision_mask,
                v17_permission_positive, v17_permission_negative,
                v17_actionable, v17_rank_pair_mask,
            )
        if is_v18:
            hard_tensors = hard_tensors + (
                v18_preservation_target,
                v18_preservation_hard,
                v18_preservation_rank_pair_mask,
            )
        hard_masks_require_grad = any(
            tensor.requires_grad for tensor in hard_tensors
        )
        status = {
            'Loss/erc_total': total.detach(),
            'Loss/erc_v13_center': center_loss.detach(),
            'Loss/erc_v13_sigma': sigma_loss.detach(),
            'Loss/erc_v13_beta': beta_loss.detach(),
            'Loss/erc_v13_sign': sign_loss.detach(),
            'Loss/erc_v13_rank': rank_loss.detach(),
            'Loss/erc_v13_preserve': preserve_loss.detach(),
            'Loss/erc_v13_noharm': noharm_loss.detach(),
            'Loss/erc_v13_reg': reg_loss.detach(),
            'Loss/erc_v14_gate': gate_loss.detach(),
            'Loss/erc_v14_gate_separation': gate_separation_loss.detach(),
            'ERC/v13_active_rate': active_mask.float().mean().detach(),
            'ERC/v13_decision_loss_scale': zero.detach().new_tensor(
                decision_scale
            ),
            'ERC/v13_teacher_forcing_ratio': zero.detach().new_tensor(
                teacher_forcing_ratio
            ),
            'ERC/v13_target_present_rate': target_present.float().mean(),
            'ERC/v13_baseline_correct_rate': baseline_correct.float().mean(),
            'ERC/v13_correctable_rate': correctable.float().mean(),
            'ERC/v13_unreachable_rate': unreachable.float().mean(),
            'ERC/v13_beta_mean': pred_beta.mean().detach(),
            'ERC/v13_beta_std': pred_beta.std(unbiased=False).detach(),
            'ERC/v13_beta_abs_mean': pred_beta.abs().mean().detach(),
            'ERC/v13_beta_target_mean': oracle_beta.mean(),
            'ERC/v13_beta_target_std': oracle_beta.std(unbiased=False),
            'ERC/v13_beta_target_abs_mean': oracle_beta.abs().mean(),
            'ERC/v13_beta_target_positive_rate': (
                oracle_beta_positive.float().mean()
            ),
            'ERC/v13_beta_target_negative_rate': (
                oracle_beta_negative.float().mean()
            ),
            'ERC/v13_beta_mae': (pred_beta.detach() - oracle_beta).abs().mean(),
            'ERC/v13_supervised_beta_std': supervised_beta.detach().std(
                unbiased=False
            ),
            'ERC/v13_supervised_beta_mae': (
                supervised_beta.detach() - oracle_beta
            ).abs().mean(),
            'ERC/v13_beta_positive_accuracy': predicted_positive_accuracy,
            'ERC/v13_beta_negative_accuracy': predicted_negative_accuracy,
            'ERC/v13_beta_balanced_accuracy': 0.5 * (
                predicted_positive_accuracy + predicted_negative_accuracy
            ),
            'ERC/v13_supervised_beta_balanced_accuracy': 0.5 * (
                supervised_positive_accuracy + supervised_negative_accuracy
            ),
            'ERC/v13_beta_saturation_rate': beta_saturation.float().mean(),
            'ERC/v13_center_positive_rate': center_hit.float().mean(),
            'ERC/v13_center_target_accuracy': (
                center_target_match[correctable].float().mean()
                if correctable.any() else zero.detach()
            ),
            'ERC/v13_oracle_anchor_positive_rate': (
                oracle_anchor_positive[correctable].float().mean()
                if correctable.any() else zero.detach()
            ),
            'ERC/v13_proposal_recall_rate': proposal_recall.float().mean(),
            'ERC/v13_proposal_count_mean': proposal_count.mean(),
            'ERC/v13_sigma_mean': pred_dict[
                'erc_v13_sigma'
            ][active_mask].mean().detach(),
            'ERC/v13_sigma_target_mean': sigma_values[
                best_sigma_index[correctable]
            ].mean() if correctable.any() else zero.detach(),
            'ERC/v13_sigma_accuracy': (
                predicted_sigma_index[correctable]
                .eq(best_sigma_index[correctable]).float().mean()
                if correctable.any() else zero.detach()
            ),
            'ERC/v13_selection_change_rate': selection_change.float().mean(),
            'ERC/v13_beneficial_change_rate': beneficial.float().mean(),
            'ERC/v13_harmful_change_rate': harmful.float().mean(),
            'ERC/v13_response_gain_mean': response_gain.mean(),
            'ERC/v13_changed_response_gain_mean': changed_gain,
            'ERC/v13_final_margin_mean': (
                final_positive - final_negative
            ).mean().detach(),
            'ERC/v13_correction_abs_mean': correction_map.abs().mean().detach(),
            'ERC/v14_gate_target_rate': (
                correctable.float().mean() if is_v14 else zero.detach()
            ),
            'ERC/v14_gate_mean': (
                predicted_gate.mean().detach() if is_v14 else zero.detach()
            ),
            'ERC/v14_gate_std': (
                predicted_gate.std(unbiased=False).detach()
                if is_v14 else zero.detach()
            ),
            'ERC/v14_gate_correctable_mean': gate_correctable_mean.detach(),
            'ERC/v14_gate_keep_mean': gate_keep_mean.detach(),
            'ERC/v14_gate_mean_separation': (
                gate_correctable_mean - gate_keep_mean
            ).detach(),
            'ERC/v14_gate_balanced_accuracy': (
                0.5 * (gate_positive_accuracy + gate_negative_accuracy)
            ).detach(),
            'ERC/v14_gate_precision': gate_precision.detach(),
            'ERC/v14_gate_recall': gate_recall.detach(),
            'ERC/v14_effective_beta_abs_mean': (
                (predicted_gate * pred_beta).abs().mean().detach()
                if is_v14 else zero.detach()
            ),
            'ERC/candidate_iou_isolated': zero.detach().new_ones(()),
            'ERC/hard_masks_require_grad': zero.detach().new_tensor(
                float(hard_masks_require_grad)
            ),
        }
        if is_v15:
            status.update(v15_status)
        if is_v16:
            status.update(v16_status)
        if is_v17_family:
            status.update(v17_status)
        if is_v18:
            status.update(v18_status)
        return total, status

    def compute_erc_local_reliability_v12_losses(self, pred_dict, gt_cls):
        """Train ERC-v12 with response advantage and ranking resistance.

        The continuous target strength is high only when the official center
        heatmap prefers another local response and the Hann-aware baseline gap
        is realistically correctable.  Its complement supervises preservation
        of the baseline ranking.  Every routing/target decision is detached;
        bbox geometry and IoU are intentionally absent from this method.
        """
        active_mask = pred_dict['erc_active_mask'].reshape(-1).bool()
        reference = pred_dict['erc_local_delta_patch']
        zero = reference.sum() * 0.0
        empty_status = {
            'Loss/erc_total': zero.detach(),
            'Loss/erc_v12_local': zero.detach(),
            'Loss/erc_v12_hard': zero.detach(),
            'Loss/erc_v12_gap': zero.detach(),
            'Loss/erc_v12_outside': zero.detach(),
            'Loss/erc_v12_reliability': zero.detach(),
            'Loss/erc_v12_preserve': zero.detach(),
            'Loss/erc_v12_identity': zero.detach(),
            'Loss/erc_v12_reg': zero.detach(),
            'ERC/v12_active_rate': active_mask.float().mean().detach(),
            'ERC/v12_target_present_rate': zero.detach(),
            'ERC/v12_clear_correction_rate': zero.detach(),
            'ERC/v12_reliability_mean': zero.detach(),
            'ERC/v12_reliability_target_mean': zero.detach(),
            'ERC/v12_selection_change_rate': zero.detach(),
            'ERC/v12_response_gain_mean': zero.detach(),
            'ERC/candidate_iou_isolated': zero.detach().new_ones(()),
            'ERC/hard_masks_require_grad': zero.detach(),
        }
        if not active_mask.any():
            return zero, empty_status

        eps = float(self.cfg.MODEL.HEAD.ERC.EPS)
        delta_max = float(self.cfg.MODEL.HEAD.ERC.LOCAL_DELTA_MAX)
        soft_gamma = float(self.cfg.TRAIN.ERC_V12_SOFT_LABEL_GAMMA)
        advantage_gamma = float(self.cfg.TRAIN.ERC_V12_ADVANTAGE_GAMMA)
        local_tau = max(float(self.cfg.TRAIN.ERC_V12_LOCAL_TAU), eps)
        gap_tau = max(float(self.cfg.TRAIN.ERC_V12_GAP_TAU), eps)
        reachability_tau = max(
            float(self.cfg.TRAIN.ERC_V12_REACHABILITY_TAU), eps
        )
        hard_margin = float(self.cfg.TRAIN.ERC_V12_HARD_MARGIN)
        gap_margin = float(self.cfg.TRAIN.ERC_V12_GAP_MARGIN)
        outside_margin = float(self.cfg.TRAIN.ERC_V12_OUTSIDE_MARGIN)
        preserve_margin = float(self.cfg.TRAIN.ERC_V12_PRESERVE_MARGIN)

        local_indices = pred_dict['erc_local_indices'][active_mask].detach()
        local_valid = (
            pred_dict['erc_local_valid_mask'][active_mask].bool().detach()
        )
        local_delta = pred_dict['erc_local_delta_patch'][active_mask]
        full_delta = pred_dict['erc_local_delta_map'][active_mask].flatten(1)
        reliability = pred_dict['erc_local_reliability'][active_mask].reshape(-1)
        reliability_logit = pred_dict[
            'erc_local_reliability_logit'
        ][active_mask].reshape(-1)
        windowed_base = pred_dict[
            'erc_windowed_base_score'
        ][active_mask].detach()
        log_base = windowed_base.clamp_min(eps).log()
        gt_flat = gt_cls.reshape(gt_cls.shape[0], -1)[active_mask].detach()
        gt_flat = gt_flat.clamp_min(0.0)

        base_patch = log_base.gather(1, local_indices)
        valid_float = local_valid.to(gt_flat.dtype)
        target_patch = gt_flat.gather(1, local_indices) * valid_float
        target_mass = target_patch.pow(soft_gamma)
        target_sum = target_mass.sum(dim=-1, keepdim=True)
        target_present = target_sum.squeeze(1).gt(eps).detach()
        target_probability = target_mass / target_sum.clamp_min(eps)
        target_index = target_patch.argmax(dim=-1).detach()
        patch_size = local_indices.shape[1]
        center_index = patch_size // 2
        center_index_tensor = torch.full_like(target_index, center_index)
        target_is_correction = target_index.ne(center_index_tensor).detach()

        target_quality = target_patch.gather(
            1, target_index[:, None]
        ).squeeze(1)
        center_quality = target_patch[:, center_index]
        response_advantage = (
            (target_quality - center_quality).clamp_min(0.0)
            / target_quality.clamp_min(eps)
        ).detach()
        base_center = base_patch[:, center_index]
        base_target = base_patch.gather(
            1, target_index[:, None]
        ).squeeze(1)
        required_gap = (
            base_center - base_target + gap_margin
        ).clamp_min(0.0).detach()

        # Both factors are continuous.  The sigmoid reflects whether a
        # centered residual bounded by +/- delta_max can span the ranking gap;
        # it is supervision only and never becomes an inference threshold.
        gap_resistance = torch.exp(-required_gap / gap_tau)
        soft_reachability = torch.sigmoid(
            (2.0 * delta_max - required_gap) / reachability_tau
        )
        reliability_target = (
            response_advantage.pow(advantage_gamma)
            * gap_resistance
            * soft_reachability
            * target_present.to(response_advantage.dtype)
            * target_is_correction.to(response_advantage.dtype)
        ).clamp(0.0, 1.0).detach()
        ambiguous_weight = (1.0 - reliability_target).detach()
        sample_count = reliability_target.new_tensor(
            float(reliability_target.numel())
        ).clamp_min(1.0)

        calibrated_patch = base_patch + local_delta
        local_element = -(
            target_probability
            * F.log_softmax(calibrated_patch / local_tau, dim=-1)
        ).sum(dim=-1)
        local_loss = (local_element * reliability_target).sum() / sample_count

        target_score = calibrated_patch.gather(
            1, target_index[:, None]
        ).squeeze(1)
        negative_mask = local_valid.clone()
        negative_mask.scatter_(1, target_index[:, None], False)
        strongest_negative = calibrated_patch.masked_fill(
            ~negative_mask, float('-inf')
        ).max(dim=-1).values
        hard_element = F.relu(
            hard_margin + strongest_negative - target_score
        )
        hard_loss = (hard_element * reliability_target).sum() / sample_count

        delta_center = local_delta[:, center_index]
        delta_target = local_delta.gather(
            1, target_index[:, None]
        ).squeeze(1)
        achieved_gap = delta_target - delta_center
        gap_element = F.relu(required_gap - achieved_gap)
        gap_loss = (gap_element * reliability_target).sum() / sample_count

        local_full_count = torch.zeros_like(log_base, dtype=torch.long)
        local_full_count.scatter_add_(
            1, local_indices, local_valid.to(dtype=torch.long)
        )
        local_full_mask = local_full_count.gt(0)
        outside_score = log_base.masked_fill(
            local_full_mask, float('-inf')
        ).max(dim=-1).values
        outside_element = F.relu(
            outside_score + outside_margin - (base_target + delta_target)
        )
        outside_loss = (
            outside_element * reliability_target
        ).sum() / sample_count

        other_mask = local_valid.clone()
        other_mask[:, center_index] = False
        calibrated_center = calibrated_patch[:, center_index]
        calibrated_other = calibrated_patch.masked_fill(
            ~other_mask, float('-inf')
        ).max(dim=-1).values
        preserve_element = F.relu(
            calibrated_other + preserve_margin - calibrated_center
        )
        preserve_loss = (
            preserve_element * ambiguous_weight
        ).sum() / sample_count

        valid_count = valid_float.sum(dim=-1).clamp_min(1.0)
        identity_element = (
            local_delta.abs() * valid_float
        ).sum(dim=-1) / valid_count
        identity_loss = (
            identity_element * ambiguous_weight
        ).sum() / sample_count
        reg_loss = (
            local_delta.abs() * valid_float
        ).sum() / valid_float.sum().clamp_min(1.0)
        reliability_loss = F.binary_cross_entropy_with_logits(
            reliability_logit, reliability_target
        )

        total = (
            self.loss_weight['erc_v12_local'] * local_loss
            + self.loss_weight['erc_v12_hard'] * hard_loss
            + self.loss_weight['erc_v12_gap'] * gap_loss
            + self.loss_weight['erc_v12_outside'] * outside_loss
            + self.loss_weight['erc_v12_reliability'] * reliability_loss
            + self.loss_weight['erc_v12_preserve'] * preserve_loss
            + self.loss_weight['erc_v12_identity'] * identity_loss
            + self.loss_weight['erc_v12_reg'] * reg_loss
        )

        with torch.no_grad():
            final_log_score = log_base + full_delta
            base_index = log_base.argmax(dim=-1)
            final_index = final_log_score.argmax(dim=-1)
            selection_change = final_index.ne(base_index)
            base_quality = gt_flat.gather(1, base_index[:, None]).squeeze(1)
            final_quality = gt_flat.gather(1, final_index[:, None]).squeeze(1)
            response_gain = final_quality - base_quality
            changed_count = selection_change.float().sum().clamp_min(1.0)
            changed_gain = (
                response_gain * selection_change.float()
            ).sum() / changed_count
            beneficial_change = selection_change & response_gain.gt(0.0)
            harmful_change = selection_change & response_gain.lt(0.0)
            saturation = local_delta.abs().ge(0.95 * delta_max) & local_valid
            clear_correction = target_present & target_is_correction

        hard_tensors = (
            local_indices, local_valid, target_present, target_index,
            target_is_correction, reliability_target,
        )
        hard_masks_require_grad = any(
            tensor.requires_grad for tensor in hard_tensors
        )
        status = {
            'Loss/erc_total': total.detach(),
            'Loss/erc_v12_local': local_loss.detach(),
            'Loss/erc_v12_hard': hard_loss.detach(),
            'Loss/erc_v12_gap': gap_loss.detach(),
            'Loss/erc_v12_outside': outside_loss.detach(),
            'Loss/erc_v12_reliability': reliability_loss.detach(),
            'Loss/erc_v12_preserve': preserve_loss.detach(),
            'Loss/erc_v12_identity': identity_loss.detach(),
            'Loss/erc_v12_reg': reg_loss.detach(),
            'ERC/v12_active_rate': active_mask.float().mean().detach(),
            'ERC/v12_target_present_rate': target_present.float().mean(),
            'ERC/v12_clear_correction_rate': clear_correction.float().mean(),
            'ERC/v12_ambiguous_rate': (1.0 - clear_correction.float()).mean(),
            'ERC/v12_response_advantage_mean': response_advantage.mean(),
            'ERC/v12_required_gap_mean': required_gap.mean(),
            'ERC/v12_gap_resistance_mean': gap_resistance.mean(),
            'ERC/v12_soft_reachability_mean': soft_reachability.mean(),
            'ERC/v12_reliability_mean': reliability.mean().detach(),
            'ERC/v12_reliability_std': reliability.std(unbiased=False).detach(),
            'ERC/v12_reliability_target_mean': reliability_target.mean(),
            'ERC/v12_reliability_target_std': reliability_target.std(
                unbiased=False
            ),
            'ERC/v12_reliability_mae': (
                reliability.detach() - reliability_target
            ).abs().mean(),
            'ERC/v12_local_delta_abs': (
                local_delta.abs() * valid_float
            ).sum().detach() / valid_float.sum().clamp_min(1.0),
            'ERC/v12_local_delta_saturation_rate': saturation.float().sum()
            / local_valid.float().sum().clamp_min(1.0),
            'ERC/v12_preserve_violation_rate': preserve_element.gt(0).float().mean(),
            'ERC/v12_selection_change_rate': selection_change.float().mean(),
            'ERC/v12_beneficial_change_rate': beneficial_change.float().mean(),
            'ERC/v12_harmful_change_rate': harmful_change.float().mean(),
            'ERC/v12_response_gain_mean': response_gain.mean(),
            'ERC/v12_changed_response_gain_mean': changed_gain,
            'ERC/candidate_iou_isolated': zero.detach().new_ones(()),
            'ERC/hard_masks_require_grad': zero.detach().new_tensor(
                float(hard_masks_require_grad)
            ),
        }
        return total, status

    def compute_erc_local_spatial_v11_losses(self, pred_dict, gt_cls):
        """Response-only local listwise and reachable-margin supervision.

        All targets come from the official center heatmap.  Hann*C*T and the
        discrete routing indices are detached, and bbox/IoU never enter this
        loss.  Therefore ERC-specific gradients can update only ERC-v11 when
        DETACH_FEATURES is enabled; the original Head continues to learn from
        the unchanged UVLTrack task losses in ``compute_losses``.
        """
        active_mask = pred_dict['erc_active_mask'].reshape(-1).bool()
        reference = pred_dict['erc_local_delta_patch']
        zero = reference.sum() * 0.0
        if not active_mask.any():
            detached_zero = zero.detach()
            return zero, {
                'Loss/erc_total': detached_zero,
                'Loss/erc_v11_local': detached_zero,
                'Loss/erc_v11_hard': detached_zero,
                'Loss/erc_v11_gap': detached_zero,
                'Loss/erc_v11_outside': detached_zero,
                'Loss/erc_v11_reg': detached_zero,
                'ERC/local_active_rate': active_mask.float().mean().detach(),
                'ERC/local_target_present_rate': detached_zero,
                'ERC/local_selection_change_rate': detached_zero,
                'ERC/local_response_gain_mean': detached_zero,
                'ERC/candidate_iou_isolated': detached_zero.new_ones(()),
                'ERC/hard_masks_require_grad': detached_zero,
            }

        eps = float(self.cfg.MODEL.HEAD.ERC.EPS)
        delta_max = float(self.cfg.MODEL.HEAD.ERC.LOCAL_DELTA_MAX)
        gamma = float(self.cfg.TRAIN.ERC_V11_SOFT_LABEL_GAMMA)
        tau = max(float(self.cfg.TRAIN.ERC_V11_LOCAL_TAU), 1e-6)
        hard_margin = float(self.cfg.TRAIN.ERC_V11_HARD_MARGIN)
        gap_margin = float(self.cfg.TRAIN.ERC_V11_GAP_MARGIN)
        outside_margin = float(self.cfg.TRAIN.ERC_V11_OUTSIDE_MARGIN)

        local_indices = pred_dict['erc_local_indices'][active_mask].detach()
        local_valid = pred_dict['erc_local_valid_mask'][active_mask].bool().detach()
        local_delta = pred_dict['erc_local_delta_patch'][active_mask]
        full_delta = pred_dict['erc_local_delta_map'][active_mask].flatten(1)
        windowed_base = pred_dict['erc_windowed_base_score'][active_mask].detach()
        log_base = windowed_base.clamp_min(eps).log()
        gt_flat = gt_cls.reshape(gt_cls.shape[0], -1)[active_mask].detach().clamp_min(0.0)

        base_patch = log_base.gather(1, local_indices)
        target_patch = gt_flat.gather(1, local_indices) * local_valid.to(gt_flat.dtype)
        target_mass = target_patch.pow(gamma)
        target_sum = target_mass.sum(dim=-1, keepdim=True)
        target_present = target_sum.squeeze(1).gt(eps).detach()
        target_probability = target_mass / target_sum.clamp_min(eps)
        calibrated_patch = base_patch + local_delta

        local_element = -(
            target_probability
            * F.log_softmax(calibrated_patch / tau, dim=-1)
        ).sum(dim=-1)
        if target_present.any():
            local_loss = local_element[target_present].mean()
        else:
            local_loss = zero

        target_index = target_patch.argmax(dim=-1).detach()
        target_score = calibrated_patch.gather(
            1, target_index[:, None]
        ).squeeze(1)
        negative_mask = local_valid.clone()
        negative_mask.scatter_(1, target_index[:, None], False)
        strongest_negative = calibrated_patch.masked_fill(
            ~negative_mask, float('-inf')
        ).max(dim=-1).values
        hard_element = F.relu(
            hard_margin + strongest_negative - target_score
        )
        if target_present.any():
            hard_loss = hard_element[target_present].mean()
        else:
            hard_loss = zero

        patch_size = local_indices.shape[1]
        center_index = patch_size // 2
        center_index_tensor = torch.full_like(target_index, center_index)
        base_center = base_patch[:, center_index]
        base_target = base_patch.gather(
            1, target_index[:, None]
        ).squeeze(1)
        delta_center = local_delta[:, center_index]
        delta_target = local_delta.gather(
            1, target_index[:, None]
        ).squeeze(1)
        required_gap = (base_center - base_target + gap_margin).detach()
        gap_reachable = (
            target_present
            & target_index.ne(center_index_tensor)
            & required_gap.le(2.0 * delta_max + 1e-6)
        ).detach()
        achieved_gap = delta_target - delta_center
        gap_element = F.relu(required_gap - achieved_gap)
        if gap_reachable.any():
            gap_loss = gap_element[gap_reachable].mean()
        else:
            gap_loss = zero

        local_full_count = torch.zeros_like(log_base, dtype=torch.long)
        local_full_count.scatter_add_(
            1, local_indices, local_valid.to(dtype=torch.long)
        )
        local_full_mask = local_full_count.gt(0)
        outside_score = log_base.masked_fill(
            local_full_mask, float('-inf')
        ).max(dim=-1).values
        outside_required = (
            outside_score + outside_margin - base_target
        ).detach()
        outside_reachable = (
            target_present & outside_required.le(delta_max + 1e-6)
        ).detach()
        outside_element = F.relu(
            outside_score + outside_margin - (base_target + delta_target)
        )
        if outside_reachable.any():
            outside_loss = outside_element[outside_reachable].mean()
        else:
            outside_loss = zero

        valid_float = local_valid.to(local_delta.dtype)
        reg_loss = (
            local_delta.abs() * valid_float
        ).sum() / valid_float.sum().clamp_min(1.0)

        total = (
            self.loss_weight['erc_v11_local'] * local_loss
            + self.loss_weight['erc_v11_hard'] * hard_loss
            + self.loss_weight['erc_v11_gap'] * gap_loss
            + self.loss_weight['erc_v11_outside'] * outside_loss
            + self.loss_weight['erc_v11_reg'] * reg_loss
        )

        with torch.no_grad():
            final_log_score = log_base + full_delta
            base_index = log_base.argmax(dim=-1)
            final_index = final_log_score.argmax(dim=-1)
            selection_change = final_index.ne(base_index)
            base_quality = gt_flat.gather(1, base_index[:, None]).squeeze(1)
            final_quality = gt_flat.gather(1, final_index[:, None]).squeeze(1)
            response_gain = final_quality - base_quality
            changed_count = selection_change.float().sum().clamp_min(1.0)
            changed_gain = (
                response_gain * selection_change.float()
            ).sum() / changed_count
            beneficial_change = selection_change & response_gain.gt(0.0)
            harmful_change = selection_change & response_gain.lt(0.0)
            final_local_slot = final_index[:, None].eq(local_indices)
            final_inside_local = (
                final_local_slot & local_valid
            ).any(dim=-1)
            target_offset_row = torch.div(
                target_index, int(self.cfg.MODEL.HEAD.ERC.LOCAL_WINDOW),
                rounding_mode='floor',
            ) - int(self.cfg.MODEL.HEAD.ERC.LOCAL_WINDOW) // 2
            target_offset_col = (
                target_index % int(self.cfg.MODEL.HEAD.ERC.LOCAL_WINDOW)
            ) - int(self.cfg.MODEL.HEAD.ERC.LOCAL_WINDOW) // 2
            target_offset = (
                target_offset_row.abs() + target_offset_col.abs()
            ).float()
            saturation = local_delta.abs().ge(0.95 * delta_max) & local_valid

        hard_tensors = (
            local_indices, local_valid, target_present, target_index,
            gap_reachable, outside_reachable,
        )
        hard_masks_require_grad = any(
            tensor.requires_grad for tensor in hard_tensors
        )
        status = {
            'Loss/erc_total': total.detach(),
            'Loss/erc_v11_local': local_loss.detach(),
            'Loss/erc_v11_hard': hard_loss.detach(),
            'Loss/erc_v11_gap': gap_loss.detach(),
            'Loss/erc_v11_outside': outside_loss.detach(),
            'Loss/erc_v11_reg': reg_loss.detach(),
            'ERC/local_active_rate': active_mask.float().mean().detach(),
            'ERC/local_target_present_rate': target_present.float().mean(),
            'ERC/local_gap_reachable_rate': gap_reachable.float().mean(),
            'ERC/local_outside_reachable_rate': outside_reachable.float().mean(),
            'ERC/local_target_offset_l1_mean': target_offset[target_present].mean()
            if target_present.any() else zero.detach(),
            'ERC/local_delta_abs': (
                local_delta.abs() * valid_float
            ).sum().detach() / valid_float.sum().clamp_min(1.0),
            'ERC/local_delta_std': local_delta[local_valid].std(unbiased=False).detach(),
            'ERC/local_delta_saturation_rate': saturation.float().sum()
            / local_valid.float().sum().clamp_min(1.0),
            'ERC/local_selection_change_rate': selection_change.float().mean(),
            'ERC/local_beneficial_change_rate': beneficial_change.float().mean(),
            'ERC/local_harmful_change_rate': harmful_change.float().mean(),
            'ERC/local_response_gain_mean': response_gain.mean(),
            'ERC/local_changed_response_gain_mean': changed_gain,
            'ERC/local_outside_takeover_rate': (~final_inside_local).float().mean(),
            'ERC/local_gap_residual_mean': (
                required_gap - achieved_gap.detach()
            )[gap_reachable].mean() if gap_reachable.any() else zero.detach(),
            'ERC/candidate_iou_isolated': zero.detach().new_ones(()),
            'ERC/hard_masks_require_grad': zero.detach().new_tensor(
                float(hard_masks_require_grad)
            ),
        }
        return total, status

    @staticmethod
    def response_alpha_oracle(
        cls_response, cont_response, gt_probability, tau, max_alpha, steps
    ):
        """Pure-response scalar-alpha headroom; never uses boxes or IoU."""
        with torch.no_grad():
            steps = max(int(steps), 3)
            if steps % 2 == 0:
                steps += 1
            alpha_grid = torch.linspace(
                -float(max_alpha),
                float(max_alpha),
                steps,
                device=cls_response.device,
                dtype=cls_response.dtype,
            )
            log_cls = cls_response.clamp_min(1e-12).log()
            log_cont = cont_response.clamp_min(1e-12).log()
            base_log = (cls_response * cont_response).clamp_min(1e-12).log()
            log_ratio = log_cls - log_cont
            logits = (
                base_log[:, None, :]
                + alpha_grid[None, :, None] * log_ratio[:, None, :]
            ) / max(float(tau), 1e-6)
            log_probability = F.log_softmax(logits, dim=-1)
            map_losses = -(
                gt_probability[:, None, :] * log_probability
            ).sum(dim=-1)
            zero_index = alpha_grid.abs().argmin()
            baseline_loss = map_losses[:, zero_index]
            oracle_loss, best_index = map_losses.min(dim=1)
            oracle_gain = (baseline_loss - oracle_loss).clamp_min(0.0)
            best_alpha = alpha_grid[best_index]
        return oracle_gain, best_alpha, baseline_loss, oracle_loss

    def compute_erc_response_v7_losses(
        self, pred_dict, gt_cls, gt_bbox, epoch=1
    ):
        """Response-only supervision for global-regional ERC-v7.

        Candidate boxes and IoU are intentionally absent from every target,
        mask, and differentiable loss below.  They are consulted only in the
        final ``torch.no_grad`` diagnostics block.
        """
        active_mask = pred_dict['erc_active_mask'].reshape(-1).bool()
        active_fraction = active_mask.float().mean()
        zero = pred_dict['bbox_map'].sum() * 0.0
        status_names = (
            'Loss/erc_total', 'Loss/erc_response_map',
            'Loss/erc_response_rank', 'Loss/erc_response_reliability',
            'Loss/erc_response_gate', 'Loss/erc_response_preference',
            'Loss/erc_response_noharm', 'Loss/erc_response_identity',
            'Loss/erc_response_smooth', 'ERC/response_map_loss_base',
            'ERC/response_map_loss_erc',
            'ERC/response_noharm_violation_rate',
            'ERC/response_correctable_rate',
            'ERC/positive_batch_rate', 'ERC/target_mass_cls',
            'ERC/target_mass_cont', 'ERC/target_mass_base',
            'ERC/target_mass_global', 'ERC/target_mass_erc',
            'ERC/peak_distance_cls', 'ERC/peak_distance_cont',
            'ERC/peak_distance_base', 'ERC/peak_distance_erc',
            'ERC/base_margin', 'ERC/erc_margin',
            'ERC/gate_pos_mean', 'ERC/gate_neg_mean',
            'ERC/gate_separation', 'ERC/reliability_cls',
            'ERC/reliability_cont', 'ERC/reliability_base',
            'ERC/reliability_cls_target',
            'ERC/reliability_cont_target',
            'ERC/reliability_base_target', 'ERC/intervention_gate',
            'ERC/intervention_gate_target', 'ERC/alpha_global_abs',
            'ERC/alpha_region_abs', 'ERC/alpha_region_std',
            'ERC/alpha_final_abs', 'ERC/alpha_near_zero_rate',
             'ERC/alpha_saturation_rate',
             'ERC/global_only_response_gain', 'ERC/full_response_gain',
             'ERC/response_map_loss_train',
             'ERC/response_map_loss_final',
             'ERC/target_mass_train', 'ERC/target_mass_final',
             'ERC/peak_distance_train', 'ERC/peak_distance_final',
             'ERC/erc_train_margin', 'ERC/erc_final_margin',
             'ERC/branch_correctable_rate',
             'ERC/response_alpha_correctable_rate',
             'ERC/response_alpha_oracle_gain',
             'ERC/best_response_alpha',
             'ERC/alpha_raw_abs', 'ERC/alpha_train_abs',
             'ERC/alpha_raw_positive_rate',
             'ERC/alpha_raw_negative_rate',
             'ERC/alpha_near_zero_rate_raw',
             'ERC/alpha_near_zero_rate_train',
             'ERC/alpha_near_zero_rate_final',
             'ERC/alpha_saturation_rate_raw',
             'ERC/alpha_saturation_rate_final',
             'ERC/gate_mean', 'ERC/lambda_gate',
             'ERC/noharm_weight_effective',
             'ERC/identity_weight_effective',
             'ERC/selection_change_rate', 'ERC/candidate_iou_gain',
            'ERC/changed_candidate_iou_gain',
            'ERC/beneficial_change_rate', 'ERC/harmful_change_rate',
            'ERC/active_fraction',
        )
        if not active_mask.any():
            status = {name: zero.detach() for name in status_names}
            status['ERC/active_fraction'] = active_fraction.detach()
            return zero, status

        height, width = gt_cls.shape[-2:]
        response_shape = (-1, height * width)
        cls_response = pred_dict['erc_cls_response'].reshape(
            response_shape
        )[active_mask]
        cont_response = pred_dict['erc_cont_response'].reshape(
            response_shape
        )[active_mask]
        base_response = pred_dict['base_fusion_score'].reshape(
            response_shape
        )[active_mask]
        erc_response = pred_dict['erc_score'].reshape(
            response_shape
        )[active_mask]
        global_response = pred_dict['erc_global_score'].reshape(
            response_shape
        )[active_mask]
        gt_heatmap = gt_cls.reshape(response_shape)[active_mask].to(
            dtype=cls_response.dtype
        ).clamp_min(0.0)
        gt_probability = gt_heatmap / gt_heatmap.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        positive_mask = gt_heatmap > 0
        # The official Gaussian label always has a positive center.  Keep a
        # defensive one-hot fallback for malformed samples.
        missing_positive = ~positive_mask.any(dim=-1)
        if missing_positive.any():
            fallback_index = gt_heatmap.argmax(dim=-1)
            positive_mask[missing_positive] = False
            positive_mask[
                missing_positive, fallback_index[missing_positive]
            ] = True

        tau = max(float(self.cfg.TRAIN.ERC_RESPONSE_TAU), 1e-6)

        def response_probability(response):
            return F.softmax(
                response.clamp_min(1e-12).log() / tau, dim=-1
            )

        probability_cls = response_probability(cls_response)
        probability_cont = response_probability(cont_response)
        probability_base = response_probability(base_response)
        probability_global = response_probability(global_response)
        probability_erc = response_probability(erc_response)

        with torch.no_grad():
            positive_float = positive_mask.to(probability_cls.dtype)
            target_cls = (probability_cls * positive_float).sum(
                dim=-1, keepdim=True
            ).detach()
            target_cont = (probability_cont * positive_float).sum(
                dim=-1, keepdim=True
            ).detach()
            target_base = (probability_base * positive_float).sum(
                dim=-1, keepdim=True
            ).detach()
            branch_response_gain = (
                torch.maximum(target_cls, target_cont) - target_base
            ).clamp(0.0, 1.0)
            response_alpha_gain, best_response_alpha, _, _ = (
                self.response_alpha_oracle(
                    cls_response,
                    cont_response,
                    gt_probability,
                    tau,
                    self.cfg.MODEL.HEAD.ERC.MAX_ALPHA,
                    self.cfg.TRAIN.ERC_RESPONSE_ALPHA_ORACLE_STEPS,
                )
            )
            response_alpha_gain = response_alpha_gain[:, None]
            gain_scale = max(
                float(self.cfg.TRAIN.ERC_RESPONSE_GAIN_SCALE), 1e-8
            )
            alpha_gain_scale = max(
                float(self.cfg.TRAIN.ERC_RESPONSE_ALPHA_GAIN_SCALE), 1e-8
            )
            normalized_branch_gain = (
                branch_response_gain / gain_scale
            ).clamp(0.0, 1.0)
            normalized_alpha_gain = (
                response_alpha_gain / alpha_gain_scale
            ).clamp(0.0, 1.0)
            gate_target_mode = str(
                self.cfg.TRAIN.ERC_RESPONSE_GATE_TARGET_MODE
            ).lower()
            if gate_target_mode == 'branch_gain':
                gate_target = normalized_branch_gain
            elif gate_target_mode == 'response_alpha_gain':
                gate_target = normalized_alpha_gain
            elif gate_target_mode == 'combined_response':
                gate_target = torch.maximum(
                    normalized_branch_gain, normalized_alpha_gain
                )
            else:
                raise ValueError(
                    'Unsupported ERC_RESPONSE_GATE_TARGET_MODE: %s'
                    % gate_target_mode
                )
            branch_correctable = branch_response_gain.squeeze(1) > 0.0
            alpha_correctable = response_alpha_gain.squeeze(1) > 0.0
            response_correctable = gate_target.squeeze(1) > 0.0
            pref_scale = max(
                float(self.cfg.TRAIN.ERC_RESPONSE_PREF_SCALE), 1e-8
            )
            preference_target = (
                (target_cls - target_cont) / pref_scale
            ).clamp(-1.0, 1.0)

        reliability_cls = pred_dict['erc_reliability_cls'][active_mask]
        reliability_cont = pred_dict['erc_reliability_cont'][active_mask]
        reliability_base = pred_dict['erc_reliability_base'][active_mask]
        intervention_gate = pred_dict['erc_intervention_gate'][active_mask]
        z_global = pred_dict['erc_z_alpha'][active_mask]
        alpha_global = pred_dict['erc_alpha_global'][active_mask]
        alpha_region = pred_dict['erc_alpha_region'][active_mask]
        alpha_final = pred_dict['erc_alpha_final'][active_mask]
        alpha_raw = pred_dict['erc_raw_alpha'][active_mask].reshape(
            -1, height, width
        )
        lambda_gate = self._gate_curriculum_lambda(
            epoch, training=self.net.training
        )
        if self.net.training:
            gate_factor_train = (
                (1.0 - lambda_gate)
                + lambda_gate * intervention_gate
            )
            alpha_train = gate_factor_train[:, :, None] * alpha_raw
        else:
            # Validation always evaluates the real inference alpha, including
            # FORCE_OFF and OFF/GLOBAL/FULL ablation semantics.
            alpha_train = alpha_final
        train_factor = torch.exp(
            alpha_train.flatten(1) * pred_dict['erc_log_ratio'][active_mask]
        )
        train_response = base_response * train_factor
        probability_train = response_probability(train_response)

        reliability_loss = (
            F.smooth_l1_loss(reliability_cls, target_cls)
            + F.smooth_l1_loss(reliability_cont, target_cont)
            + F.smooth_l1_loss(reliability_base, target_base)
        ) / 3.0

        gate_element = F.smooth_l1_loss(
            intervention_gate, gate_target, reduction='none'
        ).squeeze(1)
        positive_indices = response_correctable.nonzero(
            as_tuple=False
        ).squeeze(1)
        negative_indices = (~response_correctable).nonzero(
            as_tuple=False
        ).squeeze(1)
        if positive_indices.numel() > 0:
            positive_gate_loss = gate_element[positive_indices].mean()
            if negative_indices.numel() > 0:
                ratio = max(
                    int(self.cfg.TRAIN.ERC_RESPONSE_GATE_NEG_POS_RATIO), 0
                )
                keep_negative = min(
                    negative_indices.numel(),
                    ratio * positive_indices.numel(),
                )
                if keep_negative > 0:
                    hard_negative_local = gate_element[
                        negative_indices
                    ].topk(keep_negative).indices
                    selected_negative = negative_indices[
                        hard_negative_local
                    ]
                    gate_loss = 0.5 * (
                        positive_gate_loss
                        + gate_element[selected_negative].mean()
                    )
                else:
                    gate_loss = positive_gate_loss
            else:
                gate_loss = positive_gate_loss
        else:
            gate_loss = (
                float(self.cfg.TRAIN.ERC_RESPONSE_GATE_NEG_ONLY_WEIGHT)
                * gate_element.mean()
            )

        def map_cross_entropy(target, prediction):
            return -(
                target * prediction.clamp_min(1e-12).log()
            ).sum(dim=-1)

        map_loss_per_sample = map_cross_entropy(
            gt_probability, probability_train
        )
        with torch.no_grad():
            base_map_loss_per_sample = map_cross_entropy(
                gt_probability, probability_base
            ).detach()
            final_map_loss_per_sample = map_cross_entropy(
                gt_probability, probability_erc
            ).detach()
        map_loss = map_loss_per_sample.mean()
        noharm_loss = F.relu(
            map_loss_per_sample - base_map_loss_per_sample
        ).mean()

        log_erc_response = train_response.clamp_min(1e-12).log()
        positive_score = (
            log_erc_response * gt_probability
        ).sum(dim=-1, keepdim=True)
        negative_score_map = log_erc_response.masked_fill(
            positive_mask, float('-inf')
        )
        topk_negative = min(
            int(self.cfg.TRAIN.ERC_RESPONSE_RANK_TOPK),
            negative_score_map.shape[-1] - 1,
        )
        negative_scores = negative_score_map.topk(
            max(topk_negative, 1), dim=-1
        ).values
        rank_loss = F.softplus(
            float(self.cfg.TRAIN.ERC_RESPONSE_RANK_MARGIN)
            - positive_score
            + negative_scores
        ).mean()

        preference_loss = F.smooth_l1_loss(
            torch.tanh(z_global), preference_target
        )

        with torch.no_grad():
            base_top2 = probability_base.topk(2, dim=-1)
            base_peak_index = base_top2.indices[:, 0]
            base_margin_per_sample = (
                base_top2.values[:, 0] - base_top2.values[:, 1]
            )
            base_peak_positive = positive_mask.gather(
                1, base_peak_index[:, None]
            ).squeeze(1)
            good_base_mask = (
                base_peak_positive
                & (target_base.squeeze(1) > float(
                    self.cfg.TRAIN.ERC_RESPONSE_Q_GOOD
                ))
                & (base_margin_per_sample > float(
                    self.cfg.TRAIN.ERC_RESPONSE_M_GOOD
                ))
            )
        if good_base_mask.any():
            identity_loss = (
                probability_train[good_base_mask]
                - probability_base[good_base_mask].detach()
            ).abs().mean()
        else:
            identity_loss = alpha_train.sum() * 0.0

        vertical_tv = (
            alpha_region[:, 1:, :] - alpha_region[:, :-1, :]
        ).abs().mean()
        horizontal_tv = (
            alpha_region[:, :, 1:] - alpha_region[:, :, :-1]
        ).abs().mean()
        smooth_loss = vertical_tv + horizontal_tv

        noharm_weight, identity_weight = self._safety_curriculum_weights(
            epoch, training=self.net.training
        )

        weighted_loss = (
            self.loss_weight['erc_response_map'] * map_loss
            + self.loss_weight['erc_response_rank'] * rank_loss
            + self.loss_weight['erc_response_rel'] * reliability_loss
            + self.loss_weight['erc_response_gate'] * gate_loss
            + self.loss_weight['erc_response_pref'] * preference_loss
            + noharm_weight * noharm_loss
            + identity_weight * identity_loss
            + self.loss_weight['erc_response_smooth'] * smooth_loss
        )

        with torch.no_grad():
            target_global = (
                probability_global * positive_float
            ).sum(dim=-1, keepdim=True)
            target_train = (
                probability_train * positive_float
            ).sum(dim=-1, keepdim=True)
            target_final = (
                probability_erc * positive_float
            ).sum(dim=-1, keepdim=True)
            train_top2 = probability_train.topk(2, dim=-1).values
            final_top2 = probability_erc.topk(2, dim=-1).values
            train_margin = (
                train_top2[:, 0] - train_top2[:, 1]
            ).mean()
            final_margin = (
                final_top2[:, 0] - final_top2[:, 1]
            ).mean()

            gt_peak = gt_probability.argmax(dim=-1)

            def normalized_peak_distance(probability):
                peak = probability.argmax(dim=-1)
                peak_y = torch.div(peak, width, rounding_mode='floor')
                peak_x = peak % width
                gt_y = torch.div(gt_peak, width, rounding_mode='floor')
                gt_x = gt_peak % width
                return torch.sqrt(
                    (peak_y - gt_y).float().square()
                    + (peak_x - gt_x).float().square()
                ) / (2.0 ** 0.5 * max(max(height, width) - 1, 1))

            near_zero_threshold = max(
                1e-4,
                float(self.cfg.MODEL.HEAD.ERC.MAX_ALPHA) * 1e-3,
            )
            saturation_threshold = (
                float(self.cfg.MODEL.HEAD.ERC.MAX_ALPHA)
                * float(self.cfg.TRAIN.ERC_ALPHA_SATURATION_RATIO)
            )
            gate_pos_mean = (
                intervention_gate[response_correctable].mean()
                if response_correctable.any() else zero.detach()
            )
            gate_neg_mean = (
                intervention_gate[~response_correctable].mean()
                if (~response_correctable).any() else zero.detach()
            )

            # DIAGNOSTIC ONLY - NOT USED FOR ERC TRAINING.
            # Every candidate-IoU operation remains under no_grad and occurs
            # after ``weighted_loss`` has been fully constructed.
            candidate_iou = self.candidate_iou(
                pred_dict['bbox_map'][active_mask].detach(),
                gt_bbox[active_mask],
            )
            baseline_index = pred_dict[
                'erc_windowed_base_score'
            ][active_mask].argmax(dim=-1, keepdim=True)
            erc_index = pred_dict[
                'erc_windowed_score'
            ][active_mask].argmax(dim=-1, keepdim=True)
            baseline_iou = candidate_iou.gather(1, baseline_index)
            calibrated_iou = candidate_iou.gather(1, erc_index)
            candidate_gain = calibrated_iou - baseline_iou
            changed_mask = erc_index.ne(baseline_index)
            changed_gain = candidate_gain[changed_mask]
            beneficial = changed_mask & candidate_gain.gt(1e-7)
            harmful = changed_mask & candidate_gain.lt(-1e-7)
            changed_count = changed_mask.sum().clamp_min(1).float()

        status = {
            'Loss/erc_total': weighted_loss.detach(),
            'Loss/erc_response_map': map_loss.detach(),
            'Loss/erc_response_rank': rank_loss.detach(),
            'Loss/erc_response_reliability': reliability_loss.detach(),
            'Loss/erc_response_gate': gate_loss.detach(),
            'Loss/erc_response_preference': preference_loss.detach(),
            'Loss/erc_response_noharm': noharm_loss.detach(),
            'Loss/erc_response_identity': identity_loss.detach(),
            'Loss/erc_response_smooth': smooth_loss.detach(),
            'ERC/response_map_loss_base': base_map_loss_per_sample.mean(),
            'ERC/response_map_loss_train': map_loss_per_sample.detach().mean(),
            'ERC/response_map_loss_final': final_map_loss_per_sample.mean(),
            # Compatibility alias: inference/final ERC response quality.
            'ERC/response_map_loss_erc': final_map_loss_per_sample.mean(),
            'ERC/response_noharm_violation_rate': (
                map_loss_per_sample.detach() > base_map_loss_per_sample
            ).float().mean(),
            'ERC/response_correctable_rate': response_correctable.float().mean(),
            'ERC/branch_correctable_rate': branch_correctable.float().mean(),
            'ERC/response_alpha_correctable_rate': (
                alpha_correctable.float().mean()
            ),
            'ERC/response_alpha_oracle_gain': response_alpha_gain.mean(),
            'ERC/best_response_alpha': best_response_alpha.mean(),
            'ERC/positive_batch_rate': response_correctable.any().float(),
            'ERC/target_mass_cls': target_cls.mean(),
            'ERC/target_mass_cont': target_cont.mean(),
            'ERC/target_mass_base': target_base.mean(),
            'ERC/target_mass_global': target_global.mean(),
            'ERC/target_mass_train': target_train.mean(),
            'ERC/target_mass_final': target_final.mean(),
            'ERC/target_mass_erc': target_final.mean(),
            'ERC/peak_distance_cls': normalized_peak_distance(
                probability_cls
            ).mean(),
            'ERC/peak_distance_cont': normalized_peak_distance(
                probability_cont
            ).mean(),
            'ERC/peak_distance_base': normalized_peak_distance(
                probability_base
            ).mean(),
            'ERC/peak_distance_train': normalized_peak_distance(
                probability_train
            ).mean(),
            'ERC/peak_distance_final': normalized_peak_distance(
                probability_erc
            ).mean(),
            'ERC/peak_distance_erc': normalized_peak_distance(
                probability_erc
            ).mean(),
            'ERC/base_margin': base_margin_per_sample.mean(),
            'ERC/erc_train_margin': train_margin,
            'ERC/erc_final_margin': final_margin,
            'ERC/erc_margin': final_margin,
            'ERC/gate_pos_mean': gate_pos_mean,
            'ERC/gate_neg_mean': gate_neg_mean,
            'ERC/gate_separation': gate_pos_mean - gate_neg_mean,
            'ERC/reliability_cls': reliability_cls.detach().mean(),
            'ERC/reliability_cont': reliability_cont.detach().mean(),
            'ERC/reliability_base': reliability_base.detach().mean(),
            'ERC/reliability_cls_target': target_cls.mean(),
            'ERC/reliability_cont_target': target_cont.mean(),
            'ERC/reliability_base_target': target_base.mean(),
            'ERC/intervention_gate': intervention_gate.detach().mean(),
            'ERC/gate_mean': intervention_gate.detach().mean(),
            'ERC/intervention_gate_target': gate_target.mean(),
            'ERC/lambda_gate': alpha_final.new_tensor(lambda_gate),
            'ERC/noharm_weight_effective': alpha_final.new_tensor(
                noharm_weight
            ),
            'ERC/identity_weight_effective': alpha_final.new_tensor(
                identity_weight
            ),
            'ERC/alpha_global_abs': alpha_global.detach().abs().mean(),
            'ERC/alpha_region_abs': alpha_region.detach().abs().mean(),
            'ERC/alpha_region_std': alpha_region.detach().std(unbiased=False),
            'ERC/alpha_raw_abs': alpha_raw.detach().abs().mean(),
            'ERC/alpha_train_abs': alpha_train.detach().abs().mean(),
            'ERC/alpha_final_abs': alpha_final.detach().abs().mean(),
            'ERC/alpha_raw_positive_rate': (
                alpha_raw.detach() > near_zero_threshold
            ).float().mean(),
            'ERC/alpha_raw_negative_rate': (
                alpha_raw.detach() < -near_zero_threshold
            ).float().mean(),
            'ERC/alpha_near_zero_rate_raw': (
                alpha_raw.detach().abs() <= near_zero_threshold
            ).float().mean(),
            'ERC/alpha_near_zero_rate_train': (
                alpha_train.detach().abs() <= near_zero_threshold
            ).float().mean(),
            'ERC/alpha_near_zero_rate_final': (
                alpha_final.detach().abs() <= near_zero_threshold
            ).float().mean(),
            'ERC/alpha_near_zero_rate': (
                alpha_final.detach().abs() <= near_zero_threshold
            ).float().mean(),
            'ERC/alpha_saturation_rate_raw': (
                alpha_raw.detach().abs() >= saturation_threshold
            ).float().mean(),
            'ERC/alpha_saturation_rate_final': (
                alpha_final.detach().abs() >= saturation_threshold
            ).float().mean(),
            'ERC/alpha_saturation_rate': (
                alpha_final.detach().abs() >= saturation_threshold
            ).float().mean(),
            'ERC/global_only_response_gain': (
                target_global - target_base
            ).mean(),
            'ERC/full_response_gain': (target_final - target_base).mean(),
            'ERC/selection_change_rate': changed_mask.float().mean(),
            'ERC/candidate_iou_gain': candidate_gain.mean(),
            'ERC/changed_candidate_iou_gain': (
                changed_gain.mean() if changed_gain.numel() else zero.detach()
            ),
            'ERC/beneficial_change_rate': beneficial.sum().float() / changed_count,
            'ERC/harmful_change_rate': harmful.sum().float() / changed_count,
            'ERC/active_fraction': active_fraction.detach(),
        }
        return weighted_loss, status

    def compute_erc_response_v8_losses(self, pred_dict, gt_cls, epoch=1):
        """Hann-aware hard-decision supervision for response-only ERC-v8.

        Targets and hard indices use only C, T, the fixed Hann prior, and the
        official GT-center response heatmap.  Candidate boxes, geometry, and
        IoU are deliberately absent from this method.
        """
        is_v10 = (
            self._response_erc_design == 'response_competition_pair_v10'
        )
        is_v9 = self._response_erc_design in (
            'response_precision_incremental_v9',
            'response_competition_pair_v10',
        )
        active_mask = pred_dict['erc_active_mask'].reshape(-1).bool()
        reference = pred_dict['erc_windowed_base_score']
        zero = reference.sum() * 0.0
        active_fraction = active_mask.float().mean()
        status_names = (
            'Loss/erc_total', 'Loss/erc_v8_total',
            'Loss/erc_v8_hard_correct', 'Loss/erc_v8_map',
            'Loss/erc_v8_reliability', 'Loss/erc_v8_gate',
            'Loss/erc_v8_preference', 'Loss/erc_v8_hard_preserve',
            'Loss/erc_v8_margin_noharm', 'Loss/erc_v8_smooth',
            'Loss/erc_v8_gate_separation',
            'Loss/erc_v8_gate_pairwise',
            'Loss/erc_v8_negative_saturation',
            'Loss/erc_v8_regional_effective',
            'Loss/erc_v8_regional_noharm',
            'Loss/erc_v9_global_oracle',
            'Loss/erc_v9_regional_gate',
            'Loss/erc_v9_regional_identity',
            'Loss/erc_v10_pair_correct',
            'Loss/erc_v10_pair_gate',
            'Loss/erc_v10_pair_preserve',
            'Loss/erc_v10_pair_identity',
            'ERC/pair_correctable_rate',
            'ERC/pair_preserve_rate',
            'ERC/pair_correction_success_rate',
            'ERC/pair_preserve_success_rate',
            'ERC/pair_selection_change_rate',
            'ERC/pair_gate_correctable_mean',
            'ERC/pair_gate_negative_mean',
            'ERC/pair_gate_separation',
            'ERC/pair_beta_abs',
            'ERC/pair_achieved_delta_abs',
            'ERC/pair_base_log_margin',
            'ERC/base_top1_correct_rate', 'ERC/erc_top1_correct_rate',
            'ERC/hard_correctable_rate', 'ERC/hard_unreachable_rate',
            'ERC/hard_reachable_before_boundary_rate',
            'ERC/hard_far_reachable_rate',
            'ERC/boundary_weight_mean',
            'ERC/hard_correctable_batch_rate',
            'ERC/hard_correction_success_rate',
            'ERC/hard_preserve_success_rate',
            'ERC/hard_flip_beneficial_rate',
            'ERC/hard_flip_harmful_rate',
            'ERC/base_hard_gap_mean', 'ERC/erc_hard_gap_mean',
            'ERC/correctable_margin_before',
            'ERC/correctable_margin_after',
            'ERC/already_correct_margin_before',
            'ERC/already_correct_margin_after',
            'ERC/gate_correctable_mean', 'ERC/gate_base_correct_mean',
            'ERC/gate_unreachable_mean', 'ERC/gate_separation',
            'ERC/gate_logit_correctable_mean',
            'ERC/gate_logit_negative_mean',
            'ERC/gate_logit_separation',
            'ERC/global_oracle_alpha_abs',
            'ERC/global_oracle_margin_gain',
            'ERC/global_scalar_correctable_rate',
            'ERC/reliability_cls', 'ERC/reliability_cont',
            'ERC/reliability_base', 'ERC/reliability_cls_target',
            'ERC/reliability_cont_target',
            'ERC/reliability_base_target',
            'ERC/alpha_global_abs', 'ERC/alpha_region_abs',
            'ERC/alpha_region_std', 'ERC/alpha_raw_abs',
            'ERC/alpha_train_abs', 'ERC/alpha_final_abs',
            'ERC/alpha_saturation_rate', 'ERC/alpha_positive_rate',
            'ERC/alpha_negative_rate', 'ERC/gate_mean',
            'ERC/alpha_positive_region_mean',
            'ERC/alpha_negative_peak_mean',
            'ERC/alpha_pos_neg_difference',
            'ERC/regional_global_top1_correct_rate',
            'ERC/regional_correctable_rate',
            'ERC/regional_reachable_margin_mean',
            'ERC/regional_effective_positive',
            'ERC/regional_effective_negative',
            'ERC/regional_effective_difference',
            'ERC/regional_margin_gain',
            'ERC/regional_selection_change_rate',
            'ERC/regional_beneficial_flip_rate',
            'ERC/regional_harmful_flip_rate',
            'ERC/regional_loss_gate_detached',
            'ERC/regional_loss_head_detached',
            'ERC/regional_gate_mean',
            'ERC/regional_gate_correctable_mean',
            'ERC/regional_gate_negative_mean',
            'ERC/regional_gate_separation',
            'ERC/selection_change_rate', 'ERC/lambda_gate',
            'ERC/hard_preserve_weight_effective',
            'ERC/margin_noharm_weight_effective',
            'ERC/bounded_reachable_margin_mean',
            'ERC/log_ratio_abs_mean', 'ERC/positive_log_ratio_mean',
            'ERC/negative_peak_log_ratio_mean',
            'ERC/train_final_score_max_error',
            'Head/gt_margin_drift_mean',
            'Head/gt_margin_drift_abs',
            'ERC/hard_masks_require_grad',
            'ERC/candidate_iou_isolated', 'ERC/active_fraction',
        )
        if not active_mask.any():
            status = {name: zero.detach() for name in status_names}
            status['ERC/candidate_iou_isolated'] = zero.detach() + 1.0
            status['ERC/active_fraction'] = active_fraction.detach()
            return zero, status

        height, width = gt_cls.shape[-2:]
        num_positions = height * width

        def active_flat(name):
            return pred_dict[name].reshape(-1, num_positions)[active_mask]

        cls_response = active_flat('erc_cls_response')
        cont_response = active_flat('erc_cont_response')
        windowed_cls = active_flat('erc_windowed_cls_score')
        windowed_cont = active_flat('erc_windowed_cont_score')
        windowed_base = active_flat('erc_windowed_base_score')
        windowed_global = active_flat('erc_windowed_global_score')
        final_score = active_flat('erc_windowed_score')
        log_ratio = active_flat('erc_log_ratio')
        gt_heatmap = gt_cls.reshape(-1, num_positions)[active_mask].to(
            dtype=cls_response.dtype
        ).clamp_min(0.0)

        # Every discrete target and index is detached from the graph.
        with torch.no_grad():
            if is_v9:
                peak_target = gt_heatmap.max(dim=-1, keepdim=True).values
                positive_mask = (
                    gt_heatmap
                    >= peak_target
                    * float(self.cfg.TRAIN.ERC_V9_POSITIVE_THRESHOLD)
                ).detach()
                positive_mask &= peak_target.gt(0.0)
            else:
                positive_mask = (gt_heatmap > 0).detach()
            missing_positive = ~positive_mask.any(dim=-1)
            if missing_positive.any():
                fallback_index = gt_heatmap.argmax(dim=-1)
                positive_mask[missing_positive] = False
                positive_mask[
                    missing_positive, fallback_index[missing_positive]
                ] = True
            negative_mask = (~positive_mask).detach()

        eps = max(float(self.cfg.MODEL.HEAD.ERC.EPS), 1e-12)
        log_base = windowed_base.clamp_min(eps).log()
        log_final = final_score.clamp_min(eps).log()
        max_alpha = float(self.cfg.MODEL.HEAD.ERC.MAX_ALPHA)

        with torch.no_grad():
            log_base_target = log_base.detach()
            log_ratio_target = log_ratio.detach()
            base_index = log_base_target.argmax(dim=-1)
            base_correct = positive_mask.gather(
                1, base_index[:, None]
            ).squeeze(1).detach()

            positive_base = log_base_target.masked_fill(
                ~positive_mask, float('-inf')
            )
            negative_base = log_base_target.masked_fill(
                ~negative_mask, float('-inf')
            )
            base_positive_score, base_positive_index = positive_base.max(
                dim=-1
            )
            base_negative_score, base_negative_index = negative_base.max(
                dim=-1
            )
            base_gap_target = (
                base_negative_score - base_positive_score
            ).detach()

            positive_upper = (
                log_base_target + max_alpha * log_ratio_target.abs()
            ).masked_fill(~positive_mask, float('-inf'))
            negative_lower = (
                log_base_target - max_alpha * log_ratio_target.abs()
            ).masked_fill(~negative_mask, float('-inf'))
            best_positive_upper, reachable_positive_index = (
                positive_upper.max(dim=-1)
            )
            strongest_negative_lower = negative_lower.max(dim=-1).values
            bounded_reachable_margin = (
                best_positive_upper - strongest_negative_lower
            ).detach()
            hard_reachable_before_boundary = (
                (~base_correct)
                & (
                    bounded_reachable_margin
                    > float(self.cfg.TRAIN.ERC_V8_HARD_REACH_MARGIN)
                )
            ).detach()
            if self.cfg.TRAIN.ERC_V8_BOUNDARY_FOCUS_ENABLE:
                boundary_mask = (
                    base_gap_target
                    <= float(self.cfg.TRAIN.ERC_V8_BOUNDARY_MAX_GAP)
                ).detach()
                hard_correctable = (
                    hard_reachable_before_boundary & boundary_mask
                ).detach()
            else:
                boundary_mask = torch.ones_like(base_correct)
                hard_correctable = hard_reachable_before_boundary
            hard_far_reachable = (
                hard_reachable_before_boundary & (~hard_correctable)
            ).detach()
            hard_unreachable = (
                (~base_correct) & (~hard_correctable)
            ).detach()
            boundary_scale = max(
                float(self.cfg.TRAIN.ERC_V8_BOUNDARY_WEIGHT_SCALE), 1e-8
            )
            boundary_weight = torch.exp(
                -base_gap_target.clamp_min(0.0) / boundary_scale
            ).detach()
            gate_target = hard_correctable.to(
                dtype=cls_response.dtype
            )[:, None].detach()

            global_oracle_alpha_target = log_base_target.new_zeros(
                log_base_target.shape[0]
            )
            global_oracle_margin_gain = log_base_target.new_zeros(
                log_base_target.shape[0]
            )
            global_scalar_correctable = hard_correctable
            if is_v9:
                oracle_steps = max(
                    int(self.cfg.TRAIN.ERC_V9_GLOBAL_ORACLE_STEPS), 3
                )
                if oracle_steps % 2 == 0:
                    oracle_steps += 1
                alpha_grid = torch.linspace(
                    -float(self.cfg.MODEL.HEAD.ERC.GLOBAL_ALPHA_SCALE),
                    float(self.cfg.MODEL.HEAD.ERC.GLOBAL_ALPHA_SCALE),
                    oracle_steps,
                    dtype=log_base_target.dtype,
                    device=log_base_target.device,
                )
                oracle_log_score = (
                    log_base_target[:, None, :]
                    + alpha_grid[None, :, None]
                    * log_ratio_target[:, None, :]
                )
                oracle_positive_score = oracle_log_score.masked_fill(
                    ~positive_mask[:, None, :], float('-inf')
                ).max(dim=-1).values
                oracle_negative_score = oracle_log_score.masked_fill(
                    ~negative_mask[:, None, :], float('-inf')
                ).max(dim=-1).values
                oracle_margin_grid = (
                    oracle_positive_score - oracle_negative_score
                )
                oracle_margin, oracle_alpha_index = oracle_margin_grid.max(
                    dim=-1
                )
                global_oracle_alpha_target = alpha_grid[
                    oracle_alpha_index
                ].detach()
                baseline_margin = (
                    base_positive_score - base_negative_score
                )
                global_oracle_margin_gain = (
                    oracle_margin - baseline_margin
                ).detach()
                best_oracle_score = oracle_log_score[
                    torch.arange(
                        oracle_log_score.shape[0],
                        device=oracle_log_score.device,
                    ),
                    oracle_alpha_index,
                ]
                reachable_positive_index = best_oracle_score.masked_fill(
                    ~positive_mask, float('-inf')
                ).argmax(dim=-1).detach()
                bounded_reachable_margin = oracle_margin.detach()
                hard_reachable_before_boundary = (
                    (~base_correct)
                    & oracle_margin.gt(
                        float(self.cfg.TRAIN.ERC_V8_HARD_REACH_MARGIN)
                    )
                    & global_oracle_margin_gain.gt(
                        float(
                            self.cfg.TRAIN.ERC_V9_GLOBAL_ORACLE_MIN_GAIN
                        )
                    )
                ).detach()
                if self.cfg.TRAIN.ERC_V8_BOUNDARY_FOCUS_ENABLE:
                    boundary_mask = base_gap_target.le(
                        float(self.cfg.TRAIN.ERC_V8_BOUNDARY_MAX_GAP)
                    ).detach()
                    hard_correctable = (
                        hard_reachable_before_boundary & boundary_mask
                    ).detach()
                else:
                    boundary_mask = torch.ones_like(base_correct)
                    hard_correctable = hard_reachable_before_boundary
                global_scalar_correctable = hard_correctable
                hard_far_reachable = (
                    hard_reachable_before_boundary & (~hard_correctable)
                ).detach()
                hard_unreachable = (
                    (~base_correct) & (~hard_correctable)
                ).detach()
                gate_target = hard_correctable.to(
                    dtype=cls_response.dtype
                )[:, None].detach()

            hard_tensors = (
                positive_mask, negative_mask, base_correct,
                hard_reachable_before_boundary, hard_correctable,
                hard_far_reachable, hard_unreachable, boundary_mask,
                boundary_weight, base_index,
                base_negative_index, reachable_positive_index,
                global_oracle_alpha_target,
                global_oracle_margin_gain,
            )
            if any(tensor.requires_grad for tensor in hard_tensors):
                raise AssertionError('ERC-v8 hard masks/indices must detach')

        reliability_cls = pred_dict['erc_reliability_cls'][active_mask]
        reliability_cont = pred_dict['erc_reliability_cont'][active_mask]
        reliability_base = pred_dict['erc_reliability_base'][active_mask]
        intervention_gate = pred_dict['erc_intervention_gate'][active_mask]
        gate_logit = pred_dict['erc_gate_logit'][active_mask]
        regional_intervention_gate = pred_dict.get(
            'erc_regional_intervention_gate',
            pred_dict['erc_intervention_gate'],
        )[active_mask]
        regional_gate_logit = pred_dict.get(
            'erc_regional_gate_logit', pred_dict['erc_gate_logit']
        )[active_mask]
        z_global = pred_dict['erc_z_alpha'][active_mask]
        alpha_global = pred_dict['erc_alpha_global'][active_mask]
        alpha_region = pred_dict['erc_alpha_region'][active_mask].reshape(
            -1, height, width
        )
        alpha_raw = active_flat('erc_raw_alpha')
        alpha_final = active_flat('erc_alpha_final')

        lambda_gate = self._gate_curriculum_lambda(
            epoch, training=self.net.training
        )
        if is_v9:
            # v9 already composes independent Global and Regional Gates in
            # alpha_final.  Global losses below deliberately use only the
            # actual shared-scalar Global path.
            alpha_train = alpha_final
        elif (
            self.net.training
            and self.cfg.TRAIN.ERC_V8_MATCH_INFERENCE_GATE_TRAIN
        ):
            # v8b trains exactly the Gate-scaled alpha used by validation and
            # tracking.  The original v8 curriculum remains available through
            # the fallback branch for old configurations.
            alpha_train = intervention_gate * alpha_raw
        elif self.net.training:
            gate_factor_train = (
                (1.0 - lambda_gate)
                + lambda_gate * intervention_gate
            )
            alpha_train = gate_factor_train * alpha_raw
        else:
            # Validation always follows the real inference gate and mode.
            alpha_train = alpha_final

        # The optional v8b detach prevents ERC-specific supervision from being
        # satisfied by moving the original Head.  The official UVLTrack losses
        # still update the Head normally, so this remains Head+ERC training.
        if self.cfg.TRAIN.ERC_V8_DETACH_HEAD_IN_ERC_LOSS:
            loss_windowed_base = windowed_base.detach()
            loss_log_ratio = log_ratio.detach()
        else:
            loss_windowed_base = windowed_base
            loss_log_ratio = log_ratio
        if is_v9:
            global_alpha_train = (
                intervention_gate * alpha_global
            ).expand(-1, num_positions)
            train_score = loss_windowed_base * torch.exp(
                global_alpha_train * loss_log_ratio
            )
        else:
            train_score = loss_windowed_base * torch.exp(
                alpha_train * loss_log_ratio
            )
        log_train = train_score.clamp_min(eps).log()

        with torch.no_grad():
            current_negative_index = log_train.detach().masked_fill(
                ~negative_mask, float('-inf')
            ).argmax(dim=-1).detach()

        def gather(score, index):
            return score.gather(1, index[:, None]).squeeze(1)

        # v8d supervises only the Regional contribution actually applied at
        # inference: log(S_FULL) - log(S_GLOBAL_ONLY).  Gate, global alpha and
        # C/T responses are detached in this new loss, so it cannot improve by
        # moving the Gate, the Global branch, or the original UVLTrack Head.
        alpha_region_flat = alpha_region.flatten(1)
        detached_gate = intervention_gate.detach()
        detached_global = alpha_global.detach()
        if is_v9:
            regional_effective_alpha_live = (
                regional_intervention_gate.detach() * alpha_region_flat
            )
            regional_effective_alpha_actual = (
                regional_intervention_gate * alpha_region_flat
            )
        else:
            regional_effective_alpha_live = detached_gate * (
                (
                    detached_global + alpha_region_flat
                ).clamp(-max_alpha, max_alpha)
                - detached_global
            )
            regional_effective_alpha_actual = regional_effective_alpha_live
        regional_effective_delta_live = (
            regional_effective_alpha_live * loss_log_ratio.detach()
        )
        regional_effective_delta_actual = (
            regional_effective_alpha_actual * loss_log_ratio.detach()
        )

        with torch.no_grad():
            log_global_target = windowed_global.detach().clamp_min(eps).log()
            global_index = log_global_target.argmax(dim=-1)
            global_correct = positive_mask.gather(
                1, global_index[:, None]
            ).squeeze(1).detach()
            global_positive_score, global_positive_index = (
                log_global_target.masked_fill(
                    ~positive_mask, float('-inf')
                ).max(dim=-1)
            )
            global_negative_score, global_negative_index = (
                log_global_target.masked_fill(
                    ~negative_mask, float('-inf')
                ).max(dim=-1)
            )
            global_gap_target = (
                global_negative_score - global_positive_score
            ).detach()

            region_limit = float(
                self.cfg.MODEL.HEAD.ERC.REGION_ALPHA_SCALE
            )
            if is_v9:
                regional_low_effective = log_ratio_target.new_full(
                    (log_ratio_target.shape[0], 1), -region_limit
                )
                regional_high_effective = log_ratio_target.new_full(
                    (log_ratio_target.shape[0], 1), region_limit
                )
                regional_reference_log = log_global_target
            else:
                regional_low_alpha = (
                    detached_global - region_limit
                ).clamp(-max_alpha, max_alpha)
                regional_high_alpha = (
                    detached_global + region_limit
                ).clamp(-max_alpha, max_alpha)
                regional_low_effective = detached_gate * regional_low_alpha
                regional_high_effective = detached_gate * regional_high_alpha
                regional_reference_log = log_base_target
            regional_upper_adjustment = torch.maximum(
                regional_low_effective * loss_log_ratio.detach(),
                regional_high_effective * loss_log_ratio.detach(),
            )
            regional_lower_adjustment = torch.minimum(
                regional_low_effective * loss_log_ratio.detach(),
                regional_high_effective * loss_log_ratio.detach(),
            )
            regional_positive_upper, regional_target_positive_index = (
                (
                    regional_reference_log + regional_upper_adjustment
                ).masked_fill(
                    ~positive_mask, float('-inf')
                ).max(dim=-1)
            )
            regional_negative_lower = (
                regional_reference_log + regional_lower_adjustment
            ).masked_fill(
                ~negative_mask, float('-inf')
            ).max(dim=-1).values
            regional_reachable_margin = (
                regional_positive_upper - regional_negative_lower
            ).detach()
            if self.cfg.TRAIN.ERC_V8_BOUNDARY_FOCUS_ENABLE:
                regional_boundary = global_gap_target.le(
                    float(self.cfg.TRAIN.ERC_V8_BOUNDARY_MAX_GAP)
                )
            else:
                regional_boundary = torch.ones_like(global_correct)
            regional_correctable = (
                (~global_correct)
                & regional_boundary
                & regional_reachable_margin.gt(
                    float(self.cfg.TRAIN.ERC_V8_HARD_REACH_MARGIN)
                )
            ).detach()
            regional_boundary_weight = torch.exp(
                -global_gap_target.clamp_min(0.0)
                / max(
                    float(self.cfg.TRAIN.ERC_V8_BOUNDARY_WEIGHT_SCALE),
                    1e-8,
                )
            ).detach()

        regional_positive_live = gather(
            regional_effective_delta_live,
            regional_target_positive_index,
        )
        regional_negative_live = gather(
            regional_effective_delta_live,
            global_negative_index,
        )
        regional_difference_live = (
            regional_positive_live - regional_negative_live
        )
        if regional_correctable.any():
            regional_effective_element = F.softplus(
                float(self.cfg.TRAIN.ERC_V8_REGIONAL_EFFECTIVE_MARGIN)
                - regional_difference_live[regional_correctable]
            )
            selected_regional_weight = regional_boundary_weight[
                regional_correctable
            ]
            regional_effective_loss = (
                regional_effective_element * selected_regional_weight
            ).sum() / selected_regional_weight.sum().clamp_min(1e-8)
        else:
            regional_effective_loss = regional_effective_delta_live.sum() * 0.0

        preserve_regional_positive = gather(
            regional_effective_delta_live, global_positive_index
        )
        preserve_regional_negative = gather(
            regional_effective_delta_live, global_negative_index
        )
        preserve_regional_difference = (
            preserve_regional_positive - preserve_regional_negative
        )
        if global_correct.any():
            regional_noharm_loss = F.relu(
                -float(self.cfg.TRAIN.ERC_V8_REGIONAL_NOHARM_TOLERANCE)
                - preserve_regional_difference[global_correct]
            ).mean()
        else:
            regional_noharm_loss = regional_effective_delta_live.sum() * 0.0

        if is_v9:
            regional_gate_target = regional_correctable.to(
                dtype=regional_intervention_gate.dtype
            )[:, None]
            regional_gate_element = F.binary_cross_entropy(
                regional_intervention_gate,
                regional_gate_target,
                reduction='none',
            ).squeeze(1)
            regional_positive_indices = regional_correctable.nonzero(
                as_tuple=False
            ).squeeze(1)
            regional_negative_indices = (~regional_correctable).nonzero(
                as_tuple=False
            ).squeeze(1)
            if regional_positive_indices.numel() > 0:
                regional_positive_gate_loss = regional_gate_element[
                    regional_positive_indices
                ].mean()
                regional_ratio = max(
                    int(
                        self.cfg.TRAIN.ERC_V9_REGIONAL_GATE_NEG_POS_RATIO
                    ),
                    0,
                )
                regional_keep_negative = min(
                    regional_negative_indices.numel(),
                    regional_ratio * regional_positive_indices.numel(),
                )
                if regional_keep_negative > 0:
                    regional_negative_local = regional_gate_element[
                        regional_negative_indices
                    ].topk(regional_keep_negative).indices
                    regional_selected_negative = regional_negative_indices[
                        regional_negative_local
                    ]
                    regional_gate_loss = 0.5 * (
                        regional_positive_gate_loss
                        + regional_gate_element[
                            regional_selected_negative
                        ].mean()
                    )
                else:
                    regional_gate_loss = regional_positive_gate_loss
            else:
                regional_gate_loss = (
                    float(
                        self.cfg.TRAIN.ERC_V9_REGIONAL_GATE_NEG_ONLY_WEIGHT
                    )
                    * regional_gate_element.mean()
                )

            log_full_live = (
                log_global_target + regional_effective_delta_actual
            )
            regional_negative_target = ~regional_correctable
            if regional_negative_target.any():
                regional_reference_probability = F.softmax(
                    log_global_target[regional_negative_target], dim=-1
                ).detach()
                regional_identity_loss = F.kl_div(
                    F.log_softmax(
                        log_full_live[regional_negative_target], dim=-1
                    ),
                    regional_reference_probability,
                    reduction='batchmean',
                )
            else:
                regional_identity_loss = log_full_live.sum() * 0.0
        else:
            regional_gate_loss = regional_effective_delta_live.sum() * 0.0
            regional_identity_loss = regional_effective_delta_live.sum() * 0.0

        hard_pos_live = gather(log_train, reachable_positive_index)
        hard_neg_live = gather(log_train, current_negative_index)
        if hard_correctable.any():
            hard_correct_element = F.softplus(
                float(self.cfg.TRAIN.ERC_V8_HARD_SCORE_MARGIN)
                + hard_neg_live[hard_correctable]
                - hard_pos_live[hard_correctable]
            )
            selected_boundary_weight = boundary_weight[hard_correctable]
            hard_correct_loss = (
                hard_correct_element * selected_boundary_weight
            ).sum() / selected_boundary_weight.sum().clamp_min(1e-8)
        else:
            hard_correct_loss = log_train.sum() * 0.0

        if is_v9 and global_scalar_correctable.any():
            global_oracle_loss = F.smooth_l1_loss(
                alpha_global.squeeze(1)[global_scalar_correctable],
                global_oracle_alpha_target[global_scalar_correctable],
            )
        else:
            global_oracle_loss = alpha_global.sum() * 0.0

        preserve_pos_live = gather(log_train, base_index)
        preserve_neg_live = gather(log_train, base_negative_index)
        if base_correct.any():
            hard_preserve_loss = F.softplus(
                float(self.cfg.TRAIN.ERC_V8_PRESERVE_MARGIN)
                + preserve_neg_live[base_correct]
                - preserve_pos_live[base_correct]
            ).mean()
            baseline_preserve_margin = (
                gather(log_base, base_index)
                - gather(log_base, base_negative_index)
            ).detach()
            erc_preserve_margin = preserve_pos_live - preserve_neg_live
            margin_noharm_loss = F.relu(
                baseline_preserve_margin[base_correct]
                - erc_preserve_margin[base_correct]
                - float(self.cfg.TRAIN.ERC_V8_MARGIN_TOLERANCE)
            ).mean()
        else:
            hard_preserve_loss = log_train.sum() * 0.0
            margin_noharm_loss = log_train.sum() * 0.0

        tau = max(float(self.cfg.TRAIN.ERC_V8_TAU), 1e-6)
        gt_probability = gt_heatmap / gt_heatmap.sum(
            dim=-1, keepdim=True
        ).clamp_min(eps)
        train_log_probability = F.log_softmax(log_train / tau, dim=-1)
        map_loss = -(
            gt_probability * train_log_probability
        ).sum(dim=-1).mean()

        def response_probability_from_windowed(response):
            return F.softmax(
                response.clamp_min(eps).log() / tau, dim=-1
            )

        with torch.no_grad():
            positive_float = positive_mask.to(cls_response.dtype)
            cls_probability = response_probability_from_windowed(
                windowed_cls.detach()
            )
            cont_probability = response_probability_from_windowed(
                windowed_cont.detach()
            )
            base_probability = response_probability_from_windowed(
                windowed_base.detach()
            )
            target_cls = (
                cls_probability * positive_float
            ).sum(dim=-1, keepdim=True).detach()
            target_cont = (
                cont_probability * positive_float
            ).sum(dim=-1, keepdim=True).detach()
            target_base = (
                base_probability * positive_float
            ).sum(dim=-1, keepdim=True).detach()
            pref_scale = max(
                float(self.cfg.TRAIN.ERC_V8_PREF_SCALE), 1e-8
            )
            preference_target = (
                (target_cls - target_cont) / pref_scale
            ).clamp(-1.0, 1.0).detach()

        reliability_loss = (
            F.smooth_l1_loss(reliability_cls, target_cls)
            + F.smooth_l1_loss(reliability_cont, target_cont)
            + F.smooth_l1_loss(reliability_base, target_base)
        ) / 3.0
        preference_loss = F.smooth_l1_loss(
            torch.tanh(z_global), preference_target
        )

        gate_element = F.binary_cross_entropy(
            intervention_gate, gate_target, reduction='none'
        ).squeeze(1)
        positive_indices = hard_correctable.nonzero(
            as_tuple=False
        ).squeeze(1)
        negative_indices = (~hard_correctable).nonzero(
            as_tuple=False
        ).squeeze(1)
        if positive_indices.numel() > 0:
            positive_gate_loss = gate_element[positive_indices].mean()
            ratio = max(
                int(self.cfg.TRAIN.ERC_V8_GATE_NEG_POS_RATIO), 0
            )
            keep_negative = min(
                negative_indices.numel(),
                ratio * positive_indices.numel(),
            )
            if keep_negative > 0:
                negative_local = gate_element[
                    negative_indices
                ].topk(keep_negative).indices
                selected_negative = negative_indices[negative_local]
                gate_loss = 0.5 * (
                    positive_gate_loss
                    + gate_element[selected_negative].mean()
                )
            else:
                gate_loss = positive_gate_loss
        else:
            gate_loss = (
                float(self.cfg.TRAIN.ERC_V8_GATE_NEG_ONLY_WEIGHT)
                * gate_element.mean()
            )

        if positive_indices.numel() > 0 and negative_indices.numel() > 0:
            positive_gate_mean_live = intervention_gate[
                positive_indices
            ].mean()
            separation_negative_count = min(
                negative_indices.numel(),
                max(positive_indices.numel(), 1)
                * max(int(self.cfg.TRAIN.ERC_V8_GATE_NEG_POS_RATIO), 1),
            )
            hard_negative_gate = intervention_gate[
                negative_indices
            ].squeeze(1).topk(separation_negative_count).values.mean()
            gate_separation_loss = F.relu(
                float(self.cfg.TRAIN.ERC_V8_GATE_SEPARATION_MARGIN)
                - positive_gate_mean_live
                + hard_negative_gate
            )
        else:
            gate_separation_loss = intervention_gate.sum() * 0.0

        if positive_indices.numel() > 0 and negative_indices.numel() > 0:
            pairwise_negative_count = min(
                negative_indices.numel(),
                max(positive_indices.numel(), 1)
                * max(int(self.cfg.TRAIN.ERC_V8_GATE_NEG_POS_RATIO), 1),
            )
            hard_negative_logits = gate_logit[
                negative_indices
            ].squeeze(1).topk(pairwise_negative_count).values
            positive_logits = gate_logit[positive_indices].squeeze(1)
            gate_pairwise_loss = F.softplus(
                float(self.cfg.TRAIN.ERC_V8_GATE_PAIRWISE_MARGIN)
                - positive_logits[:, None]
                + hard_negative_logits[None, :]
            ).mean()
        else:
            gate_pairwise_loss = gate_logit.sum() * 0.0

        saturation_start = (
            max_alpha
            * float(self.cfg.TRAIN.ERC_V8_NEGATIVE_SATURATION_START)
        )
        negative_saturation_mask = (~hard_correctable)[:, None].expand_as(
            alpha_raw
        )
        if negative_saturation_mask.any() and max_alpha > saturation_start:
            saturation_excess = F.relu(
                alpha_raw.abs() - saturation_start
            ) / max(max_alpha - saturation_start, 1e-8)
            negative_saturation_loss = saturation_excess[
                negative_saturation_mask
            ].square().mean()
        else:
            negative_saturation_loss = alpha_raw.sum() * 0.0

        vertical_tv = (
            alpha_region[:, 1:, :] - alpha_region[:, :-1, :]
        ).abs().mean()
        horizontal_tv = (
            alpha_region[:, :, 1:] - alpha_region[:, :, :-1]
        ).abs().mean()
        smooth_loss = vertical_tv + horizontal_tv

        # ERC-v10 applies Regional calibration only to the Hann-aware baseline
        # peak and its strongest spatially distinct response competitor.  The
        # hard indices and GT-derived masks are detached; gradients act on the
        # exact FULL-vs-GLOBAL score difference used at inference.
        pair_correct_loss = zero
        pair_gate_loss = zero
        pair_preserve_loss = zero
        pair_identity_loss = zero
        pair_correctable = active_mask.new_zeros(
            cls_response.shape[0], dtype=torch.bool
        )
        pair_preserve = pair_correctable.clone()
        pair_top_index = base_index
        pair_competitor_index = base_negative_index
        pair_beta = regional_intervention_gate.new_zeros(
            regional_intervention_gate.shape
        )
        pair_achieved_delta = pair_beta.clone()
        pair_base_log_margin = pair_beta.clone()
        if is_v10:
            pair_top_index = pred_dict['erc_pair_top_index'].reshape(-1)[
                active_mask
            ].long().detach()
            pair_competitor_index = pred_dict[
                'erc_pair_competitor_index'
            ].reshape(-1)[active_mask].long().detach()
            pair_beta = pred_dict['erc_pair_beta'].reshape(-1, 1)[
                active_mask
            ]
            pair_achieved_delta = pred_dict[
                'erc_pair_achieved_delta'
            ].reshape(-1, 1)[active_mask]
            pair_base_log_margin = pred_dict[
                'erc_pair_base_log_margin'
            ].reshape(-1, 1)[active_mask]
            with torch.no_grad():
                pair_top_positive = positive_mask.gather(
                    1, pair_top_index[:, None]
                ).squeeze(1)
                pair_competitor_positive = positive_mask.gather(
                    1, pair_competitor_index[:, None]
                ).squeeze(1)
                pair_correctable = (
                    (~pair_top_positive) & pair_competitor_positive
                ).detach()
                pair_preserve = pair_top_positive.detach()
                pair_gate_target = pair_correctable.to(
                    regional_intervention_gate.dtype
                )[:, None]
            if pair_top_index.requires_grad or pair_competitor_index.requires_grad:
                raise AssertionError('ERC-v10 pair indices must detach')

            pair_log_full = (
                loss_windowed_base.clamp_min(eps).log()
                + alpha_final * loss_log_ratio
            )
            pair_full_top = gather(pair_log_full, pair_top_index)
            pair_full_competitor = gather(
                pair_log_full, pair_competitor_index
            )
            if pair_correctable.any():
                pair_correct_loss = F.softplus(
                    float(self.cfg.TRAIN.ERC_V10_PAIR_SCORE_MARGIN)
                    + pair_full_top[pair_correctable]
                    - pair_full_competitor[pair_correctable]
                ).mean()
            if pair_preserve.any():
                pair_preserve_loss = F.softplus(
                    float(self.cfg.TRAIN.ERC_V10_PAIR_PRESERVE_MARGIN)
                    + pair_full_competitor[pair_preserve]
                    - pair_full_top[pair_preserve]
                ).mean()

            pair_gate_element = F.binary_cross_entropy(
                regional_intervention_gate,
                pair_gate_target,
                reduction='none',
            ).squeeze(1)
            pair_positive_indices = pair_correctable.nonzero(
                as_tuple=False
            ).squeeze(1)
            pair_negative_indices = (~pair_correctable).nonzero(
                as_tuple=False
            ).squeeze(1)
            if pair_positive_indices.numel() > 0:
                pair_positive_gate_loss = pair_gate_element[
                    pair_positive_indices
                ].mean()
                pair_negative_count = min(
                    pair_negative_indices.numel(),
                    int(self.cfg.TRAIN.ERC_V10_PAIR_GATE_NEG_POS_RATIO)
                    * pair_positive_indices.numel(),
                )
                if pair_negative_count > 0:
                    selected_negative_local = pair_gate_element[
                        pair_negative_indices
                    ].topk(pair_negative_count).indices
                    selected_negative = pair_negative_indices[
                        selected_negative_local
                    ]
                    pair_gate_loss = 0.5 * (
                        pair_positive_gate_loss
                        + pair_gate_element[selected_negative].mean()
                    )
                else:
                    pair_gate_loss = pair_positive_gate_loss
            else:
                pair_gate_loss = (
                    float(
                        self.cfg.TRAIN.ERC_V10_PAIR_GATE_NEG_ONLY_WEIGHT
                    )
                    * pair_gate_element.mean()
                )

            pair_identity_mask = ~pair_correctable
            if pair_identity_mask.any():
                pair_identity_loss = regional_effective_delta_actual[
                    pair_identity_mask
                ].square().mean()

        preserve_weight, noharm_weight = (
            self._v8_safety_curriculum_weights(
                epoch, training=self.net.training
            )
        )
        weighted_loss = (
            self.loss_weight['erc_v8_hard_correct'] * hard_correct_loss
            + self.loss_weight['erc_v8_map'] * map_loss
            + self.loss_weight['erc_v8_rel'] * reliability_loss
            + self.loss_weight['erc_v8_gate'] * gate_loss
            + self.loss_weight['erc_v8_pref'] * preference_loss
            + preserve_weight * hard_preserve_loss
            + noharm_weight * margin_noharm_loss
            + self.loss_weight['erc_v8_smooth'] * smooth_loss
            + self.loss_weight['erc_v8_gate_separation']
            * gate_separation_loss
            + self.loss_weight['erc_v8_gate_pairwise']
            * gate_pairwise_loss
            + self.loss_weight['erc_v8_negative_saturation']
            * negative_saturation_loss
            + self.loss_weight['erc_v8_regional_effective']
            * regional_effective_loss
            + self.loss_weight['erc_v8_regional_noharm']
            * regional_noharm_loss
            + self.loss_weight['erc_v9_global_oracle']
            * global_oracle_loss
            + self.loss_weight['erc_v9_regional_gate']
            * regional_gate_loss
            + self.loss_weight['erc_v9_regional_identity']
            * regional_identity_loss
            + self.loss_weight['erc_v10_pair_correct']
            * pair_correct_loss
            + self.loss_weight['erc_v10_pair_gate']
            * pair_gate_loss
            + self.loss_weight['erc_v10_pair_preserve']
            * pair_preserve_loss
            + self.loss_weight['erc_v10_pair_identity']
            * pair_identity_loss
        )

        with torch.no_grad():
            final_index = log_final.argmax(dim=-1)
            final_correct = positive_mask.gather(
                1, final_index[:, None]
            ).squeeze(1)
            final_positive_score = log_final.masked_fill(
                ~positive_mask, float('-inf')
            ).max(dim=-1).values
            final_negative_score = log_final.masked_fill(
                ~negative_mask, float('-inf')
            ).max(dim=-1).values
            global_final_index = log_global_target.argmax(dim=-1)
            actual_regional_delta = log_final - log_global_target
            actual_regional_positive = gather(
                actual_regional_delta, regional_target_positive_index
            )
            actual_regional_negative = gather(
                actual_regional_delta, global_negative_index
            )
            global_positive_final = log_global_target.masked_fill(
                ~positive_mask, float('-inf')
            ).max(dim=-1).values
            global_negative_final = log_global_target.masked_fill(
                ~negative_mask, float('-inf')
            ).max(dim=-1).values
            regional_margin_gain_per_sample = (
                final_positive_score - final_negative_score
                - global_positive_final + global_negative_final
            )
            regional_selection_changed = final_index.ne(global_final_index)
            regional_beneficial_flip = (~global_correct) & final_correct
            regional_harmful_flip = global_correct & (~final_correct)
            base_hard_gap = base_negative_score - base_positive_score
            final_hard_gap = final_negative_score - final_positive_score

            def masked_mean(values, mask):
                return values[mask].mean() if mask.any() else zero.detach()

            correction_success = masked_mean(
                final_correct.float(), hard_correctable
            )
            preserve_success = masked_mean(
                final_correct.float(), base_correct
            )
            beneficial_flip = ((~base_correct) & final_correct).float().mean()
            harmful_flip = (base_correct & (~final_correct)).float().mean()
            correctable_before = masked_mean(
                -base_hard_gap, hard_correctable
            )
            correctable_after = masked_mean(
                -final_hard_gap, hard_correctable
            )
            correct_before = masked_mean(-base_hard_gap, base_correct)
            correct_after = masked_mean(-final_hard_gap, base_correct)

            gate_correctable = masked_mean(
                intervention_gate.squeeze(1), hard_correctable
            )
            gate_base_correct = masked_mean(
                intervention_gate.squeeze(1), base_correct
            )
            gate_unreachable = masked_mean(
                intervention_gate.squeeze(1), hard_unreachable
            )
            negative_gate_mask = base_correct | hard_unreachable
            gate_negative = masked_mean(
                intervention_gate.squeeze(1), negative_gate_mask
            )
            gate_logit_correctable = masked_mean(
                gate_logit.squeeze(1), hard_correctable
            )
            gate_logit_negative = masked_mean(
                gate_logit.squeeze(1), negative_gate_mask
            )
            regional_gate_correctable = masked_mean(
                regional_intervention_gate.squeeze(1),
                regional_correctable,
            )
            regional_gate_negative = masked_mean(
                regional_intervention_gate.squeeze(1),
                ~regional_correctable,
            )
            pair_correction_success = masked_mean(
                final_index.eq(pair_competitor_index).float(),
                pair_correctable,
            )
            pair_preserve_success = masked_mean(
                final_index.eq(pair_top_index).float(),
                pair_preserve,
            )
            pair_selection_change = final_index.ne(
                global_final_index
            ).float().mean()
            pair_gate_correctable = masked_mean(
                regional_intervention_gate.squeeze(1),
                pair_correctable,
            )
            pair_gate_negative = masked_mean(
                regional_intervention_gate.squeeze(1),
                ~pair_correctable,
            )

            alpha_region_flat = alpha_region.flatten(1)
            positive_count = positive_float.sum().clamp_min(1.0)
            alpha_positive_region_mean = (
                alpha_region_flat * positive_float
            ).sum() / positive_count
            alpha_negative_peak_mean = gather(
                alpha_region_flat, base_negative_index
            ).mean()
            positive_log_ratio = (
                log_ratio * positive_float
            ).sum() / positive_count
            negative_peak_log_ratio = gather(
                log_ratio, base_negative_index
            ).mean()
            saturation_threshold = (
                max_alpha
                * float(self.cfg.TRAIN.ERC_ALPHA_SATURATION_RATIO)
            )
            validation_reference_score = (
                windowed_global if is_v9 else final_score
            )
            validation_score_error = (
                train_score - validation_reference_score
            ).abs().max() if not self.net.training else zero.detach()
            if 'head_teacher_windowed_base_score' in pred_dict:
                teacher_log_base = active_flat(
                    'head_teacher_windowed_base_score'
                ).clamp_min(eps).log()
                teacher_positive_score = teacher_log_base.masked_fill(
                    ~positive_mask, float('-inf')
                ).max(dim=-1).values
                teacher_negative_score = teacher_log_base.masked_fill(
                    ~negative_mask, float('-inf')
                ).max(dim=-1).values
                teacher_margin = (
                    teacher_positive_score - teacher_negative_score
                )
                student_margin = base_positive_score - base_negative_score
                gt_margin_drift = student_margin - teacher_margin
                head_gt_margin_drift_mean = gt_margin_drift.mean()
                head_gt_margin_drift_abs = gt_margin_drift.abs().mean()
            else:
                head_gt_margin_drift_mean = zero.detach()
                head_gt_margin_drift_abs = zero.detach()

        status = {
            'Loss/erc_total': weighted_loss.detach(),
            'Loss/erc_v8_total': weighted_loss.detach(),
            'Loss/erc_v8_hard_correct': hard_correct_loss.detach(),
            'Loss/erc_v8_map': map_loss.detach(),
            'Loss/erc_v8_reliability': reliability_loss.detach(),
            'Loss/erc_v8_gate': gate_loss.detach(),
            'Loss/erc_v8_preference': preference_loss.detach(),
            'Loss/erc_v8_hard_preserve': hard_preserve_loss.detach(),
            'Loss/erc_v8_margin_noharm': margin_noharm_loss.detach(),
            'Loss/erc_v8_smooth': smooth_loss.detach(),
            'Loss/erc_v8_gate_separation': gate_separation_loss.detach(),
            'Loss/erc_v8_gate_pairwise': gate_pairwise_loss.detach(),
            'Loss/erc_v8_negative_saturation': (
                negative_saturation_loss.detach()
            ),
            'Loss/erc_v8_regional_effective': (
                regional_effective_loss.detach()
            ),
            'Loss/erc_v8_regional_noharm': regional_noharm_loss.detach(),
            'Loss/erc_v9_global_oracle': global_oracle_loss.detach(),
            'Loss/erc_v9_regional_gate': regional_gate_loss.detach(),
            'Loss/erc_v9_regional_identity': (
                regional_identity_loss.detach()
            ),
            'Loss/erc_v10_pair_correct': pair_correct_loss.detach(),
            'Loss/erc_v10_pair_gate': pair_gate_loss.detach(),
            'Loss/erc_v10_pair_preserve': pair_preserve_loss.detach(),
            'Loss/erc_v10_pair_identity': pair_identity_loss.detach(),
            'ERC/pair_correctable_rate': pair_correctable.float().mean(),
            'ERC/pair_preserve_rate': pair_preserve.float().mean(),
            'ERC/pair_correction_success_rate': pair_correction_success,
            'ERC/pair_preserve_success_rate': pair_preserve_success,
            'ERC/pair_selection_change_rate': pair_selection_change,
            'ERC/pair_gate_correctable_mean': pair_gate_correctable,
            'ERC/pair_gate_negative_mean': pair_gate_negative,
            'ERC/pair_gate_separation': (
                pair_gate_correctable - pair_gate_negative
            ),
            'ERC/pair_beta_abs': pair_beta.detach().abs().mean(),
            'ERC/pair_achieved_delta_abs': (
                pair_achieved_delta.detach().abs().mean()
            ),
            'ERC/pair_base_log_margin': pair_base_log_margin.detach().mean(),
            'ERC/base_top1_correct_rate': base_correct.float().mean(),
            'ERC/erc_top1_correct_rate': final_correct.float().mean(),
            'ERC/hard_correctable_rate': hard_correctable.float().mean(),
            'ERC/hard_unreachable_rate': hard_unreachable.float().mean(),
            'ERC/hard_reachable_before_boundary_rate': (
                hard_reachable_before_boundary.float().mean()
            ),
            'ERC/hard_far_reachable_rate': hard_far_reachable.float().mean(),
            'ERC/boundary_weight_mean': masked_mean(
                boundary_weight, hard_correctable
            ),
            'ERC/hard_correctable_batch_rate': hard_correctable.any().float(),
            'ERC/hard_correction_success_rate': correction_success,
            'ERC/hard_preserve_success_rate': preserve_success,
            'ERC/hard_flip_beneficial_rate': beneficial_flip,
            'ERC/hard_flip_harmful_rate': harmful_flip,
            'ERC/base_hard_gap_mean': base_hard_gap.mean(),
            'ERC/erc_hard_gap_mean': final_hard_gap.mean(),
            'ERC/correctable_margin_before': correctable_before,
            'ERC/correctable_margin_after': correctable_after,
            'ERC/already_correct_margin_before': correct_before,
            'ERC/already_correct_margin_after': correct_after,
            'ERC/gate_correctable_mean': gate_correctable,
            'ERC/gate_base_correct_mean': gate_base_correct,
            'ERC/gate_unreachable_mean': gate_unreachable,
            'ERC/gate_separation': gate_correctable - gate_negative,
            'ERC/gate_logit_correctable_mean': gate_logit_correctable,
            'ERC/gate_logit_negative_mean': gate_logit_negative,
            'ERC/gate_logit_separation': (
                gate_logit_correctable - gate_logit_negative
            ),
            'ERC/global_oracle_alpha_abs': (
                global_oracle_alpha_target.abs().mean()
            ),
            'ERC/global_oracle_margin_gain': (
                global_oracle_margin_gain.mean()
            ),
            'ERC/global_scalar_correctable_rate': (
                global_scalar_correctable.float().mean()
            ),
            'ERC/reliability_cls': reliability_cls.detach().mean(),
            'ERC/reliability_cont': reliability_cont.detach().mean(),
            'ERC/reliability_base': reliability_base.detach().mean(),
            'ERC/reliability_cls_target': target_cls.mean(),
            'ERC/reliability_cont_target': target_cont.mean(),
            'ERC/reliability_base_target': target_base.mean(),
            'ERC/alpha_global_abs': alpha_global.detach().abs().mean(),
            'ERC/alpha_region_abs': alpha_region.detach().abs().mean(),
            'ERC/alpha_region_std': alpha_region.detach().std(),
            'ERC/alpha_raw_abs': alpha_raw.detach().abs().mean(),
            'ERC/alpha_train_abs': alpha_train.detach().abs().mean(),
            'ERC/alpha_final_abs': alpha_final.detach().abs().mean(),
            'ERC/alpha_saturation_rate': (
                alpha_raw.detach().abs() >= saturation_threshold
            ).float().mean(),
            'ERC/alpha_positive_rate': (alpha_final.detach() > 0).float().mean(),
            'ERC/alpha_negative_rate': (alpha_final.detach() < 0).float().mean(),
            'ERC/gate_mean': intervention_gate.detach().mean(),
            'ERC/alpha_positive_region_mean': alpha_positive_region_mean,
            'ERC/alpha_negative_peak_mean': alpha_negative_peak_mean,
            'ERC/alpha_pos_neg_difference': (
                alpha_positive_region_mean - alpha_negative_peak_mean
            ),
            'ERC/regional_global_top1_correct_rate': (
                global_correct.float().mean()
            ),
            'ERC/regional_correctable_rate': (
                regional_correctable.float().mean()
            ),
            'ERC/regional_reachable_margin_mean': (
                regional_reachable_margin.mean()
            ),
            'ERC/regional_effective_positive': masked_mean(
                actual_regional_positive, regional_correctable
            ),
            'ERC/regional_effective_negative': masked_mean(
                actual_regional_negative, regional_correctable
            ),
            'ERC/regional_effective_difference': masked_mean(
                actual_regional_positive - actual_regional_negative,
                regional_correctable,
            ),
            'ERC/regional_margin_gain': regional_margin_gain_per_sample.mean(),
            'ERC/regional_selection_change_rate': (
                regional_selection_changed.float().mean()
            ),
            'ERC/regional_beneficial_flip_rate': (
                regional_beneficial_flip.float().mean()
            ),
            'ERC/regional_harmful_flip_rate': (
                regional_harmful_flip.float().mean()
            ),
            'ERC/regional_loss_gate_detached': zero.detach() + 1.0,
            'ERC/regional_loss_head_detached': zero.detach() + 1.0,
            'ERC/regional_gate_mean': (
                regional_intervention_gate.detach().mean()
            ),
            'ERC/regional_gate_correctable_mean': (
                regional_gate_correctable
            ),
            'ERC/regional_gate_negative_mean': regional_gate_negative,
            'ERC/regional_gate_separation': (
                regional_gate_correctable - regional_gate_negative
            ),
            'ERC/selection_change_rate': final_index.ne(
                base_index
            ).float().mean(),
            'ERC/lambda_gate': zero.detach() + float(lambda_gate),
            'ERC/hard_preserve_weight_effective': (
                zero.detach() + preserve_weight
            ),
            'ERC/margin_noharm_weight_effective': (
                zero.detach() + noharm_weight
            ),
            'ERC/bounded_reachable_margin_mean': (
                bounded_reachable_margin.mean()
            ),
            'ERC/log_ratio_abs_mean': log_ratio.detach().abs().mean(),
            'ERC/positive_log_ratio_mean': positive_log_ratio,
            'ERC/negative_peak_log_ratio_mean': negative_peak_log_ratio,
            'ERC/train_final_score_max_error': validation_score_error,
            'Head/gt_margin_drift_mean': head_gt_margin_drift_mean,
            'Head/gt_margin_drift_abs': head_gt_margin_drift_abs,
            'ERC/hard_masks_require_grad': zero.detach(),
            'ERC/candidate_iou_isolated': zero.detach() + 1.0,
            'ERC/active_fraction': active_fraction.detach(),
        }
        return weighted_loss, status

    def compute_erc_free_gate_losses(self, pred_dict, gt_bbox):
        """Train v6 free-direction ERC without changing baseline box geometry."""
        active_mask = pred_dict.get('erc_active_mask')
        if active_mask is None:
            active_mask = torch.ones(
                pred_dict['bbox_map'].shape[0], dtype=torch.bool,
                device=pred_dict['bbox_map'].device,
            )
        else:
            active_mask = active_mask.reshape(-1).bool()
        active_fraction = active_mask.float().mean()

        status_names = (
            'Loss/erc_total', 'Loss/erc_reliability', 'Loss/erc_gate',
            'Loss/erc_alpha_sup', 'Loss/erc_decision', 'Loss/erc_harm',
            'Loss/erc_geo', 'ERC/reliability_cls',
            'ERC/reliability_cont', 'ERC/reliability_base',
            'ERC/reliability_cls_std', 'ERC/reliability_cont_std',
            'ERC/reliability_base_std', 'ERC/reliability_cls_target',
            'ERC/reliability_cont_target', 'ERC/reliability_base_target',
            'ERC/reliability_cls_mae', 'ERC/reliability_cont_mae',
            'ERC/reliability_base_mae', 'ERC/intervention_gate',
            'ERC/intervention_gate_target', 'ERC/intervention_gate_std',
            'ERC/intervention_gate_mae', 'ERC/gate_positive_target_rate',
            'ERC/z_alpha_mean', 'ERC/z_alpha_std', 'ERC/alpha_mean',
            'ERC/alpha_abs', 'ERC/alpha_std', 'ERC/alpha_target_abs',
            'ERC/alpha_target_nonzero_rate', 'ERC/positive_alpha_rate',
            'ERC/negative_alpha_rate', 'ERC/zero_near_alpha_rate',
            'ERC/target_positive_alpha_rate',
            'ERC/target_negative_alpha_rate', 'ERC/alpha_saturation_rate',
            'ERC/baseline_selected_iou', 'ERC/calibrated_selected_iou',
            'ERC/candidate_iou_gain', 'ERC/selection_change_rate',
            'ERC/alpha_causes_change_rate', 'ERC/changed_candidate_count',
            'ERC/changed_candidate_iou_gain', 'ERC/good_change_rate',
            'ERC/bad_change_rate', 'ERC/neutral_change_rate',
            'ERC/mean_good_gain', 'ERC/mean_bad_loss',
            'ERC/harmful_decision_rate', 'ERC/mean_alpha_when_changed',
            'ERC/mean_alpha_when_unchanged', 'ERC/score_margin_base',
            'ERC/score_margin_erc', 'ERC/soft_expected_iou',
            'ERC/soft_baseline_expected_iou',
            'ERC/soft_expected_iou_gain', 'ERC/geo_risk',
            'ERC/geo_risk_baseline', 'ERC/scalar_oracle_iou',
            'ERC/scalar_oracle_gain', 'ERC/scalar_correctable_rate',
            'ERC/reachable_oracle_iou', 'ERC/reachable_oracle_gain',
            'ERC/oracle_candidate_iou', 'ERC/oracle_headroom',
            'ERC/active_fraction',
        )
        if not active_mask.any():
            zero = pred_dict['bbox_map'].sum() * 0.0
            status = {name: zero.detach() for name in status_names}
            status['ERC/active_fraction'] = active_fraction.detach()
            return zero, status

        # Candidate IoUs and all hard target choices are label-generation
        # operations.  Detaching bbox_map prevents ERC losses from moving box
        # geometry through their IoU targets.
        with torch.no_grad():
            candidate_boxes = pred_dict['bbox_map'][active_mask].detach()
            candidate_iou = self.candidate_iou(
                candidate_boxes, gt_bbox[active_mask]
            )
            cls_index = pred_dict[
                'erc_windowed_cls_score'
            ][active_mask].argmax(dim=-1, keepdim=True)
            cont_index = pred_dict[
                'erc_windowed_cont_score'
            ][active_mask].argmax(dim=-1, keepdim=True)
            baseline_index = pred_dict[
                'erc_windowed_base_score'
            ][active_mask].argmax(dim=-1, keepdim=True)
            reliability_cls_target = candidate_iou.gather(1, cls_index)
            reliability_cont_target = candidate_iou.gather(1, cont_index)
            reliability_base_target = candidate_iou.gather(
                1, baseline_index
            )

        reliability_cls = pred_dict['erc_reliability_cls'][active_mask]
        reliability_cont = pred_dict['erc_reliability_cont'][active_mask]
        reliability_base = pred_dict['erc_reliability_base'][active_mask]
        intervention_gate = pred_dict['erc_intervention_gate'][active_mask]
        z_alpha = pred_dict['erc_z_alpha'][active_mask]
        alpha = pred_dict['erc_alpha'][active_mask]

        reliability_loss = (
            F.smooth_l1_loss(reliability_cls, reliability_cls_target)
            + F.smooth_l1_loss(reliability_cont, reliability_cont_target)
            + F.smooth_l1_loss(reliability_base, reliability_base_target)
        ) / 3.0

        # Keep live C/T scores for differentiable decision, harm, and geometry
        # losses.  Detached copies below are used only to generate targets.
        base_log_score = pred_dict[
            'erc_windowed_base_score'
        ][active_mask].clamp_min(1e-6).log()
        log_ratio = pred_dict['erc_log_ratio'][active_mask]
        base_log_target = base_log_score.detach()
        log_ratio_target = log_ratio.detach()
        calibrated_logits = base_log_score + alpha * log_ratio

        max_alpha = float(self.cfg.MODEL.HEAD.ERC.MAX_ALPHA)
        target_steps = max(int(self.cfg.TRAIN.ERC_ALPHA_TARGET_STEPS), 3)
        if target_steps % 2 == 0:
            target_steps += 1
        min_gain = float(self.cfg.TRAIN.ERC_ALPHA_TARGET_MIN_GAIN)
        gate_scale = max(float(self.cfg.TRAIN.ERC_GATE_SCALE), 1e-6)

        with torch.no_grad():
            unit_grid = torch.linspace(
                -1.0, 1.0, target_steps,
                device=alpha.device, dtype=alpha.dtype,
            )
            full_alpha_grid = max_alpha * unit_grid[None, :]
            full_grid_logits = (
                base_log_target[:, None, :]
                + full_alpha_grid[:, :, None] * log_ratio_target[:, None, :]
            )
            full_grid_index = full_grid_logits.argmax(dim=-1)
            full_grid_iou = candidate_iou.gather(1, full_grid_index)
            scalar_oracle_iou = full_grid_iou.max(dim=1).values
            full_best_mask = full_grid_iou >= (
                scalar_oracle_iou[:, None] - 1e-7
            )
            full_tie_cost = torch.where(
                full_best_mask,
                full_alpha_grid.abs().expand_as(full_grid_iou),
                torch.full_like(full_grid_iou, float('inf')),
            )
            full_best_grid_index = full_tie_cost.argmin(dim=1)
            scalar_oracle_gain = (
                scalar_oracle_iou - reliability_base_target.squeeze(1)
            ).clamp_min(0.0)
            correctable_mask = scalar_oracle_gain > min_gain
            gate_target = torch.where(
                correctable_mask,
                (scalar_oracle_gain / gate_scale).clamp(0.0, 1.0),
                torch.zeros_like(scalar_oracle_gain),
            ).unsqueeze(1)

            # alpha = gate*A*tanh(z) can only reach +/-gate*A.  The full
            # oracle supervises gate, while alpha uses a second oracle inside
            # the interval reachable at gate_target.
            reachable_limit = gate_target * max_alpha
            reachable_alpha_grid = (
                reachable_limit[:, None, :] * unit_grid[None, :, None]
            ).squeeze(-1)
            reachable_grid_logits = (
                base_log_target[:, None, :]
                + reachable_alpha_grid[:, :, None]
                * log_ratio_target[:, None, :]
            )
            reachable_grid_index = reachable_grid_logits.argmax(dim=-1)
            reachable_grid_iou = candidate_iou.gather(
                1, reachable_grid_index
            )
            reachable_oracle_iou = reachable_grid_iou.max(dim=1).values
            reachable_best_mask = reachable_grid_iou >= (
                reachable_oracle_iou[:, None] - 1e-7
            )
            reachable_tie_cost = torch.where(
                reachable_best_mask,
                reachable_alpha_grid.abs(),
                torch.full_like(reachable_grid_iou, float('inf')),
            )
            reachable_best_grid_index = reachable_tie_cost.argmin(dim=1)
            alpha_target = reachable_alpha_grid.gather(
                1, reachable_best_grid_index[:, None]
            )
            alpha_target = torch.where(
                correctable_mask[:, None],
                alpha_target,
                torch.zeros_like(alpha_target),
            )
            reachable_target_index = reachable_grid_index.gather(
                1, reachable_best_grid_index[:, None]
            )
            reachable_oracle_gain = (
                reachable_oracle_iou
                - reliability_base_target.squeeze(1)
            ).clamp_min(0.0)

        gate_element = F.smooth_l1_loss(
            intervention_gate, gate_target, reduction='none'
        ).squeeze(1)
        negative_gate_mask = ~correctable_mask
        if correctable_mask.any() and negative_gate_mask.any():
            gate_loss = 0.5 * (
                gate_element[correctable_mask].mean()
                + gate_element[negative_gate_mask].mean()
            )
        else:
            gate_loss = gate_element.mean()

        if correctable_mask.any() and max_alpha > 0:
            alpha_sup_loss = F.smooth_l1_loss(
                alpha[correctable_mask] / max_alpha,
                alpha_target[correctable_mask] / max_alpha,
            )
        else:
            alpha_sup_loss = alpha.sum() * 0.0

        decision_loss = alpha.sum() * 0.0
        if correctable_mask.any():
            target_logit = calibrated_logits.gather(
                1, reachable_target_index
            ).squeeze(1)
            baseline_logit = calibrated_logits.gather(
                1, baseline_index
            ).squeeze(1)
            decision_loss = F.softplus(
                baseline_logit - target_logit
                + float(self.cfg.TRAIN.ERC_DECISION_MARGIN)
            )[correctable_mask].mean()

        with torch.no_grad():
            calibrated_index = calibrated_logits.argmax(
                dim=-1, keepdim=True
            )
            calibrated_iou = candidate_iou.gather(1, calibrated_index)
            hard_gain = calibrated_iou - reliability_base_target
            changed_mask = calibrated_index.ne(baseline_index)
            harm_mask = hard_gain.squeeze(1) < -float(
                self.cfg.TRAIN.ERC_HARM_IOU_DEADZONE
            )

        harm_loss = alpha.sum() * 0.0
        if harm_mask.any():
            harmful_logit = calibrated_logits.gather(
                1, calibrated_index
            ).squeeze(1)
            baseline_calibrated_logit = calibrated_logits.gather(
                1, baseline_index
            ).squeeze(1)
            harm_loss = F.softplus(
                harmful_logit - baseline_calibrated_logit
                + float(self.cfg.TRAIN.ERC_HARM_SCORE_MARGIN)
            )[harm_mask].mean()

        # A differentiable conditional geometry risk replaces the non-
        # differentiable hard-box penalty.  Geometry is detached; only the
        # calibrated ranking scores receive gradients.
        geo_topk = int(self.cfg.TRAIN.ERC_GEO_TOPK)
        if 0 < geo_topk < calibrated_logits.shape[-1]:
            topk_index = base_log_target.topk(geo_topk, dim=-1).indices
        else:
            topk_index = torch.arange(
                calibrated_logits.shape[-1], device=alpha.device
            )[None, :].expand(calibrated_logits.shape[0], -1)
        topk_calibrated_logits = calibrated_logits.gather(1, topk_index)
        topk_base_logits = base_log_score.gather(1, topk_index)
        topk_iou = candidate_iou.gather(1, topk_index)
        geo_temperature = max(
            float(self.cfg.TRAIN.ERC_GEO_TEMPERATURE), 1e-6
        )
        calibrated_probability = F.softmax(
            topk_calibrated_logits / geo_temperature, dim=-1
        )
        baseline_probability = F.softmax(
            topk_base_logits / geo_temperature, dim=-1
        )
        soft_expected_iou_per_sample = (
            calibrated_probability * topk_iou
        ).sum(dim=-1)
        soft_baseline_iou_per_sample = (
            baseline_probability * topk_iou
        ).sum(dim=-1)

        gather_box_index = topk_index[..., None].expand(-1, -1, 4)
        topk_boxes = candidate_boxes.gather(1, gather_box_index)
        baseline_boxes = candidate_boxes.gather(
            1, baseline_index[..., None].expand(-1, -1, 4)
        )
        center_shift = (
            topk_boxes[..., :2] - baseline_boxes[..., :2]
        ).square().sum(dim=-1).sqrt() / (2.0 ** 0.5)
        scale_shift = (
            topk_boxes[..., 2:].clamp_min(1e-6).log()
            - baseline_boxes[..., 2:].clamp_min(1e-6).log()
        ).abs().sum(dim=-1)
        geometry_distance = center_shift + scale_shift
        calibrated_geo_risk = (
            calibrated_probability * geometry_distance
        ).sum(dim=-1)
        baseline_geo_risk = (
            baseline_probability * geometry_distance
        ).sum(dim=-1)
        risk_weight = (
            (1.0 - gate_target.squeeze(1))
            * reliability_base_target.squeeze(1)
        )
        geo_loss = (
            risk_weight
            * F.relu(calibrated_geo_risk - baseline_geo_risk.detach())
        ).mean()

        weighted_loss = (
            self.loss_weight['erc_reliability'] * reliability_loss
            + self.loss_weight['erc_gate'] * gate_loss
            + self.loss_weight['erc_alpha_sup'] * alpha_sup_loss
            + self.loss_weight['erc_decision'] * decision_loss
            + self.loss_weight['erc_harm'] * harm_loss
            + self.loss_weight['erc_geo'] * geo_loss
        )

        with torch.no_grad():
            changed_count = changed_mask.sum()
            changed_gain = hard_gain[changed_mask]
            changed_gain_mean = (
                changed_gain.mean()
                if changed_gain.numel() else hard_gain.new_zeros(())
            )
            good_mask = changed_mask & hard_gain.gt(1e-7)
            bad_mask = changed_mask & hard_gain.lt(-1e-7)
            neutral_mask = changed_mask & ~(good_mask | bad_mask)
            changed_denominator = changed_count.clamp_min(1).float()
            mean_good_gain = (
                hard_gain[good_mask].mean()
                if good_mask.any() else hard_gain.new_zeros(())
            )
            mean_bad_loss = (
                -hard_gain[bad_mask].mean()
                if bad_mask.any() else hard_gain.new_zeros(())
            )
            alpha_detached = alpha.detach()
            zero_threshold = max(1e-4, max_alpha * 1e-3)
            positive_alpha_mask = alpha_detached.squeeze(1) > zero_threshold
            negative_alpha_mask = alpha_detached.squeeze(1) < -zero_threshold
            zero_alpha_mask = ~(positive_alpha_mask | negative_alpha_mask)
            changed_flat = changed_mask.squeeze(1)
            mean_alpha_when_changed = (
                alpha_detached[changed_flat].abs().mean()
                if changed_flat.any() else alpha_detached.new_zeros(())
            )
            unchanged_flat = ~changed_flat
            mean_alpha_when_unchanged = (
                alpha_detached[unchanged_flat].abs().mean()
                if unchanged_flat.any() else alpha_detached.new_zeros(())
            )
            target_positive_rate = (
                (alpha_target[correctable_mask] > 0).float().mean()
                if correctable_mask.any() else alpha.new_zeros(())
            )
            target_negative_rate = (
                (alpha_target[correctable_mask] < 0).float().mean()
                if correctable_mask.any() else alpha.new_zeros(())
            )
            base_top2 = base_log_target.topk(2, dim=-1).values
            erc_top2 = calibrated_logits.detach().topk(2, dim=-1).values
            score_margin_base = (base_top2[:, 0] - base_top2[:, 1]).mean()
            score_margin_erc = (erc_top2[:, 0] - erc_top2[:, 1]).mean()
            saturation_ratio = float(
                self.cfg.TRAIN.ERC_ALPHA_SATURATION_RATIO
            )
            alpha_saturation_rate = (
                alpha_detached.abs() >= saturation_ratio * max_alpha
            ).float().mean() if max_alpha > 0 else alpha.new_zeros(())
            oracle_candidate_iou = candidate_iou.max(dim=-1).values

        status = {
            'Loss/erc_total': weighted_loss.detach(),
            'Loss/erc_reliability': reliability_loss.detach(),
            'Loss/erc_gate': gate_loss.detach(),
            'Loss/erc_alpha_sup': alpha_sup_loss.detach(),
            'Loss/erc_decision': decision_loss.detach(),
            'Loss/erc_harm': harm_loss.detach(),
            'Loss/erc_geo': geo_loss.detach(),
            'ERC/reliability_cls': reliability_cls.detach().mean(),
            'ERC/reliability_cont': reliability_cont.detach().mean(),
            'ERC/reliability_base': reliability_base.detach().mean(),
            'ERC/reliability_cls_std': reliability_cls.detach().std(unbiased=False),
            'ERC/reliability_cont_std': reliability_cont.detach().std(unbiased=False),
            'ERC/reliability_base_std': reliability_base.detach().std(unbiased=False),
            'ERC/reliability_cls_target': reliability_cls_target.mean(),
            'ERC/reliability_cont_target': reliability_cont_target.mean(),
            'ERC/reliability_base_target': reliability_base_target.mean(),
            'ERC/reliability_cls_mae': (reliability_cls.detach() - reliability_cls_target).abs().mean(),
            'ERC/reliability_cont_mae': (reliability_cont.detach() - reliability_cont_target).abs().mean(),
            'ERC/reliability_base_mae': (reliability_base.detach() - reliability_base_target).abs().mean(),
            'ERC/intervention_gate': intervention_gate.detach().mean(),
            'ERC/intervention_gate_target': gate_target.mean(),
            'ERC/intervention_gate_std': intervention_gate.detach().std(unbiased=False),
            'ERC/intervention_gate_mae': (intervention_gate.detach() - gate_target).abs().mean(),
            'ERC/gate_positive_target_rate': correctable_mask.float().mean(),
            'ERC/z_alpha_mean': z_alpha.detach().mean(),
            'ERC/z_alpha_std': z_alpha.detach().std(unbiased=False),
            'ERC/alpha_mean': alpha_detached.mean(),
            'ERC/alpha_abs': alpha_detached.abs().mean(),
            'ERC/alpha_std': alpha_detached.std(unbiased=False),
            'ERC/alpha_target_abs': alpha_target.abs().mean(),
            'ERC/alpha_target_nonzero_rate': correctable_mask.float().mean(),
            'ERC/positive_alpha_rate': positive_alpha_mask.float().mean(),
            'ERC/negative_alpha_rate': negative_alpha_mask.float().mean(),
            'ERC/zero_near_alpha_rate': zero_alpha_mask.float().mean(),
            'ERC/target_positive_alpha_rate': target_positive_rate,
            'ERC/target_negative_alpha_rate': target_negative_rate,
            'ERC/alpha_saturation_rate': alpha_saturation_rate,
            'ERC/baseline_selected_iou': reliability_base_target.mean(),
            'ERC/calibrated_selected_iou': calibrated_iou.mean(),
            'ERC/candidate_iou_gain': hard_gain.mean(),
            'ERC/selection_change_rate': changed_mask.float().mean(),
            'ERC/alpha_causes_change_rate': changed_mask.float().mean(),
            'ERC/changed_candidate_count': changed_count.float(),
            'ERC/changed_candidate_iou_gain': changed_gain_mean,
            'ERC/good_change_rate': good_mask.sum().float() / changed_denominator,
            'ERC/bad_change_rate': bad_mask.sum().float() / changed_denominator,
            'ERC/neutral_change_rate': neutral_mask.sum().float() / changed_denominator,
            'ERC/mean_good_gain': mean_good_gain,
            'ERC/mean_bad_loss': mean_bad_loss,
            'ERC/harmful_decision_rate': harm_mask.float().mean(),
            'ERC/mean_alpha_when_changed': mean_alpha_when_changed,
            'ERC/mean_alpha_when_unchanged': mean_alpha_when_unchanged,
            'ERC/score_margin_base': score_margin_base,
            'ERC/score_margin_erc': score_margin_erc,
            'ERC/soft_expected_iou': soft_expected_iou_per_sample.detach().mean(),
            'ERC/soft_baseline_expected_iou': soft_baseline_iou_per_sample.detach().mean(),
            'ERC/soft_expected_iou_gain': (
                soft_expected_iou_per_sample - soft_baseline_iou_per_sample
            ).detach().mean(),
            'ERC/geo_risk': calibrated_geo_risk.detach().mean(),
            'ERC/geo_risk_baseline': baseline_geo_risk.detach().mean(),
            'ERC/scalar_oracle_iou': scalar_oracle_iou.mean(),
            'ERC/scalar_oracle_gain': scalar_oracle_gain.mean(),
            'ERC/scalar_correctable_rate': correctable_mask.float().mean(),
            'ERC/reachable_oracle_iou': reachable_oracle_iou.mean(),
            'ERC/reachable_oracle_gain': reachable_oracle_gain.mean(),
            'ERC/oracle_candidate_iou': oracle_candidate_iou.mean(),
            'ERC/oracle_headroom': (
                oracle_candidate_iou - reliability_base_target.squeeze(1)
            ).mean(),
            'ERC/active_fraction': active_fraction.detach(),
        }
        return weighted_loss, status

    def compute_erc_branch_gate_losses(self, pred_dict, gt_bbox):
        """Train frame-level branch reliability and correctability calibration."""
        active_mask = pred_dict.get('erc_active_mask')
        if active_mask is None:
            active_mask = torch.ones(
                pred_dict['bbox_map'].shape[0], dtype=torch.bool,
                device=pred_dict['bbox_map'].device,
            )
        else:
            active_mask = active_mask.reshape(-1).bool()
        active_fraction = active_mask.float().mean()

        status_names = (
            'Loss/erc_total', 'Loss/erc_reliability', 'Loss/erc_gate',
            'Loss/erc_alpha_sup', 'Loss/erc_decision', 'Loss/erc_noharm',
            'Loss/erc_delta', 'ERC/reliability_cls',
            'ERC/reliability_cont', 'ERC/reliability_base',
            'ERC/reliability_cls_std', 'ERC/reliability_cont_std',
            'ERC/reliability_base_std',
            'ERC/reliability_cls_target', 'ERC/reliability_cont_target',
            'ERC/reliability_base_target', 'ERC/reliability_cls_mae',
            'ERC/reliability_cont_mae', 'ERC/reliability_base_mae',
            'ERC/intervention_gate', 'ERC/intervention_gate_target',
            'ERC/intervention_gate_std',
            'ERC/intervention_gate_mae', 'ERC/intervention_gate_weight',
            'ERC/delta_mean', 'ERC/delta_abs', 'ERC/delta_std',
            'ERC/alpha_mean', 'ERC/alpha_abs', 'ERC/alpha_std',
            'ERC/alpha_target_abs', 'ERC/alpha_target_nonzero_rate',
            'ERC/alpha_supervision_weight',
            'ERC/alpha_saturation_rate', 'ERC/baseline_selected_iou',
            'ERC/calibrated_selected_iou', 'ERC/candidate_iou_gain',
            'ERC/selection_change_rate', 'ERC/changed_candidate_count',
            'ERC/changed_candidate_iou_gain', 'ERC/good_change_rate',
            'ERC/bad_change_rate', 'ERC/neutral_change_rate',
            'ERC/mean_good_gain', 'ERC/mean_bad_loss',
            'ERC/soft_expected_iou', 'ERC/soft_baseline_expected_iou',
            'ERC/soft_expected_iou_gain', 'ERC/scalar_oracle_iou',
            'ERC/scalar_oracle_gain', 'ERC/scalar_correctable_rate',
            'ERC/oracle_candidate_iou', 'ERC/oracle_headroom',
            'ERC/active_fraction',
        )
        if not active_mask.any():
            zero = pred_dict['bbox_map'].sum() * 0.0
            status = {name: zero.detach() for name in status_names}
            status['ERC/active_fraction'] = active_fraction.detach()
            return zero, status

        # IoU supervision is deliberately detached from the box geometry.  ERC
        # may learn to rank existing candidates, but cannot move their boxes.
        with torch.no_grad():
            candidate_iou = self.candidate_iou(
                pred_dict['bbox_map'][active_mask].detach(),
                gt_bbox[active_mask],
            )
            cls_index = pred_dict['erc_windowed_cls_score'][active_mask].argmax(
                dim=-1, keepdim=True
            )
            cont_index = pred_dict['erc_windowed_cont_score'][active_mask].argmax(
                dim=-1, keepdim=True
            )
            baseline_index = pred_dict['erc_windowed_base_score'][active_mask].argmax(
                dim=-1, keepdim=True
            )
            reliability_cls_target = candidate_iou.gather(1, cls_index)
            reliability_cont_target = candidate_iou.gather(1, cont_index)
            reliability_base_target = candidate_iou.gather(1, baseline_index)

        reliability_cls = pred_dict['erc_reliability_cls'][active_mask]
        reliability_cont = pred_dict['erc_reliability_cont'][active_mask]
        reliability_base = pred_dict['erc_reliability_base'][active_mask]
        intervention_gate = pred_dict['erc_intervention_gate'][active_mask]
        delta = pred_dict['erc_delta'][active_mask]
        alpha = pred_dict['erc_alpha'][active_mask]

        reliability_loss = (
            F.smooth_l1_loss(reliability_cls, reliability_cls_target)
            + F.smooth_l1_loss(reliability_cont, reliability_cont_target)
            + F.smooth_l1_loss(reliability_base, reliability_base_target)
        ) / 3.0

        base_log_score = pred_dict['erc_windowed_base_score'][active_mask].detach().clamp_min(1e-6).log()
        log_ratio = pred_dict['erc_log_ratio'][active_mask].detach()
        calibrated_logits = base_log_score + alpha * log_ratio
        max_alpha = float(self.cfg.MODEL.HEAD.ERC.MAX_ALPHA)
        direction_scale = float(self.cfg.MODEL.HEAD.ERC.DIRECTION_SCALE)
        delta_scale = float(self.cfg.MODEL.HEAD.ERC.DELTA_SCALE)
        target_steps = max(int(self.cfg.TRAIN.ERC_ALPHA_TARGET_STEPS), 3)

        # Search the exact scalar interval reachable by v5 when g spans [0, 1]
        # and delta spans its configured range.  Including alpha=0 guarantees
        # that the correctability oracle can always recover the baseline.
        with torch.no_grad():
            unit_grid = torch.linspace(
                0.0, 1.0, target_steps,
                device=alpha.device, dtype=alpha.dtype,
            )
            target_direction = direction_scale * (
                reliability_cls_target - reliability_cont_target
            )
            raw_alpha_min = max_alpha * torch.tanh(
                target_direction - delta_scale
            )
            raw_alpha_max = max_alpha * torch.tanh(
                target_direction + delta_scale
            )
            zero_alpha = torch.zeros_like(raw_alpha_min)
            scalar_alpha_min = torch.minimum(raw_alpha_min, zero_alpha)
            scalar_alpha_max = torch.maximum(raw_alpha_max, zero_alpha)
            scalar_alpha_grid = (
                scalar_alpha_min[:, None, :]
                + unit_grid[None, :, None]
                * (scalar_alpha_max - scalar_alpha_min)[:, None, :]
            ).squeeze(-1)
            scalar_grid_logits = (
                base_log_score[:, None, :]
                + scalar_alpha_grid[:, :, None] * log_ratio[:, None, :]
            )
            scalar_grid_index = scalar_grid_logits.argmax(dim=-1)
            scalar_grid_iou = candidate_iou.gather(1, scalar_grid_index)
            scalar_oracle_iou = scalar_grid_iou.max(dim=1).values
            scalar_oracle_gain = (
                scalar_oracle_iou - reliability_base_target.squeeze(1)
            ).clamp_min(0.0)

            gate_scale = max(float(self.cfg.TRAIN.ERC_GATE_SCALE), 1e-6)
            gate_target = (
                scalar_oracle_gain / (scalar_oracle_gain + gate_scale)
            ).unsqueeze(1)

            # Once g has its continuous target, alpha supervision is searched
            # only inside the interval the same formula can reach with that g.
            reachable_alpha_grid = (
                gate_target[:, None, :]
                * (
                    raw_alpha_min[:, None, :]
                    + unit_grid[None, :, None]
                    * (raw_alpha_max - raw_alpha_min)[:, None, :]
                )
            ).squeeze(-1)
            reachable_logits = (
                base_log_score[:, None, :]
                + reachable_alpha_grid[:, :, None] * log_ratio[:, None, :]
            )
            reachable_index = reachable_logits.argmax(dim=-1)
            reachable_iou = candidate_iou.gather(1, reachable_index)
            reachable_best_iou = reachable_iou.max(dim=1).values
            best_mask = reachable_iou >= (reachable_best_iou[:, None] - 1e-7)
            tie_cost = torch.where(
                best_mask,
                reachable_alpha_grid.abs(),
                torch.full_like(reachable_iou, float('inf')),
            )
            best_grid_index = tie_cost.argmin(dim=1)
            alpha_target = reachable_alpha_grid.gather(
                1, best_grid_index[:, None]
            )
            reachable_target_index = reachable_index.gather(
                1, best_grid_index[:, None]
            )
            alpha_target_nonzero = scalar_oracle_gain > 1e-7

        gate_element = F.smooth_l1_loss(
            intervention_gate, gate_target, reduction='none'
        )
        gate_pos_weight = float(self.cfg.TRAIN.ERC_GATE_POS_WEIGHT)
        gate_weight = 1.0 + (gate_pos_weight - 1.0) * gate_target.detach()
        gate_loss = (
            gate_element * gate_weight
        ).sum() / gate_weight.sum().clamp_min(1e-6)
        alpha_sup_element = F.smooth_l1_loss(
            alpha, alpha_target, reduction='none'
        )
        alpha_pos_weight = float(self.cfg.TRAIN.ERC_ALPHA_TARGET_POS_WEIGHT)
        alpha_sup_weight = 1.0 + (
            alpha_pos_weight - 1.0
        ) * alpha_target_nonzero[:, None].float()
        alpha_sup_loss = (
            alpha_sup_element * alpha_sup_weight
        ).sum() / alpha_sup_weight.sum().clamp_min(1e-6)

        decision_loss = alpha.new_zeros(())
        if alpha_target_nonzero.any():
            target_logit = calibrated_logits.gather(
                1, reachable_target_index
            ).squeeze(1)
            baseline_logit = calibrated_logits.gather(
                1, baseline_index
            ).squeeze(1)
            decision_loss = F.softplus(
                baseline_logit - target_logit
                + float(self.cfg.TRAIN.ERC_DECISION_MARGIN)
            )[alpha_target_nonzero].mean()

        # Differentiable no-harm training uses the same candidates and the same
        # calibrated logits as inference, but replaces hard argmax by Top-K soft
        # expectation.  Negative per-sample gains cannot cancel positive ones.
        rank_topk = int(self.cfg.TRAIN.ERC_RANK_TOPK)
        if 0 < rank_topk < calibrated_logits.shape[-1]:
            topk_index = base_log_score.topk(rank_topk, dim=-1).indices
            soft_logits = calibrated_logits.gather(1, topk_index)
            soft_base_logits = base_log_score.gather(1, topk_index)
            soft_iou = candidate_iou.gather(1, topk_index)
        else:
            soft_logits = calibrated_logits
            soft_base_logits = base_log_score
            soft_iou = candidate_iou
        temperature = max(float(self.cfg.TRAIN.ERC_RANK_TEMPERATURE), 1e-6)
        calibrated_probability = F.softmax(soft_logits / temperature, dim=-1)
        baseline_probability = F.softmax(soft_base_logits / temperature, dim=-1)
        soft_expected_iou_per_sample = (
            calibrated_probability * soft_iou
        ).sum(dim=-1)
        soft_baseline_iou_per_sample = (
            baseline_probability * soft_iou
        ).sum(dim=-1)
        soft_gain_per_sample = (
            soft_expected_iou_per_sample - soft_baseline_iou_per_sample
        )
        noharm_loss = F.relu(-soft_gain_per_sample).mean()
        delta_loss = delta.square().mean()

        weighted_loss = (
            self.loss_weight['erc_reliability'] * reliability_loss
            + self.loss_weight['erc_gate'] * gate_loss
            + self.loss_weight['erc_alpha_sup'] * alpha_sup_loss
            + self.loss_weight['erc_decision'] * decision_loss
            + self.loss_weight['erc_noharm'] * noharm_loss
            + self.loss_weight['erc_delta'] * delta_loss
        )

        with torch.no_grad():
            calibrated_index = calibrated_logits.argmax(dim=-1, keepdim=True)
            calibrated_iou = candidate_iou.gather(1, calibrated_index)
            hard_gain = calibrated_iou - reliability_base_target
            changed_mask = calibrated_index.ne(baseline_index)
            changed_count = changed_mask.sum()
            changed_gain = hard_gain[changed_mask]
            changed_gain_mean = (
                changed_gain.mean() if changed_gain.numel() else hard_gain.new_zeros(())
            )
            good_mask = changed_mask & hard_gain.gt(1e-7)
            bad_mask = changed_mask & hard_gain.lt(-1e-7)
            neutral_mask = changed_mask & ~(good_mask | bad_mask)
            denominator = changed_count.clamp_min(1).float()
            mean_good_gain = (
                hard_gain[good_mask].mean() if good_mask.any()
                else hard_gain.new_zeros(())
            )
            mean_bad_loss = (
                -hard_gain[bad_mask].mean() if bad_mask.any()
                else hard_gain.new_zeros(())
            )
            oracle_candidate_iou = candidate_iou.max(dim=-1).values
            saturation_ratio = float(self.cfg.TRAIN.ERC_ALPHA_SATURATION_RATIO)
            alpha_saturation_rate = (
                alpha.abs() >= saturation_ratio * max_alpha
            ).float().mean() if max_alpha > 0 else alpha.new_zeros(())

        status = {
            'Loss/erc_total': weighted_loss.detach(),
            'Loss/erc_reliability': reliability_loss.detach(),
            'Loss/erc_gate': gate_loss.detach(),
            'Loss/erc_alpha_sup': alpha_sup_loss.detach(),
            'Loss/erc_decision': decision_loss.detach(),
            'Loss/erc_noharm': noharm_loss.detach(),
            'Loss/erc_delta': delta_loss.detach(),
            'ERC/reliability_cls': reliability_cls.detach().mean(),
            'ERC/reliability_cont': reliability_cont.detach().mean(),
            'ERC/reliability_base': reliability_base.detach().mean(),
            'ERC/reliability_cls_std': reliability_cls.detach().std(unbiased=False),
            'ERC/reliability_cont_std': reliability_cont.detach().std(unbiased=False),
            'ERC/reliability_base_std': reliability_base.detach().std(unbiased=False),
            'ERC/reliability_cls_target': reliability_cls_target.mean(),
            'ERC/reliability_cont_target': reliability_cont_target.mean(),
            'ERC/reliability_base_target': reliability_base_target.mean(),
            'ERC/reliability_cls_mae': (reliability_cls.detach() - reliability_cls_target).abs().mean(),
            'ERC/reliability_cont_mae': (reliability_cont.detach() - reliability_cont_target).abs().mean(),
            'ERC/reliability_base_mae': (reliability_base.detach() - reliability_base_target).abs().mean(),
            'ERC/intervention_gate': intervention_gate.detach().mean(),
            'ERC/intervention_gate_target': gate_target.mean(),
            'ERC/intervention_gate_std': intervention_gate.detach().std(unbiased=False),
            'ERC/intervention_gate_mae': (intervention_gate.detach() - gate_target).abs().mean(),
            'ERC/intervention_gate_weight': gate_weight.mean(),
            'ERC/delta_mean': delta.detach().mean(),
            'ERC/delta_abs': delta.detach().abs().mean(),
            'ERC/delta_std': delta.detach().std(unbiased=False),
            'ERC/alpha_mean': alpha.detach().mean(),
            'ERC/alpha_abs': alpha.detach().abs().mean(),
            'ERC/alpha_std': alpha.detach().std(unbiased=False),
            'ERC/alpha_target_abs': alpha_target.abs().mean(),
            'ERC/alpha_target_nonzero_rate': alpha_target_nonzero.float().mean(),
            'ERC/alpha_supervision_weight': alpha_sup_weight.mean(),
            'ERC/alpha_saturation_rate': alpha_saturation_rate,
            'ERC/baseline_selected_iou': reliability_base_target.mean(),
            'ERC/calibrated_selected_iou': calibrated_iou.mean(),
            'ERC/candidate_iou_gain': hard_gain.mean(),
            'ERC/selection_change_rate': changed_mask.float().mean(),
            'ERC/changed_candidate_count': changed_count.float(),
            'ERC/changed_candidate_iou_gain': changed_gain_mean,
            'ERC/good_change_rate': good_mask.sum().float() / denominator,
            'ERC/bad_change_rate': bad_mask.sum().float() / denominator,
            'ERC/neutral_change_rate': neutral_mask.sum().float() / denominator,
            'ERC/mean_good_gain': mean_good_gain,
            'ERC/mean_bad_loss': mean_bad_loss,
            'ERC/soft_expected_iou': soft_expected_iou_per_sample.detach().mean(),
            'ERC/soft_baseline_expected_iou': soft_baseline_iou_per_sample.mean(),
            'ERC/soft_expected_iou_gain': soft_gain_per_sample.detach().mean(),
            'ERC/scalar_oracle_iou': scalar_oracle_iou.mean(),
            'ERC/scalar_oracle_gain': scalar_oracle_gain.mean(),
            'ERC/scalar_correctable_rate': scalar_oracle_gain.gt(1e-7).float().mean(),
            'ERC/oracle_candidate_iou': oracle_candidate_iou.mean(),
            'ERC/oracle_headroom': (oracle_candidate_iou - reliability_base_target.squeeze(1)).mean(),
            'ERC/active_fraction': active_fraction.detach(),
        }
        return weighted_loss, status

    def compute_erc_legacy_losses(self, pred_dict, gt_bbox):
        """Train ERC while keeping its supervision gradients out of the baseline head."""
        active_mask = pred_dict.get('erc_active_mask')
        if active_mask is None:
            active_mask = torch.ones(
                pred_dict['bbox_map'].shape[0],
                dtype=torch.bool,
                device=pred_dict['bbox_map'].device,
            )
        else:
            active_mask = active_mask.reshape(-1).bool()
        active_fraction = active_mask.float().mean()
        if not active_mask.any():
            zero = pred_dict['bbox_map'].sum() * 0.0
            status = {
                'Loss/erc_total': zero.detach(),
                'Loss/erc_reliability': zero.detach(),
                'Loss/erc_reliability_rank': zero.detach(),
                'Loss/erc_rank': zero.detach(),
                'Loss/erc_alpha_reg': zero.detach(),
                'Loss/erc_alpha_sup': zero.detach(),
                'Loss/erc_decision': zero.detach(),
                'ERC/reliability': zero.detach(),
                'ERC/reliability_target': zero.detach(),
                'ERC/reliability_mae': zero.detach(),
                'ERC/alpha_mean': zero.detach(),
                'ERC/alpha_abs': zero.detach(),
                'ERC/alpha_std': zero.detach(),
                'ERC/raw_alpha_mean': zero.detach(),
                'ERC/raw_alpha_abs': zero.detach(),
                'ERC/reliability_gate': zero.detach(),
                'ERC/reliability_weight': zero.detach(),
                'ERC/low_quality_fraction': zero.detach(),
                'ERC/alpha_positive_fraction': zero.detach(),
                'ERC/alpha_saturation_rate': zero.detach(),
                'ERC/reliability_std': zero.detach(),
                'ERC/baseline_selected_iou': zero.detach(),
                'ERC/calibrated_selected_iou': zero.detach(),
                'ERC/candidate_iou_gain': zero.detach(),
                'ERC/changed_candidate_count': zero.detach(),
                'ERC/changed_candidate_iou_gain': zero.detach(),
                'ERC/selection_change_rate': zero.detach(),
                'ERC/soft_expected_iou': zero.detach(),
                'ERC/soft_expected_iou_gain': zero.detach(),
                'ERC/oracle_candidate_iou': zero.detach(),
                'ERC/oracle_headroom': zero.detach(),
                'ERC/scalar_oracle_iou': zero.detach(),
                'ERC/scalar_oracle_gain': zero.detach(),
                'ERC/alpha_target_abs': zero.detach(),
                'ERC/alpha_target_nonzero_rate': zero.detach(),
                'ERC/active_fraction': active_fraction.detach(),
            }
            return zero, status

        with torch.no_grad():
            candidate_iou = self.candidate_iou(
                pred_dict['bbox_map'][active_mask], gt_bbox[active_mask]
            )
            baseline_index = pred_dict['erc_windowed_base_score'][active_mask].argmax(
                dim=-1, keepdim=True
            )
            reliability_target = candidate_iou.gather(1, baseline_index)
            best_candidate_index = candidate_iou.argmax(dim=-1)

        reliability = pred_dict['erc_reliability'][active_mask]
        reliability_loss_type = self.cfg.TRAIN.ERC_RELIABILITY_LOSS.lower()
        reliability_weight = reliability_target.new_ones(reliability_target.shape)
        if reliability_loss_type == 'smooth_l1':
            reliability_loss = F.smooth_l1_loss(reliability, reliability_target)
        elif reliability_loss_type == 'weighted_bce':
            low_iou_weight = float(
                self.cfg.TRAIN.ERC_RELIABILITY_LOW_IOU_WEIGHT
            )
            reliability_weight = 1.0 + low_iou_weight * (1.0 - reliability_target)
            reliability_bce = F.binary_cross_entropy(
                reliability.clamp(min=1e-6, max=1.0 - 1e-6),
                reliability_target,
                reduction='none',
            )
            reliability_loss = (
                reliability_bce * reliability_weight
            ).sum() / reliability_weight.sum().clamp_min(1e-6)
        else:
            raise ValueError(
                "Unsupported ERC reliability loss: %s"
                % self.cfg.TRAIN.ERC_RELIABILITY_LOSS
            )

        reliability_rank_loss = reliability.new_zeros(())
        reliability_rank_margin = float(
            self.cfg.TRAIN.ERC_RELIABILITY_RANK_MARGIN
        )
        target_difference = reliability_target - reliability_target.transpose(0, 1)
        prediction_difference = reliability - reliability.transpose(0, 1)
        reliability_pair_mask = torch.triu(
            target_difference.abs() >= reliability_rank_margin,
            diagonal=1,
        )
        if reliability_pair_mask.any():
            pair_direction = target_difference.sign()
            pair_loss = F.softplus(-pair_direction * prediction_difference)
            reliability_rank_loss = pair_loss[reliability_pair_mask].mean()

        base_log_score = (
            pred_dict['erc_windowed_base_score'][active_mask]
            .detach().clamp_min(1e-6).log()
        )
        log_ratio = pred_dict['erc_log_ratio'][active_mask].detach()
        alpha = pred_dict['erc_alpha'][active_mask]
        calibrated_logits = base_log_score + alpha * log_ratio
        rank_temperature = float(self.cfg.TRAIN.ERC_RANK_TEMPERATURE)
        if rank_temperature <= 0:
            raise ValueError("TRAIN.ERC_RANK_TEMPERATURE must be positive")
        rank_loss_type = self.cfg.TRAIN.ERC_RANK_LOSS.lower()
        soft_expected_iou = reliability.new_zeros(())
        soft_baseline_expected_iou = reliability.new_zeros(())
        if rank_loss_type == 'hard_ce':
            rank_loss = F.cross_entropy(
                calibrated_logits / rank_temperature,
                best_candidate_index,
            )
        elif rank_loss_type == 'soft_iou':
            rank_topk = int(self.cfg.TRAIN.ERC_RANK_TOPK)
            if 0 < rank_topk < calibrated_logits.shape[-1]:
                rank_indices = base_log_score.topk(rank_topk, dim=-1).indices
                rank_logits = calibrated_logits.gather(1, rank_indices)
                rank_base_logits = base_log_score.gather(1, rank_indices)
                rank_candidate_iou = candidate_iou.gather(1, rank_indices)
            else:
                rank_logits = calibrated_logits
                rank_base_logits = base_log_score
                rank_candidate_iou = candidate_iou
            rank_probability = F.softmax(rank_logits / rank_temperature, dim=-1)
            baseline_probability = F.softmax(
                rank_base_logits / rank_temperature,
                dim=-1,
            )
            soft_expected_iou = (
                rank_probability * rank_candidate_iou
            ).sum(dim=-1).mean()
            soft_baseline_expected_iou = (
                baseline_probability * rank_candidate_iou
            ).sum(dim=-1).mean()
            rank_loss = 1.0 - soft_expected_iou
        else:
            raise ValueError(
                "Unsupported ERC rank loss: %s"
                % self.cfg.TRAIN.ERC_RANK_LOSS
            )

        raw_alpha = pred_dict.get(
            'erc_raw_alpha', pred_dict['erc_alpha']
        )[active_mask]
        alpha_for_regularization = raw_alpha
        max_alpha = float(self.cfg.MODEL.HEAD.ERC.MAX_ALPHA)
        if self.cfg.TRAIN.ERC_NORMALIZED_ALPHA_REG and max_alpha > 0:
            alpha_for_regularization = alpha_for_regularization / max_alpha
        alpha_reg_loss = alpha_for_regularization.square().mean()

        alpha_sup_loss = reliability.new_zeros(())
        decision_loss = reliability.new_zeros(())
        scalar_oracle_iou = reliability_target.squeeze(1)
        scalar_oracle_gain = scalar_oracle_iou - reliability_target.squeeze(1)
        alpha_target = alpha.detach().new_zeros(alpha.shape)
        alpha_target_nonzero = active_mask.new_zeros(alpha.shape[0])
        alpha_target_steps = int(self.cfg.TRAIN.ERC_ALPHA_TARGET_STEPS)
        if alpha_target_steps > 1 and max_alpha > 0:
            with torch.no_grad():
                unit_alpha_grid = torch.linspace(
                    -1.0,
                    1.0,
                    alpha_target_steps,
                    device=alpha.device,
                    dtype=alpha.dtype,
                )
                # Alpha is reliability-gated in the forward path.  Build each
                # sample's oracle target inside the range that the predictor can
                # actually reach, instead of supervising it with an impossible
                # value from the ungated [-MAX_ALPHA, MAX_ALPHA] interval.
                if self.cfg.MODEL.HEAD.ERC.RELIABILITY_GATED_ALPHA:
                    effective_alpha_limit = (
                        max_alpha
                        * pred_dict['erc_reliability_gate'][active_mask]
                    )
                else:
                    effective_alpha_limit = alpha.new_full(
                        alpha.shape, max_alpha
                    )
                alpha_grid = (
                    effective_alpha_limit * unit_alpha_grid[None, :]
                )
                grid_logits = (
                    base_log_score[:, None, :]
                    + alpha_grid[:, :, None] * log_ratio[:, None, :]
                )
                grid_candidate_index = grid_logits.argmax(dim=-1)
                grid_iou = candidate_iou.gather(1, grid_candidate_index)
                scalar_oracle_iou = grid_iou.max(dim=1).values
                best_grid_mask = grid_iou >= (
                    scalar_oracle_iou[:, None] - 1e-7
                )
                tie_cost = torch.where(
                    best_grid_mask,
                    alpha_grid.abs(),
                    torch.full_like(grid_iou, float('inf')),
                )
                best_grid_index = tie_cost.argmin(dim=1)
                alpha_target = alpha_grid.gather(
                    1, best_grid_index[:, None]
                )
                scalar_oracle_index = grid_candidate_index.gather(
                    1, best_grid_index[:, None]
                )
                scalar_oracle_gain = (
                    scalar_oracle_iou - reliability_target.squeeze(1)
                )
                alpha_target_nonzero = scalar_oracle_gain > float(
                    self.cfg.TRAIN.ERC_ALPHA_TARGET_MIN_GAIN
                )
                alpha_target = torch.where(
                    alpha_target_nonzero[:, None],
                    alpha_target,
                    torch.zeros_like(alpha_target),
                )

            alpha_sup_element = F.smooth_l1_loss(
                alpha, alpha_target, reduction='none'
            ).squeeze(1)
            alpha_pos_weight = float(
                self.cfg.TRAIN.ERC_ALPHA_TARGET_POS_WEIGHT
            )
            alpha_sup_weight = 1.0 + (
                alpha_pos_weight - 1.0
            ) * alpha_target_nonzero.float()
            alpha_sup_loss = (
                alpha_sup_element * alpha_sup_weight
            ).sum() / alpha_sup_weight.sum().clamp_min(1e-6)

            if alpha_target_nonzero.any():
                target_logit = calibrated_logits.gather(
                    1, scalar_oracle_index
                ).squeeze(1)
                baseline_logit = calibrated_logits.gather(
                    1, baseline_index
                ).squeeze(1)
                decision_margin = float(self.cfg.TRAIN.ERC_DECISION_MARGIN)
                decision_element = F.softplus(
                    baseline_logit - target_logit + decision_margin
                )
                decision_loss = decision_element[alpha_target_nonzero].mean()

        # Paired diagnostics use the exact test-time ranking path: the official
        # Hann-windowed baseline score multiplied by ERC's calibration factor.
        # Since both candidates are evaluated on the same sample and bbox map,
        # their IoU difference isolates ERC's candidate-selection effect.
        with torch.no_grad():
            calibrated_index = calibrated_logits.argmax(dim=-1, keepdim=True)
            calibrated_candidate_iou = candidate_iou.gather(1, calibrated_index)
            candidate_iou_gain = calibrated_candidate_iou - reliability_target
            changed_candidate_mask = calibrated_index != baseline_index
            changed_candidate_count = changed_candidate_mask.sum()
            changed_candidate_iou_gain = (
                candidate_iou_gain
                * changed_candidate_mask.to(candidate_iou_gain.dtype)
            ).sum() / changed_candidate_count.clamp_min(1).to(candidate_iou_gain.dtype)
            selection_change_rate = changed_candidate_mask.float().mean()
            oracle_candidate_iou = candidate_iou.max(dim=-1, keepdim=True).values

            alpha_detached = alpha.detach()
            saturation_ratio = float(
                self.cfg.TRAIN.ERC_ALPHA_SATURATION_RATIO
            )
            if max_alpha > 0:
                alpha_saturation_rate = (
                    alpha_detached.abs() >= saturation_ratio * max_alpha
                ).float().mean()
            else:
                alpha_saturation_rate = alpha.new_zeros(())
            reliability_mae = (
                reliability.detach() - reliability_target
            ).abs().mean()

        weighted_loss = (
            self.loss_weight['erc_reliability'] * reliability_loss
            + self.loss_weight['erc_reliability_rank'] * reliability_rank_loss
            + self.loss_weight['erc_rank'] * rank_loss
            + self.loss_weight['erc_alpha_reg'] * alpha_reg_loss
            + self.loss_weight.get('erc_alpha_sup', 0.0) * alpha_sup_loss
            + self.loss_weight.get('erc_decision', 0.0) * decision_loss
        )
        status = {
            'Loss/erc_total': weighted_loss,
            'Loss/erc_reliability': reliability_loss,
            'Loss/erc_reliability_rank': reliability_rank_loss,
            'Loss/erc_rank': rank_loss,
            'Loss/erc_alpha_reg': alpha_reg_loss,
            'Loss/erc_alpha_sup': alpha_sup_loss,
            'Loss/erc_decision': decision_loss,
            'ERC/reliability': reliability.detach().mean(),
            'ERC/reliability_target': reliability_target.detach().mean(),
            'ERC/reliability_mae': reliability_mae,
            'ERC/alpha_mean': alpha_detached.mean(),
            'ERC/alpha_abs': alpha_detached.abs().mean(),
            'ERC/alpha_std': alpha_detached.std(unbiased=False),
            'ERC/raw_alpha_mean': raw_alpha.detach().mean(),
            'ERC/raw_alpha_abs': raw_alpha.detach().abs().mean(),
            'ERC/reliability_gate': pred_dict[
                'erc_reliability_gate'
            ][active_mask].detach().mean(),
            'ERC/reliability_weight': reliability_weight.detach().mean(),
            'ERC/low_quality_fraction': (reliability_target < 0.5).float().mean(),
            'ERC/alpha_positive_fraction': (
                alpha_detached > 0
            ).float().mean(),
            'ERC/alpha_saturation_rate': alpha_saturation_rate,
            'ERC/reliability_std': reliability.detach().std(unbiased=False),
            'ERC/baseline_selected_iou': reliability_target.mean(),
            'ERC/calibrated_selected_iou': calibrated_candidate_iou.mean(),
            'ERC/candidate_iou_gain': candidate_iou_gain.mean(),
            # Trainer averages every status value across batches.  Keep the
            # count in floating point so a later non-integer running average
            # cannot be accumulated into an int64 tensor.
            'ERC/changed_candidate_count': changed_candidate_count.float(),
            'ERC/changed_candidate_iou_gain': changed_candidate_iou_gain,
            'ERC/selection_change_rate': selection_change_rate,
            'ERC/soft_expected_iou': soft_expected_iou.detach(),
            'ERC/soft_expected_iou_gain': (
                soft_expected_iou - soft_baseline_expected_iou
            ).detach(),
            'ERC/oracle_candidate_iou': oracle_candidate_iou.mean(),
            'ERC/oracle_headroom': (
                oracle_candidate_iou - reliability_target
            ).mean(),
            'ERC/scalar_oracle_iou': scalar_oracle_iou.mean(),
            'ERC/scalar_oracle_gain': scalar_oracle_gain.mean(),
            'ERC/alpha_target_abs': alpha_target.abs().mean(),
            'ERC/alpha_target_nonzero_rate': alpha_target_nonzero.float().mean(),
            'ERC/active_fraction': active_fraction.detach(),
        }
        return weighted_loss, status

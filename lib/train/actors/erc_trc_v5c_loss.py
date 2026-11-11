"""Deployment-aligned and batch-invariant supervision for TRC-v5c."""

import math

import torch
from torch.nn import functional as F

from lib.train.actors.erc_trc_v4_loss import (
    hard_decision_action_oracle,
    masked_average,
)
from lib.train.actors.erc_trc_v5_loss import (
    balanced_strength_loss,
    permission_losses,
)
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy


def batch_invariant_direction_loss(logits, target, valid, cont_weight):
    """Apply the audited global C:T weight without per-batch renormalization."""
    if not valid.any():
        return logits.sum() * 0.0
    per_frame = F.cross_entropy(logits[valid], target[valid], reduction='none')
    weight = torch.where(
        target[valid].eq(1),
        per_frame.new_full(per_frame.shape, float(cont_weight)),
        per_frame.new_ones(per_frame.shape),
    )
    return (per_frame * weight).mean()


def batch_invariant_direction_margin_loss(logits, target, valid, cont_weight,
                                          margin):
    if not valid.any():
        return logits.sum() * 0.0
    valid_logits, valid_target = logits[valid], target[valid]
    row = torch.arange(valid_logits.shape[0], device=logits.device)
    correct = valid_logits[row, valid_target]
    other = valid_logits[row, 1 - valid_target]
    per_frame = F.relu(float(margin) - (correct - other))
    weight = torch.where(
        valid_target.eq(1),
        per_frame.new_full(per_frame.shape, float(cont_weight)),
        per_frame.new_ones(per_frame.shape),
    )
    return (per_frame * weight).mean()


def permission_boundary_losses(logit, target, valid, threshold,
                               permission_temperature, margin,
                               smooth_temperature):
    """Place positives/negatives on the correct side of the deployed boundary."""
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError('permission threshold must be in (0, 1)')
    permission_temperature = max(float(permission_temperature), 1e-6)
    smooth_temperature = max(float(smooth_temperature), 1e-6)
    boundary = permission_temperature * math.log(
        float(threshold) / (1.0 - float(threshold))
    )
    positive = valid & target.bool()
    negative = valid & ~target.bool()
    positive_loss = masked_average(
        F.softplus(
            (boundary + float(margin) - logit) / smooth_temperature
        ) * smooth_temperature,
        positive,
    )
    negative_loss = masked_average(
        F.softplus(
            (logit - (boundary - float(margin))) / smooth_temperature
        ) * smooth_temperature,
        negative,
    )
    return positive_loss, negative_loss


def compute_trc_v5c_loss_and_metrics(pred, gt_heatmap, gt_bbox, cfg):
    packet = pred['erc_trc_output']
    oracle = hard_decision_action_oracle(
        packet, gt_heatmap, cfg.TRAIN.ERC_TRC_V5C_ORACLE_STEPS,
        cfg.TRAIN.ERC_TRC_V5C_SCORE_TEMPERATURE,
        cfg.TRAIN.ERC_TRC_V5C_MIN_TARGET_GAIN,
    )
    valid = oracle['valid']
    target_action = oracle['target_action']
    corrective = valid & target_action.ne(0)
    preserve = valid & target_action.eq(0)
    permission_target = corrective.to(packet['permission_logit'])
    permission_loss, permission_rank_loss = permission_losses(
        packet['permission_logit'], permission_target, valid,
        cfg.TRAIN.ERC_TRC_V5C_PERMISSION_RANK_MARGIN,
    )
    permission_positive_boundary, permission_negative_boundary = (
        permission_boundary_losses(
            packet['permission_logit'], permission_target, valid,
            packet['permission_threshold'],
            packet['permission_temperature'],
            cfg.TRAIN.ERC_TRC_V5C_PERMISSION_BOUNDARY_MARGIN,
            cfg.TRAIN.ERC_TRC_V5C_PERMISSION_BOUNDARY_TEMPERATURE,
        )
    )

    direction_target = (target_action - 1).clamp(0, 1)
    direction_loss = batch_invariant_direction_loss(
        packet['direction_logits'], direction_target, corrective,
        cfg.TRAIN.ERC_TRC_V5C_DIRECTION_CONT_WEIGHT,
    )
    direction_margin_loss = batch_invariant_direction_margin_loss(
        packet['direction_logits'], direction_target, corrective,
        cfg.TRAIN.ERC_TRC_V5C_DIRECTION_CONT_WEIGHT,
        cfg.TRAIN.ERC_TRC_V5C_DIRECTION_MARGIN,
    )
    strength_target = torch.where(
        corrective, oracle['target_strength'],
        torch.zeros_like(oracle['target_strength']),
    )
    strength_loss = balanced_strength_loss(
        packet['strength'], strength_target, corrective, preserve,
    )

    base_log = packet['log_base'].detach().flatten(1)
    difference = packet['evidence']['d'].detach().flatten(1)
    support = packet['support'].detach().flatten(1)
    model_log = base_log + packet['soft_alpha'].flatten(1) * difference
    batch_index = torch.arange(model_log.shape[0], device=model_log.device)
    target_value = model_log[batch_index, oracle['target_index']]
    base_value = model_log[batch_index, oracle['base_index']]
    margin = float(cfg.TRAIN.ERC_TRC_V5C_DECISION_MARGIN)
    temperature = max(float(cfg.TRAIN.ERC_TRC_V5C_MARGIN_TEMPERATURE), 1e-6)
    decision_loss = masked_average(
        F.softplus((margin - (target_value - base_value)) / temperature)
        * temperature,
        corrective,
    )
    alternatives = model_log.masked_fill(~support, -1e4).clone()
    alternatives[batch_index, oracle['base_index']] = -1e4
    strongest_alternative = alternatives.max(-1).values
    preserve_loss = masked_average(
        F.softplus(
            (margin - (base_value - strongest_alternative)) / temperature
        ) * temperature,
        preserve,
    )
    action_risk = F.relu(
        oracle['action_hard_quality'][:, :1] - oracle['action_hard_quality']
    )
    risk_loss = masked_average(
        (packet['action_probability'] * action_risk).sum(-1), valid
    )

    loss = (
        cfg.TRAIN.ERC_TRC_V5C_PERMISSION_WEIGHT * permission_loss
        + cfg.TRAIN.ERC_TRC_V5C_PERMISSION_RANK_WEIGHT * permission_rank_loss
        + cfg.TRAIN.ERC_TRC_V5C_PERMISSION_POS_BOUNDARY_WEIGHT
        * permission_positive_boundary
        + cfg.TRAIN.ERC_TRC_V5C_PERMISSION_NEG_BOUNDARY_WEIGHT
        * permission_negative_boundary
        + cfg.TRAIN.ERC_TRC_V5C_DIRECTION_WEIGHT * direction_loss
        + cfg.TRAIN.ERC_TRC_V5C_DIRECTION_MARGIN_WEIGHT * direction_margin_loss
        + cfg.TRAIN.ERC_TRC_V5C_STRENGTH_WEIGHT * strength_loss
        + cfg.TRAIN.ERC_TRC_V5C_DECISION_WEIGHT * decision_loss
        + cfg.TRAIN.ERC_TRC_V5C_PRESERVE_WEIGHT * preserve_loss
        + cfg.TRAIN.ERC_TRC_V5C_RISK_WEIGHT * risk_loss
    )
    status = {
        'Loss/erc_trc_v5c_permission': permission_loss.detach(),
        'Loss/erc_trc_v5c_permission_rank': permission_rank_loss.detach(),
        'Loss/erc_trc_v5c_permission_positive_boundary':
            permission_positive_boundary.detach(),
        'Loss/erc_trc_v5c_permission_negative_boundary':
            permission_negative_boundary.detach(),
        'Loss/erc_trc_v5c_direction': direction_loss.detach(),
        'Loss/erc_trc_v5c_direction_margin': direction_margin_loss.detach(),
        'Loss/erc_trc_v5c_strength': strength_loss.detach(),
        'Loss/erc_trc_v5c_decision': decision_loss.detach(),
        'Loss/erc_trc_v5c_preserve': preserve_loss.detach(),
        'Loss/erc_trc_v5c_risk': risk_loss.detach(),
        'Loss/erc_total': loss.detach(),
    }

    with torch.no_grad():
        active = packet['active']
        zero = loss.detach().new_zeros(())

        def mean(value, mask):
            return value[mask].mean() if mask.any() else zero

        def std(value, mask):
            return value[mask].std(unbiased=False) if mask.any() else zero

        base = packet['base'].flatten(1)
        score = packet['score'].flatten(1)
        base_index, full_index = base.argmax(-1), score.argmax(-1)
        changed = base_index.ne(full_index) & active
        relevance = gt_heatmap.detach().flatten(1)
        response_gain = (
            relevance[batch_index, full_index]
            - relevance[batch_index, base_index]
        )
        predicted_direction = packet['hard_direction_index']
        predicted_permission = packet['hard_permission'].bool()
        true_positive = predicted_permission & corrective
        predicted_positive = predicted_permission & valid
        precision = (
            true_positive.sum().float()
            / predicted_positive.sum().clamp_min(1).float()
        )
        cls_target = corrective & direction_target.eq(0)
        cont_target = corrective & direction_target.eq(1)
        active_alpha = active[:, None, None].expand_as(packet['alpha'])
        status.update({
            'ERC/active_fraction': active.float().mean(),
            'ERC/alpha_abs': mean(packet['alpha'].abs(), active_alpha),
            'ERC/alpha_mean': mean(packet['alpha'], active_alpha),
            'ERC/alpha_std': std(packet['alpha'], active_alpha),
            'ERC/permission_threshold': zero.new_tensor(
                float(packet['permission_threshold'])),
            'ERC/permission_mean': mean(packet['permission_probability'], active),
            'ERC/hard_permission_fraction': mean(packet['hard_permission'], active),
            'ERC/permission_precision': precision,
            'ERC/permission_true_positive_rate': mean(
                packet['hard_permission'], corrective),
            'ERC/permission_false_positive_rate': mean(
                packet['hard_permission'], preserve),
            'ERC/permission_positive_mean': mean(
                packet['permission_probability'], corrective),
            'ERC/permission_negative_mean': mean(
                packet['permission_probability'], preserve),
            'ERC/permission_true_positive_count': true_positive.sum().float(),
            'ERC/permission_predicted_positive_count':
                predicted_positive.sum().float(),
            'ERC/permission_corrective_count': corrective.sum().float(),
            'ERC/permission_preserve_count': preserve.sum().float(),
            'ERC/direction_cls_mean': mean(
                packet['direction_probability'][:, 0], active),
            'ERC/direction_cont_mean': mean(
                packet['direction_probability'][:, 1], active),
            'ERC/direction_accuracy': mean(
                predicted_direction.eq(direction_target).float(), corrective),
            'ERC/direction_cls_recall': mean(
                predicted_direction.eq(0).float(), cls_target),
            'ERC/direction_cont_recall': mean(
                predicted_direction.eq(1).float(), cont_target),
            'ERC/direction_cls_correct_count': (
                predicted_direction.eq(0) & cls_target).sum().float(),
            'ERC/direction_cont_correct_count': (
                predicted_direction.eq(1) & cont_target).sum().float(),
            'ERC/direction_cls_target_count': cls_target.sum().float(),
            'ERC/direction_cont_target_count': cont_target.sum().float(),
            'ERC/hard_keep_fraction': mean(packet['hard_action'].eq(0).float(), active),
            'ERC/hard_cls_fraction': mean(packet['hard_action'].eq(1).float(), active),
            'ERC/hard_cont_fraction': mean(packet['hard_action'].eq(2).float(), active),
            'ERC/oracle_keep_fraction': mean(target_action.eq(0).float(), valid),
            'ERC/oracle_cls_fraction': mean(target_action.eq(1).float(), valid),
            'ERC/oracle_cont_fraction': mean(target_action.eq(2).float(), valid),
            'ERC/oracle_response_advantage': mean(oracle['target_gain'], valid),
            'ERC/strength_mean': mean(packet['strength'], active),
            'ERC/strength_corrective_mean': mean(packet['strength'], corrective),
            'ERC/strength_preserve_mean': mean(packet['strength'], preserve),
            'ERC/strength_target_mean': mean(strength_target, corrective),
            'ERC/selection_change_rate': mean(changed.float(), active),
            'ERC/changed_response_target_gain': mean(response_gain, changed),
            'ERC/changed_response_negative_rate': mean(
                response_gain.lt(-1e-6).float(), changed),
        })

        boxes = box_cxcywh_to_xyxy(pred['bbox_map'].detach()).clamp(0, 1)
        gt = box_xywh_to_xyxy(gt_bbox.detach()).clamp(0, 1)[:, None]
        intersection_wh = (
            torch.minimum(boxes[..., 2:], gt[..., 2:])
            - torch.maximum(boxes[..., :2], gt[..., :2])
        ).clamp_min(0)
        intersection = intersection_wh.prod(-1)
        union = (
            (boxes[..., 2:] - boxes[..., :2]).clamp_min(0).prod(-1)
            + (gt[..., 2:] - gt[..., :2]).clamp_min(0).prod(-1)
            - intersection
        )
        candidate_iou = intersection / union.clamp_min(1e-6)
        iou_gain = (
            candidate_iou[batch_index, full_index]
            - candidate_iou[batch_index, base_index]
        )
        valid_box = (
            active & torch.isfinite(gt_bbox).all(-1)
            & (gt_bbox[:, 2:] > 0).all(-1)
        )
        status.update({
            'ERC/candidate_iou_gain': mean(iou_gain, valid_box),
            'ERC/changed_candidate_iou_gain': mean(
                iou_gain, changed & valid_box),
            'ERC/changed_candidate_count': (changed & valid_box).sum().float(),
            'ERC/changed_candidate_negative_rate': mean(
                iou_gain.lt(-1e-6).float(), changed & valid_box),
        })
    return loss, status

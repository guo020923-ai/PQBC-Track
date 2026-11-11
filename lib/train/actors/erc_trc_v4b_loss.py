"""Factorized hard-decision supervision for TRC-v4b."""

import torch
from torch.nn import functional as F

from lib.train.actors.erc_trc_v4_loss import (
    hard_decision_action_oracle,
    masked_average,
)
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy


def balanced_permission_loss(
        logit, target, valid, maximum_positive_weight,
        fixed_positive_weight=0.0):
    positive = (valid & target.bool()).sum().float()
    negative = (valid & ~target.bool()).sum().float()
    if float(fixed_positive_weight) > 0:
        positive_weight = positive.new_tensor(float(fixed_positive_weight))
    else:
        positive_weight = torch.where(
            positive > 0,
            (negative / positive.clamp_min(1.0)).clamp(
                1.0, float(maximum_positive_weight)),
            positive.new_ones(()),
        )
    loss = F.binary_cross_entropy_with_logits(
        logit, target, reduction='none', pos_weight=positive_weight
    )
    return masked_average(loss, valid), positive_weight


def balanced_direction_loss(logits, target, valid, maximum_balance):
    per_frame = F.cross_entropy(logits, target, reduction='none')
    counts = torch.stack([(valid & target.eq(index)).sum() for index in range(2)]).float()
    total = valid.sum().float()
    raw = total / (2.0 * counts.clamp_min(1.0))
    limit = max(float(maximum_balance), 1.0)
    class_weight = raw.clamp(1.0 / limit, limit)
    class_weight = torch.where(counts > 0, class_weight, torch.zeros_like(class_weight))
    return masked_average(per_frame, valid, class_weight[target]), class_weight


def compute_trc_v4b_loss_and_metrics(pred, gt_heatmap, gt_bbox, cfg):
    packet = pred['erc_trc_output']
    oracle = hard_decision_action_oracle(
        packet, gt_heatmap, cfg.TRAIN.ERC_TRC_V4B_ORACLE_STEPS,
        cfg.TRAIN.ERC_TRC_V4B_SCORE_TEMPERATURE,
        cfg.TRAIN.ERC_TRC_V4B_MIN_TARGET_GAIN,
    )
    valid = oracle['valid']
    target_action = oracle['target_action']
    corrective = valid & target_action.ne(0)
    preserve_mask = valid & target_action.eq(0)
    permission_target = target_action.ne(0).to(packet['permission_logit'])
    permission_loss, positive_weight = balanced_permission_loss(
        packet['permission_logit'], permission_target, valid,
        cfg.TRAIN.ERC_TRC_V4B_MAX_POSITIVE_WEIGHT,
        cfg.TRAIN.ERC_TRC_V4B_FIXED_POSITIVE_WEIGHT,
    )
    direction_target = (target_action - 1).clamp(0, 1)
    direction_loss, direction_weight = balanced_direction_loss(
        packet['direction_logits'], direction_target, corrective,
        cfg.TRAIN.ERC_TRC_V4B_MAX_DIRECTION_BALANCE,
    )
    strength_loss = masked_average(
        F.smooth_l1_loss(
            packet['strength'], oracle['target_strength'], reduction='none'
        ), corrective,
    )

    base_log = packet['log_base'].detach().flatten(1)
    difference = packet['evidence']['d'].detach().flatten(1)
    support = packet['support'].detach().flatten(1)
    model_log = base_log + packet['soft_alpha'].flatten(1) * difference
    batch_index = torch.arange(model_log.shape[0], device=model_log.device)
    target_value = model_log[batch_index, oracle['target_index']]
    base_value = model_log[batch_index, oracle['base_index']]
    margin = float(cfg.TRAIN.ERC_TRC_V4B_DECISION_MARGIN)
    temperature = max(float(cfg.TRAIN.ERC_TRC_V4B_MARGIN_TEMPERATURE), 1e-6)
    decision_loss = masked_average(
        F.softplus((margin - (target_value - base_value)) / temperature) * temperature,
        corrective,
    )
    alternatives = model_log.masked_fill(~support, -1e4).clone()
    alternatives[batch_index, oracle['base_index']] = -1e4
    strongest_alternative = alternatives.max(-1).values
    preserve_loss = masked_average(
        F.softplus(
            (margin - (base_value - strongest_alternative)) / temperature
        ) * temperature,
        preserve_mask,
    )
    action_risk = F.relu(
        oracle['action_hard_quality'][:, :1] - oracle['action_hard_quality']
    )
    risk_loss = masked_average(
        (packet['action_probability'] * action_risk).sum(-1), valid
    )
    loss = (
        cfg.TRAIN.ERC_TRC_V4B_PERMISSION_WEIGHT * permission_loss
        + cfg.TRAIN.ERC_TRC_V4B_DIRECTION_WEIGHT * direction_loss
        + cfg.TRAIN.ERC_TRC_V4B_STRENGTH_WEIGHT * strength_loss
        + cfg.TRAIN.ERC_TRC_V4B_DECISION_WEIGHT * decision_loss
        + cfg.TRAIN.ERC_TRC_V4B_PRESERVE_WEIGHT * preserve_loss
        + cfg.TRAIN.ERC_TRC_V4B_RISK_WEIGHT * risk_loss
    )
    status = {
        'Loss/erc_trc_v4b_permission': permission_loss.detach(),
        'Loss/erc_trc_v4b_direction': direction_loss.detach(),
        'Loss/erc_trc_v4b_strength': strength_loss.detach(),
        'Loss/erc_trc_v4b_decision': decision_loss.detach(),
        'Loss/erc_trc_v4b_preserve': preserve_loss.detach(),
        'Loss/erc_trc_v4b_risk': risk_loss.detach(),
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
        batch_index = torch.arange(base.shape[0], device=base.device)
        relevance = gt_heatmap.detach().flatten(1)
        response_gain = relevance[batch_index, full_index] - relevance[batch_index, base_index]
        predicted_action = packet['hard_action']
        active_alpha = active[:, None, None].expand_as(packet['alpha'])
        predicted_direction = packet['hard_direction_index']
        permission_positive = valid & permission_target.bool()
        permission_negative = valid & ~permission_target.bool()
        status.update({
            'ERC/active_fraction': active.float().mean(),
            'ERC/alpha_abs': mean(packet['alpha'].abs(), active_alpha),
            'ERC/alpha_mean': mean(packet['alpha'], active_alpha),
            'ERC/alpha_std': std(packet['alpha'], active_alpha),
            'ERC/permission_mean': mean(packet['permission_probability'], active),
            'ERC/hard_permission_fraction': mean(packet['hard_permission'], active),
            'ERC/permission_accuracy': mean(
                packet['hard_permission'].eq(permission_target).float(), valid),
            'ERC/permission_positive_mean': mean(
                packet['permission_probability'], permission_positive),
            'ERC/permission_negative_mean': mean(
                packet['permission_probability'], permission_negative),
            'ERC/permission_true_positive_rate': mean(
                packet['hard_permission'], permission_positive),
            'ERC/permission_false_positive_rate': mean(
                packet['hard_permission'], permission_negative),
            'ERC/direction_cls_mean': mean(packet['direction_probability'][:, 0], active),
            'ERC/direction_cont_mean': mean(packet['direction_probability'][:, 1], active),
            'ERC/direction_accuracy': mean(
                predicted_direction.eq(direction_target).float(), corrective),
            'ERC/hard_keep_fraction': mean(predicted_action.eq(0).float(), active),
            'ERC/hard_cls_fraction': mean(predicted_action.eq(1).float(), active),
            'ERC/hard_cont_fraction': mean(predicted_action.eq(2).float(), active),
            'ERC/oracle_keep_fraction': mean(target_action.eq(0).float(), valid),
            'ERC/oracle_cls_fraction': mean(target_action.eq(1).float(), valid),
            'ERC/oracle_cont_fraction': mean(target_action.eq(2).float(), valid),
            'ERC/oracle_response_advantage': mean(oracle['target_gain'], valid),
            'ERC/strength_mean': mean(packet['strength'], active),
            'ERC/strength_target_mean': mean(oracle['target_strength'], corrective),
            'ERC/selection_change_rate': mean(changed.float(), active),
            'ERC/changed_response_target_gain': mean(response_gain, changed),
            'ERC/changed_response_negative_rate': mean(
                response_gain.lt(-1e-6).float(), changed),
            'ERC/permission_positive_weight': positive_weight,
            'ERC/direction_weight_cls': direction_weight[0],
            'ERC/direction_weight_cont': direction_weight[1],
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
        iou_gain = candidate_iou[batch_index, full_index] - candidate_iou[batch_index, base_index]
        valid_box = (
            active & torch.isfinite(gt_bbox).all(-1)
            & (gt_bbox[:, 2:] > 0).all(-1)
        )
        status.update({
            'ERC/candidate_iou_gain': mean(iou_gain, valid_box),
            'ERC/changed_candidate_iou_gain': mean(iou_gain, changed & valid_box),
            'ERC/changed_candidate_count': (changed & valid_box).sum().float(),
            'ERC/changed_candidate_negative_rate': mean(
                iou_gain.lt(-1e-6).float(), changed & valid_box),
        })
    return loss, status

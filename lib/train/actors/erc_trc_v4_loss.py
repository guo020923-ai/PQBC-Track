"""Hard-decision supervision for TRC-v4.

Only actions that change the deployed Hann-aware argmax and improve the
official target-centre response are labelled corrective.  Candidate boxes and
IoU remain detached diagnostics, preserving the boundary between ERC and the
future candidate-quality module.
"""

import torch
from torch.nn import functional as F

from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy


def masked_average(values, mask, weight=None):
    mask = mask.to(values.dtype)
    if weight is not None:
        mask = mask * weight.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def hard_decision_action_oracle(packet, relevance, steps, score_temperature,
                                minimum_target_gain=0.0):
    """Find a reachable KEEP/C/T action aligned with deployed hard selection."""
    relevance = relevance.detach().flatten(1)
    base_log = packet['log_base'].detach().flatten(1)
    difference = packet['evidence']['d'].detach().flatten(1)
    support = packet['support'].detach().flatten(1)
    active = packet['active'].detach()
    count = max(int(steps), 2)
    magnitude = torch.linspace(
        float(packet['alpha_limit']) / count,
        float(packet['alpha_limit']), count,
        device=base_log.device, dtype=base_log.dtype,
    )
    base = base_log.masked_fill(~support, -1e4)
    cls_scores = base[:, None] + magnitude[None, :, None] * difference[:, None]
    cont_scores = base[:, None] - magnitude[None, :, None] * difference[:, None]
    cls_indices = cls_scores.argmax(-1)
    cont_indices = cont_scores.argmax(-1)
    base_index = base.argmax(-1)
    batch_index = torch.arange(base.shape[0], device=base.device)
    expanded_relevance = relevance[:, None].expand(-1, count, -1)
    cls_hard = expanded_relevance.gather(2, cls_indices[..., None])[..., 0]
    cont_hard = expanded_relevance.gather(2, cont_indices[..., None])[..., 0]
    keep_hard = relevance[batch_index, base_index]

    temperature = max(float(score_temperature), 1e-6)
    keep_probability = F.softmax(base / temperature, -1)
    cls_probability = F.softmax(cls_scores / temperature, -1)
    cont_probability = F.softmax(cont_scores / temperature, -1)
    keep_soft = (keep_probability * relevance).sum(-1)
    cls_soft = (cls_probability * relevance[:, None]).sum(-1)
    cont_soft = (cont_probability * relevance[:, None]).sum(-1)

    cls_gain = cls_hard - keep_hard[:, None]
    cont_gain = cont_hard - keep_hard[:, None]
    minimum = float(minimum_target_gain)
    cls_available = cls_indices.ne(base_index[:, None]) & (cls_gain > minimum)
    cont_available = cont_indices.ne(base_index[:, None]) & (cont_gain > minimum)
    # Maximize deployed hard-response gain first.  torch.argmax returns the
    # first maximum, so equal-quality corrections use the smallest reachable
    # magnitude rather than the strongest intervention.  Soft quality only
    # breaks a tie between the best C and T families afterwards.
    unavailable = base.new_full(cls_gain.shape, -1e4)
    cls_hard_value = torch.where(cls_available, cls_gain, unavailable)
    cont_hard_value = torch.where(cont_available, cont_gain, unavailable)
    cls_best_gain, cls_step = cls_hard_value.max(-1)
    cont_best_gain, cont_step = cont_hard_value.max(-1)
    cls_best_soft_gain = (
        cls_soft - keep_soft[:, None]
    ).gather(1, cls_step[:, None])[:, 0]
    cont_best_soft_gain = (
        cont_soft - keep_soft[:, None]
    ).gather(1, cont_step[:, None])[:, 0]
    cls_best_value = cls_best_gain + 0.001 * F.relu(cls_best_soft_gain)
    cont_best_value = cont_best_gain + 0.001 * F.relu(cont_best_soft_gain)
    cls_exists, cont_exists = cls_available.any(-1), cont_available.any(-1)

    zero = base.new_zeros(base.shape[0])
    action_value = torch.stack([
        zero,
        torch.where(cls_exists, cls_best_value, zero.new_full(zero.shape, -1e4)),
        torch.where(cont_exists, cont_best_value, zero.new_full(zero.shape, -1e4)),
    ], -1)
    target_action = action_value.argmax(-1)
    cls_target_index = cls_indices.gather(1, cls_step[:, None])[:, 0]
    cont_target_index = cont_indices.gather(1, cont_step[:, None])[:, 0]
    target_index = torch.where(
        target_action.eq(1), cls_target_index,
        torch.where(target_action.eq(2), cont_target_index, base_index),
    )
    cls_strength = magnitude[cls_step] / float(packet['alpha_limit'])
    cont_strength = magnitude[cont_step] / float(packet['alpha_limit'])
    target_strength = torch.where(
        target_action.eq(1), cls_strength,
        torch.where(target_action.eq(2), cont_strength, zero),
    )
    target_gain = relevance[batch_index, target_index] - keep_hard
    valid = active & support.any(-1) & (
        relevance.max(-1).values - relevance.min(-1).values > 1e-6
    )
    target_action = torch.where(valid, target_action, torch.zeros_like(target_action))
    target_strength = torch.where(valid, target_strength, zero)
    target_gain = torch.where(valid, target_gain, zero)

    cls_best_hard = cls_hard.gather(1, cls_step[:, None])[:, 0]
    cont_best_hard = cont_hard.gather(1, cont_step[:, None])[:, 0]
    action_hard_quality = torch.stack([keep_hard, cls_best_hard, cont_best_hard], -1)
    return {
        'valid': valid,
        'target_action': target_action,
        'target_strength': target_strength,
        'target_index': target_index,
        'target_gain': target_gain,
        'base_index': base_index,
        'action_hard_quality': action_hard_quality,
        'cls_available': cls_exists & valid,
        'cont_available': cont_exists & valid,
    }


def _balanced_action_loss(logits, target, valid, maximum_balance):
    per_frame = F.cross_entropy(logits, target, reduction='none')
    counts = torch.stack([(valid & target.eq(index)).sum() for index in range(3)]).float()
    total = valid.sum().float()
    raw = total / (3.0 * counts.clamp_min(1.0))
    limit = max(float(maximum_balance), 1.0)
    class_weight = raw.clamp(1.0 / limit, limit)
    class_weight = torch.where(counts > 0, class_weight, torch.zeros_like(class_weight))
    return masked_average(per_frame, valid, class_weight[target]), class_weight


def compute_trc_v4_loss_and_metrics(pred, gt_heatmap, gt_bbox, cfg):
    packet = pred['erc_trc_output']
    oracle = hard_decision_action_oracle(
        packet, gt_heatmap, cfg.TRAIN.ERC_TRC_V4_ORACLE_STEPS,
        cfg.TRAIN.ERC_TRC_V4_SCORE_TEMPERATURE,
        cfg.TRAIN.ERC_TRC_V4_MIN_TARGET_GAIN,
    )
    valid = oracle['valid']
    target_action = oracle['target_action']
    corrective = valid & target_action.ne(0)
    preserve_mask = valid & target_action.eq(0)
    action_loss, class_weight = _balanced_action_loss(
        packet['action_logits'], target_action, valid,
        cfg.TRAIN.ERC_TRC_V4_MAX_CLASS_BALANCE,
    )
    strength_error = F.smooth_l1_loss(
        packet['strength'], oracle['target_strength'], reduction='none'
    )
    strength_loss = masked_average(strength_error, corrective)

    base_log = packet['log_base'].detach().flatten(1)
    difference = packet['evidence']['d'].detach().flatten(1)
    support = packet['support'].detach().flatten(1)
    # The soft policy supplies stable gradients, while the forward/tracker
    # path uses the deterministic hard action from the same logits.
    model_log = base_log + packet['soft_alpha'].flatten(1) * difference
    batch_index = torch.arange(model_log.shape[0], device=model_log.device)
    target_value = model_log[batch_index, oracle['target_index']]
    base_value = model_log[batch_index, oracle['base_index']]
    margin = float(cfg.TRAIN.ERC_TRC_V4_DECISION_MARGIN)
    temperature = max(float(cfg.TRAIN.ERC_TRC_V4_MARGIN_TEMPERATURE), 1e-6)
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
        cfg.TRAIN.ERC_TRC_V4_ACTION_WEIGHT * action_loss
        + cfg.TRAIN.ERC_TRC_V4_STRENGTH_WEIGHT * strength_loss
        + cfg.TRAIN.ERC_TRC_V4_DECISION_WEIGHT * decision_loss
        + cfg.TRAIN.ERC_TRC_V4_PRESERVE_WEIGHT * preserve_loss
        + cfg.TRAIN.ERC_TRC_V4_RISK_WEIGHT * risk_loss
    )
    status = {
        'Loss/erc_trc_v4_action': action_loss.detach(),
        'Loss/erc_trc_v4_strength': strength_loss.detach(),
        'Loss/erc_trc_v4_decision': decision_loss.detach(),
        'Loss/erc_trc_v4_preserve': preserve_loss.detach(),
        'Loss/erc_trc_v4_risk': risk_loss.detach(),
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
        response_gain = (
            relevance[batch_index, full_index]
            - relevance[batch_index, base_index]
        )
        predicted_action = packet['hard_action']
        active_alpha = active[:, None, None].expand_as(packet['alpha'])
        status.update({
            'ERC/active_fraction': active.float().mean(),
            'ERC/alpha_abs': mean(packet['alpha'].abs(), active_alpha),
            'ERC/alpha_mean': mean(packet['alpha'], active_alpha),
            'ERC/alpha_std': std(packet['alpha'], active_alpha),
            'ERC/action_keep_mean': mean(packet['action_probability'][:, 0], active),
            'ERC/action_cls_mean': mean(packet['action_probability'][:, 1], active),
            'ERC/action_cont_mean': mean(packet['action_probability'][:, 2], active),
            'ERC/hard_keep_fraction': mean(predicted_action.eq(0).float(), active),
            'ERC/hard_cls_fraction': mean(predicted_action.eq(1).float(), active),
            'ERC/hard_cont_fraction': mean(predicted_action.eq(2).float(), active),
            'ERC/action_accuracy': mean(predicted_action.eq(target_action).float(), valid),
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
            'ERC/action_class_weight_keep': class_weight[0],
            'ERC/action_class_weight_cls': class_weight[1],
            'ERC/action_class_weight_cont': class_weight[2],
        })

        # Diagnostic only: bbox geometry and IoU never enter an ERC target.
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
            'ERC/changed_candidate_iou_gain': mean(iou_gain, changed & valid_box),
            'ERC/changed_candidate_count': (changed & valid_box).sum().float(),
            'ERC/changed_candidate_negative_rate': mean(
                iou_gain.lt(-1e-6).float(), changed & valid_box),
        })
    return loss, status

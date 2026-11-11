"""Detached candidate-IoU utility supervision for FER-v7.

The deployable module remains response-only.  Candidate boxes and ground truth
are used here only to construct detached action targets and differentiable
score-ranking losses.  Consequently this objective can teach ERC which C/T
calibration action improves the final tracking box without allowing ERC loss
gradients to alter bbox geometry or turning ERC into a candidate-quality head.
"""

import torch
from torch.nn import functional as F

from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy

from .erc_branch_aligned_tristate_multiaction_loss import (
    _balanced_mean, _masked_mean,
)


def _candidate_iou_map(candidate_boxes, gt_bbox):
    """Return detached pairwise IoU for BxS cxcywh candidates and B xywh GT."""
    candidates = box_cxcywh_to_xyxy(candidate_boxes.detach()).clamp(0.0, 1.0)
    target = box_xywh_to_xyxy(gt_bbox.detach()).clamp(0.0, 1.0)[:, None]
    intersection_lt = torch.maximum(candidates[..., :2], target[..., :2])
    intersection_rb = torch.minimum(candidates[..., 2:], target[..., 2:])
    intersection_wh = (intersection_rb - intersection_lt).clamp_min(0.0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    candidate_wh = (candidates[..., 2:] - candidates[..., :2]).clamp_min(0.0)
    target_wh = (target[..., 2:] - target[..., :2]).clamp_min(0.0)
    candidate_area = candidate_wh[..., 0] * candidate_wh[..., 1]
    target_area = target_wh[..., 0] * target_wh[..., 1]
    union = candidate_area + target_area - intersection
    return torch.nan_to_num(intersection / union.clamp_min(1e-6))


def _action_magnitudes(pred, number_actions, device):
    magnitude_index = pred['erc_v6_action_magnitude_index'].detach()
    magnitudes = pred['erc_v6_magnitudes'].detach().to(device=device)
    result = torch.zeros(number_actions, device=device, dtype=magnitudes.dtype)
    nonkeep = magnitude_index.ge(0)
    result[nonkeep] = magnitudes[magnitude_index[nonkeep]]
    return result


@torch.no_grad()
def build_iou_utility_oracle(pred, gt_heatmap, gt_bbox, cfg):
    """Choose the smallest deployable action with a robust candidate-IoU gain."""
    log_base = pred['erc_v6_log_base'].detach().flatten(1)
    corrections = pred['erc_v6_action_correction_maps'].detach().flatten(2)
    support = pred['erc_v6_support_mask'].detach().flatten(1).bool()
    response_target = gt_heatmap.detach().flatten(1).clamp_min(0.0)
    candidate_iou = _candidate_iou_map(pred['bbox_map'], gt_bbox)
    active = pred['erc_active_mask'].detach().reshape(-1).bool()
    valid = (
        active & support.any(-1) & response_target.max(-1).values.gt(0.0)
        & torch.isfinite(candidate_iou).all(-1)
    )

    scores = (log_base[:, None] + corrections).masked_fill(
        ~support[:, None], torch.finfo(log_base.dtype).min)
    hard_index = scores.argmax(-1)
    hard_iou = candidate_iou[:, None].expand_as(scores).gather(
        -1, hard_index[..., None]).squeeze(-1)
    hard_iou_gain = hard_iou - hard_iou[:, :1]
    changed = hard_index.ne(hard_index[:, :1])

    rows = torch.arange(scores.shape[0], device=scores.device)[:, None]
    hard_response = response_target[rows, hard_index]
    hard_response_gain = hard_response - hard_response[:, :1]

    topk = min(int(cfg.TRAIN.ERC_V7_ORACLE_TOPK), scores.shape[-1])
    top_values, top_indices = scores.topk(topk, dim=-1)
    temperature = max(float(cfg.TRAIN.ERC_V7_ORACLE_TEMPERATURE), 1e-6)
    top_probability = F.softmax(top_values / temperature, -1)
    top_iou = candidate_iou[:, None].expand_as(scores).gather(-1, top_indices)
    soft_iou = (top_probability * top_iou).sum(-1)
    soft_iou_gain = soft_iou - soft_iou[:, :1]
    top_response = response_target[:, None].expand_as(scores).gather(
        -1, top_indices)
    soft_response = (top_probability * top_response).sum(-1)
    soft_response_gain = soft_response - soft_response[:, :1]

    eligible = (
        changed
        & hard_iou_gain.ge(float(cfg.TRAIN.ERC_V7_ORACLE_MIN_IOU_GAIN))
        & soft_iou_gain.ge(float(cfg.TRAIN.ERC_V7_ORACLE_MIN_SOFT_IOU_GAIN))
        & hard_response_gain.ge(
            float(cfg.TRAIN.ERC_V7_ORACLE_MIN_RESPONSE_GAIN))
        & valid[:, None]
    )
    eligible[:, 0] = False
    action_magnitude = _action_magnitudes(
        pred, scores.shape[1], scores.device)
    tie_tolerance = max(float(cfg.TRAIN.ERC_V7_ORACLE_IOU_TIE_TOL), 0.0)
    target = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
    for batch_index in valid.nonzero(as_tuple=False).flatten().tolist():
        candidates = eligible[batch_index].nonzero(as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        gains = hard_iou_gain[batch_index, candidates]
        candidates = candidates[gains.ge(gains.max() - tie_tolerance)]
        sizes = action_magnitude[candidates]
        candidates = candidates[sizes.le(sizes.min() + 1e-7)]
        if candidates.numel() > 1:
            gains = soft_iou_gain[batch_index, candidates]
            candidates = candidates[gains.ge(gains.max() - 1e-7)]
        if candidates.numel() > 1:
            gains = soft_response_gain[batch_index, candidates]
            candidates = candidates[gains.ge(gains.max() - 1e-7)]
        target[batch_index] = candidates[0]

    return {
        'valid': valid,
        'target': target,
        'scores': scores,
        'hard_index': hard_index,
        'hard_iou': hard_iou,
        'hard_iou_gain': hard_iou_gain,
        'soft_iou': soft_iou,
        'soft_iou_gain': soft_iou_gain,
        'candidate_iou': candidate_iou,
        'hard_response_gain': hard_response_gain,
        'soft_response': soft_response,
        'soft_response_gain': soft_response_gain,
        'changed': changed,
        'eligible': eligible,
    }


def _utility_balanced_permission_loss(
        permission_logit, permission_target, positive, negative,
        empty_negative_weight, zero):
    frame_loss = F.binary_cross_entropy_with_logits(
        permission_logit, permission_target, reduction='none')
    positive_loss = _masked_mean(frame_loss, positive, zero)
    negative_loss = _masked_mean(frame_loss, negative, zero)
    positive_batch = positive.any()
    if positive_batch:
        loss = 0.5 * (positive_loss + negative_loss)
    else:
        loss = float(empty_negative_weight) * negative_loss
    return loss, positive_loss, negative_loss, positive_batch


def _utility_permission_rank_loss(
        permission_logit, positive, negative, margin, zero):
    """Separate sparse actionable frames from KEEP frames within a batch.

    Empty-positive batches intentionally contribute no ranking gradient.  This
    prevents the much more frequent all-KEEP batches from collapsing the gate,
    while positive batches still compare every actionable example with the
    available KEEP examples.
    """
    if not positive.any() or not negative.any():
        return zero
    positive_logit = permission_logit[positive][:, None]
    negative_logit = permission_logit[negative][None, :]
    return F.softplus(
        float(margin) - positive_logit + negative_logit).mean()


def compute_iou_utility_loss_and_metrics(
        pred, gt_heatmap, gt_bbox, cfg):
    permission_logit = pred['erc_v6_permission_logit']
    zero = permission_logit.sum() * 0.0
    oracle = build_iou_utility_oracle(pred, gt_heatmap, gt_bbox, cfg)
    valid = oracle['valid']
    if not valid.any():
        return zero, {
            'Loss/erc_total': zero.detach(),
            'ERC/v7_valid_fraction': valid.float().mean(),
        }

    target = oracle['target']
    direction = pred['erc_v6_action_direction'].long()
    scope = pred['erc_v6_action_scope'].long()
    magnitude_index = pred['erc_v6_action_magnitude_index'].long()
    number_actions = direction.numel()
    intervene = target.ne(0)
    target_direction = (direction[target] - 1).clamp_min(0)
    target_scope = scope[target]
    target_magnitude = magnitude_index[target].clamp_min(0)

    rows = torch.arange(target.shape[0], device=target.device)
    target_iou_gain = oracle['hard_iou_gain'][rows, target]
    minimum_gain = float(cfg.TRAIN.ERC_V7_ORACLE_MIN_IOU_GAIN)
    gain_scale = max(float(cfg.TRAIN.ERC_V7_PERMISSION_GAIN_SCALE), minimum_gain + 1e-6)
    minimum_target = float(cfg.TRAIN.ERC_V7_PERMISSION_MIN_TARGET)
    utility_strength = (
        (target_iou_gain - minimum_gain) / (gain_scale - minimum_gain)
    ).clamp(0.0, 1.0)
    permission_target = torch.where(
        intervene,
        minimum_target + (1.0 - minimum_target) * utility_strength,
        torch.zeros_like(utility_strength),
    )
    permission_loss, permission_positive_loss, permission_negative_loss, \
        permission_positive_batch = _utility_balanced_permission_loss(
            permission_logit,
            permission_target,
            valid & intervene,
            valid & ~intervene,
            cfg.TRAIN.ERC_V7_PERMISSION_EMPTY_NEGATIVE_WEIGHT,
            zero,
        )
    permission_rank_loss = _utility_permission_rank_loss(
        permission_logit,
        valid & intervene,
        valid & ~intervene,
        cfg.TRAIN.ERC_V7_PERMISSION_RANK_MARGIN,
        zero,
    )

    ema = pred['erc_v6_action_class_ema']
    with torch.no_grad():
        frequency = torch.bincount(
            target[valid], minlength=number_actions).to(ema)
        frequency = frequency / frequency.sum().clamp_min(1)
        if permission_logit.requires_grad:
            momentum = float(cfg.TRAIN.ERC_V7_CLASS_EMA_MOMENTUM)
            ema.mul_(momentum).add_(frequency, alpha=1.0 - momentum)
            ema.div_(ema.sum().clamp_min(1e-6))
    max_weight = float(cfg.TRAIN.ERC_V7_MAX_CLASS_WEIGHT)

    direction_frame = F.cross_entropy(
        pred['erc_v6_direction_logits'], target_direction, reduction='none')
    direction_frequency = torch.stack([
        ema[direction.eq(1)].sum(), ema[direction.eq(2)].sum()])
    direction_loss = _balanced_mean(
        direction_frame[valid & intervene], target_direction[valid & intervene],
        direction_frequency, max_weight, zero)

    scope_valid = valid & intervene & direction[target].eq(2)
    scope_frame = F.cross_entropy(
        pred['erc_v6_scope_logits'], target_scope, reduction='none')
    scope_frequency = torch.stack([
        ema[(direction.eq(2) & scope.eq(0))].sum(),
        ema[(direction.eq(2) & scope.eq(1))].sum(),
    ])
    scope_loss = _balanced_mean(
        scope_frame[scope_valid], target_scope[scope_valid],
        scope_frequency, max_weight, zero)

    magnitude_frame = F.cross_entropy(
        pred['erc_v6_magnitude_logits'], target_magnitude, reduction='none')
    magnitude_frequency = torch.stack([
        ema[magnitude_index.eq(index)].sum()
        for index in range(pred['erc_v6_magnitudes'].numel())
    ])
    magnitude_loss = _balanced_mean(
        magnitude_frame[valid & intervene], target_magnitude[valid & intervene],
        magnitude_frequency, max_weight, zero)

    joint_probability = pred['erc_v6_joint_probability'].clamp_min(1e-8)
    action_frame = -joint_probability.log().gather(
        1, target[:, None]).squeeze(1)
    if (bool(cfg.TRAIN.ERC_V7_SKIP_EMPTY_ACTION_BATCH)
            and not permission_positive_batch):
        action_loss = zero
    else:
        action_loss = _balanced_mean(
            action_frame[valid], target[valid], ema, max_weight, zero)

    log_base = pred['erc_v6_log_base'].detach().flatten(1)
    learned_score = log_base + pred[
        'erc_v6_predicted_correction_map'].flatten(1)
    support = pred['erc_v6_support_mask'].flatten(1).bool()
    learned_score = learned_score.masked_fill(
        ~support, torch.finfo(learned_score.dtype).min)
    topk = min(int(cfg.TRAIN.ERC_V7_ORACLE_TOPK), learned_score.shape[-1])
    temperature = max(float(cfg.TRAIN.ERC_V7_ORACLE_TEMPERATURE), 1e-6)
    learned_values, learned_indices = learned_score.topk(topk, -1)
    learned_probability = F.softmax(learned_values / temperature, -1)
    candidate_iou = oracle['candidate_iou']
    learned_iou_expectation = (
        learned_probability * candidate_iou.gather(-1, learned_indices)
    ).sum(-1)
    oracle_iou_expectation = oracle['soft_iou'][rows, target].detach()
    decision_loss = _masked_mean(
        F.relu(oracle_iou_expectation - learned_iou_expectation),
        valid & intervene, zero)

    response_target = gt_heatmap.detach().flatten(1).clamp_min(0.0)
    learned_response_expectation = (
        learned_probability * response_target.gather(-1, learned_indices)
    ).sum(-1)
    oracle_response_expectation = oracle['soft_response'][rows, target].detach()
    response_aux_loss = _masked_mean(
        F.relu(oracle_response_expectation - learned_response_expectation),
        valid & intervene, zero)

    baseline_iou_expectation = oracle['soft_iou'][:, 0].detach()
    soft_noharm = F.relu(baseline_iou_expectation - learned_iou_expectation)
    base_index = log_base.argmax(-1)
    learned_index = learned_score.argmax(-1)
    changed = learned_index.ne(base_index)
    baseline_iou = candidate_iou[rows, base_index]
    learned_iou = candidate_iou[rows, learned_index]
    candidate_iou_gain = learned_iou - baseline_iou
    harmful = valid & changed & candidate_iou_gain.lt(0.0)
    hard_noharm = F.relu(
        learned_score[rows, learned_index] - learned_score[rows, base_index])
    noharm_loss = (
        _masked_mean(soft_noharm, valid & ~intervene, zero)
        + _masked_mean(hard_noharm, harmful, zero)
    )

    response_gain = (
        response_target[rows, learned_index]
        - response_target[rows, base_index]
    )
    total = (
        float(cfg.TRAIN.ERC_V7_PERMISSION_WEIGHT) * permission_loss
        + float(cfg.TRAIN.ERC_V7_PERMISSION_RANK_WEIGHT)
        * permission_rank_loss
        + float(cfg.TRAIN.ERC_V7_DIRECTION_WEIGHT) * direction_loss
        + float(cfg.TRAIN.ERC_V7_SCOPE_WEIGHT) * scope_loss
        + float(cfg.TRAIN.ERC_V7_MAGNITUDE_WEIGHT) * magnitude_loss
        + float(cfg.TRAIN.ERC_V7_ACTION_WEIGHT) * action_loss
        + float(cfg.TRAIN.ERC_V7_DECISION_WEIGHT) * decision_loss
        + float(cfg.TRAIN.ERC_V7_NOHARM_WEIGHT) * noharm_loss
        + float(cfg.TRAIN.ERC_V7_RESPONSE_AUX_WEIGHT) * response_aux_loss
    )

    with torch.no_grad():
        predicted = pred['erc_v6_raw_hard_action']
        predicted_permission = permission_logit.gt(0.0)

        def mean(values, mask=valid):
            return _masked_mean(values, mask, zero.detach())

        status = {
            'Loss/erc_total': total.detach(),
            'Loss/erc_v7_permission': permission_loss.detach(),
            'Loss/erc_v7_permission_positive': permission_positive_loss.detach(),
            'Loss/erc_v7_permission_negative': permission_negative_loss.detach(),
            'Loss/erc_v7_permission_rank': permission_rank_loss.detach(),
            'Loss/erc_v7_direction': direction_loss.detach(),
            'Loss/erc_v7_scope': scope_loss.detach(),
            'Loss/erc_v7_magnitude': magnitude_loss.detach(),
            'Loss/erc_v7_action': action_loss.detach(),
            'Loss/erc_v7_decision_iou': decision_loss.detach(),
            'Loss/erc_v7_noharm_iou': noharm_loss.detach(),
            'Loss/erc_v7_response_aux': response_aux_loss.detach(),
            'ERC/v7_valid_fraction': valid.float().mean(),
            'ERC/v7_oracle_keep_rate': mean(target.eq(0).float()),
            'ERC/v7_oracle_cls_rate': mean(direction[target].eq(1).float()),
            'ERC/v7_oracle_cont_rate': mean(direction[target].eq(2).float()),
            'ERC/v7_oracle_regional_rate': mean(scope[target].eq(1).float()),
            'ERC/v7_pred_keep_rate': mean(predicted.eq(0).float()),
            'ERC/v7_pred_cls_rate': mean(direction[predicted].eq(1).float()),
            'ERC/v7_pred_cont_rate': mean(direction[predicted].eq(2).float()),
            'ERC/v7_pred_regional_rate': mean(scope[predicted].eq(1).float()),
            'ERC/v7_action_accuracy': mean(predicted.eq(target).float()),
            'ERC/v7_selection_change_rate': mean(changed.float()),
            'ERC/v7_candidate_iou_gain': mean(candidate_iou_gain),
            'ERC/v7_changed_candidate_iou_gain': mean(
                candidate_iou_gain, valid & changed),
            'ERC/v7_beneficial_change_rate': mean(
                (changed & candidate_iou_gain.gt(0.0)).float()),
            'ERC/v7_harmful_change_rate': mean(
                (changed & candidate_iou_gain.lt(0.0)).float()),
            'ERC/v7_response_gain': mean(response_gain),
            'ERC/v7_permission_mean': mean(
                pred['erc_v6_permission_probability']),
            'ERC/v7_permission_target_mean': mean(permission_target),
            'ERC/v7_permission_positive_batch': (
                permission_positive_batch.to(permission_logit).detach()),
            'ERC/v7_permission_positive_probability': mean(
                pred['erc_v6_permission_probability'], valid & intervene),
            'ERC/v7_permission_negative_probability': mean(
                pred['erc_v6_permission_probability'], valid & ~intervene),
            'ERC/v7_permission_recall': mean(
                predicted_permission.float(), valid & intervene),
            'ERC/v7_permission_false_positive_rate': mean(
                predicted_permission.float(), valid & ~intervene),
            'ERC/v7_oracle_hard_iou_gain': mean(target_iou_gain),
            'ERC/v7_oracle_soft_iou_gain': mean(
                oracle['soft_iou_gain'][rows, target]),
            'ERC/v7_oracle_hard_response_gain': mean(
                oracle['hard_response_gain'][rows, target]),
        }
    return total, status

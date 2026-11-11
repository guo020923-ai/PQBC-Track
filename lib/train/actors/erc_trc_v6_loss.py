"""Deployment-aligned response-utility supervision for TRC-v6.

All targets are derived from the official centre-response heatmap and the
reachable C/T calibration family.  Candidate boxes and IoU are detached
diagnostics only; they never supervise ERC.
"""

import torch
from torch.nn import functional as F

from lib.train.actors.erc_trc_v4_loss import masked_average
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy


@torch.no_grad()
def branch_response_utility_oracle(packet, relevance, steps,
                                   minimum_target_gain=0.05):
    """Return best reachable response gain and strength for each C/T action.

    Unlike a one-hot action Oracle, this retains the signed value of both
    branches.  A branch whose every decision-changing action is harmful gets
    a negative target; a branch that cannot change the decision gets zero.
    """
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
    base_index = base.argmax(-1)
    row = torch.arange(base.shape[0], device=base.device)
    base_quality = relevance[row, base_index]

    gains, strengths, indices, exists = [], [], [], []
    for sign in (1.0, -1.0):
        score = base[:, None] + sign * magnitude[None, :, None] * difference[:, None]
        index = score.argmax(-1)
        quality = relevance[:, None].expand_as(score).gather(
            2, index[..., None]
        )[..., 0]
        gain = quality - base_quality[:, None]
        changed = index.ne(base_index[:, None])
        available = changed.any(-1)
        masked_gain = gain.masked_fill(~changed, -1e4)
        best_gain, best_step = masked_gain.max(-1)
        best_gain = torch.where(available, best_gain, torch.zeros_like(best_gain))
        best_index = index.gather(1, best_step[:, None])[:, 0]
        best_index = torch.where(available, best_index, base_index)
        gains.append(best_gain)
        strengths.append(magnitude[best_step] / float(packet['alpha_limit']))
        indices.append(best_index)
        exists.append(available)

    branch_gain = torch.stack(gains, -1).clamp(-1.0, 1.0)
    branch_strength = torch.stack(strengths, -1)
    branch_index = torch.stack(indices, -1)
    branch_exists = torch.stack(exists, -1)
    valid = active & support.any(-1) & (
        relevance.max(-1).values - relevance.min(-1).values > 1e-6
    )
    branch_gain = torch.where(
        valid[:, None], branch_gain, torch.zeros_like(branch_gain)
    )
    branch_strength = torch.where(
        (valid[:, None] & branch_exists), branch_strength,
        torch.zeros_like(branch_strength),
    )
    minimum = float(minimum_target_gain)
    corrective_gain = torch.where(
        branch_exists & (branch_gain > minimum), branch_gain,
        branch_gain.new_full(branch_gain.shape, -1e4),
    )
    best_gain, best_branch = corrective_gain.max(-1)
    corrective = valid & (best_gain > minimum)
    target_action = torch.where(
        corrective, best_branch + 1, torch.zeros_like(best_branch)
    )
    target_index = branch_index.gather(
        1, best_branch[:, None]
    )[:, 0]
    target_index = torch.where(corrective, target_index, base_index)
    target_strength = branch_strength.gather(
        1, best_branch[:, None]
    )[:, 0]
    target_strength = torch.where(
        corrective, target_strength, torch.zeros_like(target_strength)
    )
    return {
        'valid': valid,
        'branch_gain': branch_gain,
        'branch_strength': branch_strength,
        'branch_exists': branch_exists,
        'target_action': target_action,
        'target_index': target_index,
        'target_strength': target_strength,
        'base_index': base_index,
    }


def _balanced_action_loss(logits, target, valid, maximum_balance):
    per_frame = F.cross_entropy(logits, target, reduction='none')
    counts = torch.stack([
        (valid & target.eq(index)).sum() for index in range(3)
    ]).to(logits)
    total = valid.sum().to(logits)
    raw = total / (3.0 * counts.clamp_min(1.0))
    limit = max(float(maximum_balance), 1.0)
    class_weight = raw.clamp(1.0 / limit, limit)
    class_weight = torch.where(
        counts > 0, class_weight, torch.zeros_like(class_weight)
    )
    return masked_average(per_frame, valid, class_weight[target]), class_weight


def _balanced_utility_loss(prediction, target, valid, minimum, positive_weight):
    error = F.smooth_l1_loss(prediction, target, reduction='none')
    positive = valid[:, None] & (target > float(minimum))
    negative = valid[:, None] & ~positive
    weight = torch.where(
        positive, error.new_full(error.shape, float(positive_weight)),
        torch.ones_like(error),
    )
    mask = positive | negative
    return (error * weight * mask).sum() / (weight * mask).sum().clamp_min(1.0)


def compute_trc_v6_loss_and_metrics(pred, gt_heatmap, gt_bbox, cfg):
    packet = pred['erc_trc_output']
    minimum = cfg.TRAIN.ERC_TRC_V6_MIN_TARGET_GAIN
    oracle = branch_response_utility_oracle(
        packet, gt_heatmap, cfg.TRAIN.ERC_TRC_V6_ORACLE_STEPS, minimum,
    )
    valid = oracle['valid']
    target_action = oracle['target_action']
    corrective = valid & target_action.ne(0)
    preserve = valid & target_action.eq(0)

    action_loss, class_weight = _balanced_action_loss(
        packet['action_logits'], target_action, valid,
        cfg.TRAIN.ERC_TRC_V6_MAX_CLASS_BALANCE,
    )
    utility_loss = _balanced_utility_loss(
        packet['branch_utility'], oracle['branch_gain'], valid, minimum,
        cfg.TRAIN.ERC_TRC_V6_POSITIVE_UTILITY_WEIGHT,
    )

    target_logit = packet['action_logits'].gather(
        1, target_action[:, None]
    )[:, 0]
    alternative = packet['action_logits'].clone()
    alternative.scatter_(1, target_action[:, None], -1e4)
    strongest_alternative = alternative.max(-1).values
    ranking_loss = masked_average(
        F.softplus(
            cfg.TRAIN.ERC_TRC_V6_ACTION_MARGIN
            - target_logit + strongest_alternative
        ), valid,
    )

    selected_branch = (target_action - 1).clamp(0, 1)
    predicted_strength = packet['branch_strength'].gather(
        1, selected_branch[:, None]
    )[:, 0]
    strength_loss = masked_average(
        F.smooth_l1_loss(
            predicted_strength, oracle['target_strength'], reduction='none'
        ), corrective,
    )

    # Keep ERC auxiliary gradients inside ERC.  Head+ERC are jointly trained,
    # but the original Head is updated by its native losses/distillation, not
    # by detached Oracle response targets.
    model_log = (
        packet['log_base'].detach()
        + packet['alpha'] * packet['evidence']['d'].detach()
    ).flatten(1)
    row = torch.arange(model_log.shape[0], device=model_log.device)
    target_value = model_log[row, oracle['target_index']]
    base_value = model_log[row, oracle['base_index']]
    decision_temperature = max(
        float(cfg.TRAIN.ERC_TRC_V6_DECISION_TEMPERATURE), 1e-6
    )
    decision_loss = masked_average(
        F.softplus((
            cfg.TRAIN.ERC_TRC_V6_DECISION_MARGIN
            - target_value + base_value
        ) / decision_temperature) * decision_temperature,
        corrective,
    )

    maximum_utility = packet['branch_utility'].max(-1).values
    noharm_boundary = (
        float(packet['utility_threshold'])
        - cfg.TRAIN.ERC_TRC_V6_NOHARM_MARGIN
    )
    noharm_loss = masked_average(
        F.softplus((maximum_utility - noharm_boundary)
                   / decision_temperature) * decision_temperature,
        preserve,
    )
    branch_risk = F.relu(float(minimum) - oracle['branch_gain'])
    risk_loss = masked_average(
        (packet['action_probability'][:, 1:] * branch_risk).sum(-1), valid
    )

    loss = (
        cfg.TRAIN.ERC_TRC_V6_ACTION_WEIGHT * action_loss
        + cfg.TRAIN.ERC_TRC_V6_UTILITY_WEIGHT * utility_loss
        + cfg.TRAIN.ERC_TRC_V6_RANK_WEIGHT * ranking_loss
        + cfg.TRAIN.ERC_TRC_V6_STRENGTH_WEIGHT * strength_loss
        + cfg.TRAIN.ERC_TRC_V6_DECISION_WEIGHT * decision_loss
        + cfg.TRAIN.ERC_TRC_V6_NOHARM_WEIGHT * noharm_loss
        + cfg.TRAIN.ERC_TRC_V6_RISK_WEIGHT * risk_loss
    )
    status = {
        'Loss/erc_trc_v6_action': action_loss.detach(),
        'Loss/erc_trc_v6_utility': utility_loss.detach(),
        'Loss/erc_trc_v6_rank': ranking_loss.detach(),
        'Loss/erc_trc_v6_strength': strength_loss.detach(),
        'Loss/erc_trc_v6_decision': decision_loss.detach(),
        'Loss/erc_trc_v6_noharm': noharm_loss.detach(),
        'Loss/erc_trc_v6_risk': risk_loss.detach(),
        'Loss/erc_total': loss.detach(),
    }

    with torch.no_grad():
        zero = loss.detach().new_zeros(())

        def mean(value, mask):
            return value[mask].mean() if mask.any() else zero

        active = packet['active']
        base = packet['base'].flatten(1)
        score = packet['score'].flatten(1)
        base_index, full_index = base.argmax(-1), score.argmax(-1)
        changed = base_index.ne(full_index) & active
        relevance = gt_heatmap.detach().flatten(1)
        response_gain = relevance[row, full_index] - relevance[row, base_index]
        hard_action = packet['hard_action']
        status.update({
            'ERC/active_fraction': active.float().mean(),
            'ERC/hard_keep_fraction': mean(hard_action.eq(0).float(), active),
            'ERC/hard_cls_fraction': mean(hard_action.eq(1).float(), active),
            'ERC/hard_cont_fraction': mean(hard_action.eq(2).float(), active),
            'ERC/action_accuracy': mean(hard_action.eq(target_action).float(), valid),
            'ERC/oracle_keep_fraction': mean(target_action.eq(0).float(), valid),
            'ERC/oracle_cls_fraction': mean(target_action.eq(1).float(), valid),
            'ERC/oracle_cont_fraction': mean(target_action.eq(2).float(), valid),
            'ERC/utility_cls_mean': mean(packet['branch_utility'][:, 0], valid),
            'ERC/utility_cont_mean': mean(packet['branch_utility'][:, 1], valid),
            'ERC/utility_cls_target_mean': mean(oracle['branch_gain'][:, 0], valid),
            'ERC/utility_cont_target_mean': mean(oracle['branch_gain'][:, 1], valid),
            'ERC/action_class_weight_keep': class_weight[0],
            'ERC/action_class_weight_cls': class_weight[1],
            'ERC/action_class_weight_cont': class_weight[2],
            'ERC/alpha_abs': mean(packet['alpha_scalar'].abs(), active),
            'ERC/selection_change_rate': mean(changed.float(), active),
            'ERC/changed_response_target_gain': mean(response_gain, changed),
            'ERC/changed_response_negative_rate': mean(
                response_gain.lt(-1e-6).float(), changed),
        })

        # Evaluation-only geometry diagnostic.  No value below contributes to
        # any target or loss term above.
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
        iou_gain = candidate_iou[row, full_index] - candidate_iou[row, base_index]
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

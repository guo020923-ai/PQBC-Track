"""Positive-anchored, deployment-aligned objective for TRC-v6b."""

import torch
from torch.nn import functional as F

from lib.train.actors.erc_trc_v4_loss import masked_average
from lib.train.actors.erc_trc_v6_loss import branch_response_utility_oracle
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy


def positive_anchored_action_losses(logits, target, valid, margin):
    """Balance present KEEP/C/T classes, skipping all-KEEP batches.

    All-KEEP batches are handled by explicit no-harm/risk objectives.  They do
    not repeatedly overwhelm the rare corrective actions in classification
    and ranking losses.
    """
    corrective = valid & target.ne(0)
    zero = logits.sum() * 0.0
    if not corrective.any():
        return zero, zero, logits.new_zeros(3), logits.new_zeros(())
    counts = torch.stack([
        (valid & target.eq(index)).sum() for index in range(3)
    ]).to(logits)
    present = counts > 0
    class_weight = torch.where(
        present, counts.clamp_min(1.0).reciprocal(), torch.zeros_like(counts)
    )
    per_frame = F.cross_entropy(logits, target, reduction='none')
    action = masked_average(per_frame, valid, class_weight[target])
    target_logit = logits.gather(1, target[:, None])[:, 0]
    alternatives = logits.clone()
    alternatives.scatter_(1, target[:, None], -1e4)
    per_frame_rank = F.softplus(
        float(margin) - target_logit + alternatives.max(-1).values
    )
    ranking = masked_average(
        per_frame_rank, valid, class_weight[target]
    )
    # Report normalized, human-readable class contributions.
    contribution = class_weight * counts
    normalized = contribution / contribution.sum().clamp_min(1e-12)
    return action, ranking, normalized, logits.new_ones(())


def positive_anchored_utility_loss(prediction, target, valid, minimum):
    """Give positive/non-positive frames equal weight for each seen branch."""
    losses, eligible = [], []
    for branch in range(2):
        positive = valid & (target[:, branch] > float(minimum))
        negative = valid & ~positive
        if not positive.any() or not negative.any():
            losses.append(prediction[:, branch].sum() * 0.0)
            eligible.append(prediction.new_zeros(()))
            continue
        error = F.smooth_l1_loss(
            prediction[:, branch], target[:, branch], reduction='none'
        )
        losses.append(0.5 * (error[positive].mean() + error[negative].mean()))
        eligible.append(prediction.new_ones(()))
    eligible_count = torch.stack(eligible).sum()
    if not bool(eligible_count):
        return prediction.sum() * 0.0, torch.stack(eligible)
    return torch.stack(losses).sum() / eligible_count, torch.stack(eligible)


def compute_trc_v6b_loss_and_metrics(pred, gt_heatmap, gt_bbox, cfg):
    packet = pred['erc_trc_output']
    minimum = cfg.TRAIN.ERC_TRC_V6B_MIN_TARGET_GAIN
    oracle = branch_response_utility_oracle(
        packet, gt_heatmap, cfg.TRAIN.ERC_TRC_V6B_ORACLE_STEPS, minimum,
    )
    valid = oracle['valid']
    target_action = oracle['target_action']
    corrective = valid & target_action.ne(0)
    preserve = valid & target_action.eq(0)

    action_loss, ranking_loss, class_weight, supervised_action = (
        positive_anchored_action_losses(
            packet['action_logits'], target_action, valid,
            cfg.TRAIN.ERC_TRC_V6B_ACTION_MARGIN,
        )
    )
    utility_loss, supervised_branch = positive_anchored_utility_loss(
        packet['branch_utility'], oracle['branch_gain'], valid, minimum,
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

    # This is exactly the straight-through action used by forward deployment,
    # with response evidence detached so ERC auxiliary loss cannot train Head.
    model_log = (
        packet['log_base'].detach()
        + packet['alpha'] * packet['evidence']['d'].detach()
    ).flatten(1)
    row = torch.arange(model_log.shape[0], device=model_log.device)
    target_value = model_log[row, oracle['target_index']]
    base_value = model_log[row, oracle['base_index']]
    temperature = max(
        float(cfg.TRAIN.ERC_TRC_V6B_DECISION_TEMPERATURE), 1e-6
    )
    decision_loss = masked_average(
        F.softplus((
            cfg.TRAIN.ERC_TRC_V6B_DECISION_MARGIN
            - target_value + base_value
        ) / temperature) * temperature,
        corrective,
    )

    maximum_utility = packet['branch_utility'].max(-1).values
    noharm_boundary = (
        float(packet['utility_threshold'])
        - cfg.TRAIN.ERC_TRC_V6B_NOHARM_MARGIN
    )
    noharm_loss = masked_average(
        F.softplus((maximum_utility - noharm_boundary) / temperature)
        * temperature,
        preserve,
    )
    branch_risk = F.relu(float(minimum) - oracle['branch_gain'])
    risk_loss = masked_average(
        (packet['action_probability'][:, 1:] * branch_risk).sum(-1), valid
    )

    loss = (
        cfg.TRAIN.ERC_TRC_V6B_ACTION_WEIGHT * action_loss
        + cfg.TRAIN.ERC_TRC_V6B_UTILITY_WEIGHT * utility_loss
        + cfg.TRAIN.ERC_TRC_V6B_RANK_WEIGHT * ranking_loss
        + cfg.TRAIN.ERC_TRC_V6B_STRENGTH_WEIGHT * strength_loss
        + cfg.TRAIN.ERC_TRC_V6B_DECISION_WEIGHT * decision_loss
        + cfg.TRAIN.ERC_TRC_V6B_NOHARM_WEIGHT * noharm_loss
        + cfg.TRAIN.ERC_TRC_V6B_RISK_WEIGHT * risk_loss
    )
    status = {
        'Loss/erc_trc_v6b_action': action_loss.detach(),
        'Loss/erc_trc_v6b_utility': utility_loss.detach(),
        'Loss/erc_trc_v6b_rank': ranking_loss.detach(),
        'Loss/erc_trc_v6b_strength': strength_loss.detach(),
        'Loss/erc_trc_v6b_decision': decision_loss.detach(),
        'Loss/erc_trc_v6b_noharm': noharm_loss.detach(),
        'Loss/erc_trc_v6b_risk': risk_loss.detach(),
        'Loss/erc_total': loss.detach(),
        'ERC/action_supervised_batch': supervised_action.detach(),
        'ERC/utility_cls_supervised_batch': supervised_branch[0].detach(),
        'ERC/utility_cont_supervised_batch': supervised_branch[1].detach(),
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

        # Detached IoU remains a diagnostic only; RQC is not implemented here.
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

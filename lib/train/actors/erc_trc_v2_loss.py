"""Decision-aligned response-only supervision for TRC-v2.

GT center heatmaps provide training supervision.  Candidate geometry/IoU is
used only in the detached diagnostic block, never by an ERC loss or inference.
All response tensors used as action targets are detached, so ERC auxiliary
losses cannot co-adapt the original Head.  The Head remains trainable through
the official UVLTrack objectives.
"""

import math

import torch
from torch.nn import functional as F

from lib.models.heads.target_reference_erc import (
    local_rank_reliability_targets,
)
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy


def masked_average(values, mask):
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def relative_preference_loss(packet, labels, target_temperature):
    """Supervise C-vs-T preference, not two unrelated absolute scores."""
    quality_difference = (
        labels['targets'][:, 0] - labels['targets'][:, 1]
    ).detach()
    target = torch.sigmoid(
        quality_difference / max(float(target_temperature), 1e-6)
    )
    logit = (
        packet['reliability_logits'][:, 0]
        - packet['reliability_logits'][:, 1]
    )
    point = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
    per_frame = (point * labels['point_weight']).flatten(1).sum(-1)
    valid_frame = labels['valid'].flatten(1).any(-1)
    return masked_average(per_frame, valid_frame), target, quality_difference


def topk_decision_losses(packet, relevance, topk, score_temperature):
    """Align the learned correction with the actual Hann-aware Top-K action."""
    evidence = packet['evidence']
    batch = relevance.shape[0]
    relevance = relevance.detach().flatten(1)
    support = packet['support'].detach().flatten(1)
    active = packet['active'].detach()

    # Critical isolation: base response and C/T disagreement are fixed targets
    # for the ERC loss.  Only alpha/reliability parameters receive gradients.
    base_log = packet['log_base'].detach().flatten(1)
    disagreement = evidence['d'].detach().flatten(1)
    alpha = packet['alpha'].flatten(1)
    calibrated_log = base_log + alpha * disagreement
    k = min(max(int(topk), 2), base_log.shape[-1])
    supported_base = base_log.masked_fill(~support, -torch.inf)
    indices = supported_base.topk(k, -1).indices
    base_top = base_log.gather(1, indices)
    calibrated_top = calibrated_log.gather(1, indices)
    target_top = relevance.gather(1, indices)

    target_max = target_top.max(-1).values
    target_min = target_top.min(-1).values
    valid = active & torch.isfinite(base_top).all(-1)
    valid &= (target_max - target_min) > 1e-6
    target_position = target_top.argmax(-1)
    baseline_is_best = target_position.eq(0)
    logits = calibrated_top / max(float(score_temperature), 1e-6)
    # Normalize by the uniform K-way entropy.  This makes the auxiliary loss
    # scale stable when K changes and prevents it from monopolizing the
    # repository's global gradient clipping budget.
    per_frame_ce = F.cross_entropy(
        logits, target_position, reduction='none'
    ) / math.log(k)
    decision_mask = valid & ~baseline_is_best
    preserve_mask = valid & baseline_is_best
    decision = masked_average(per_frame_ce, decision_mask)
    preserve = masked_average(per_frame_ce, preserve_mask)

    base_probability = F.softmax(
        base_top / max(float(score_temperature), 1e-6), -1
    ).detach()
    calibrated_probability = F.softmax(logits, -1)
    base_expected = (base_probability * target_top).sum(-1)
    calibrated_expected = (calibrated_probability * target_top).sum(-1)
    expected_gain = calibrated_expected - base_expected
    noharm = masked_average(F.relu(-expected_gain), valid)
    return {
        'decision': decision,
        'preserve': preserve,
        'noharm': noharm,
        'valid': valid,
        'corrective': decision_mask,
        'baseline_is_best': preserve_mask,
        'expected_gain': expected_gain,
        'indices': indices,
        'target_position': target_position,
    }


def compute_trc_v2_loss_and_metrics(pred, gt_heatmap, gt_bbox, cfg):
    packet = pred['erc_trc_output']
    evidence = packet['evidence']
    labels = local_rank_reliability_targets(
        evidence['log_c'], evidence['log_t'], gt_heatmap,
        packet['support'], packet['active'],
    )
    preference, preference_target, quality_difference = relative_preference_loss(
        packet, labels, cfg.TRAIN.ERC_TRC_V2_TARGET_TEMPERATURE
    )
    action = topk_decision_losses(
        packet, gt_heatmap, cfg.TRAIN.ERC_TRC_V2_TOPK,
        cfg.TRAIN.ERC_TRC_V2_SCORE_TEMPERATURE,
    )
    loss = (
        cfg.TRAIN.ERC_TRC_V2_PREFERENCE_WEIGHT * preference
        + cfg.TRAIN.ERC_TRC_V2_DECISION_WEIGHT * action['decision']
        + cfg.TRAIN.ERC_TRC_V2_PRESERVE_WEIGHT * action['preserve']
        + cfg.TRAIN.ERC_TRC_V2_NOHARM_WEIGHT * action['noharm']
    )
    status = {
        'Loss/erc_trc_v2_preference': preference.detach(),
        'Loss/erc_trc_v2_decision': action['decision'].detach(),
        'Loss/erc_trc_v2_preserve': action['preserve'].detach(),
        'Loss/erc_trc_v2_noharm': action['noharm'].detach(),
        'Loss/erc_total': loss.detach(),
    }

    with torch.no_grad():
        active = packet['active']
        active_locations = active[:, None, None] & packet['support']
        zero = loss.detach().new_zeros(())

        def mean(value, mask):
            return value[mask].mean() if mask.any() else zero

        def std(value, mask):
            return value[mask].std(unbiased=False) if mask.any() else zero

        r_cls = packet['reliability'][:, 0]
        r_cont = packet['reliability'][:, 1]
        alpha = packet['alpha']
        base = packet['base'].flatten(1)
        score = packet['score'].flatten(1)
        base_index = base.argmax(-1)
        full_index = score.argmax(-1)
        changed = (base_index != full_index) & active
        batch_index = torch.arange(base.shape[0], device=base.device)
        target = preference_target
        predicted_preference = r_cls
        preference_valid = labels['valid']
        preference_distinct = preference_valid & (
            quality_difference.abs() > 1e-6)
        heat = gt_heatmap.detach().flatten(1)
        response_gain = (
            heat[batch_index, full_index] - heat[batch_index, base_index]
        )
        status.update({
            'ERC/active_fraction': active.float().mean(),
            'ERC/alpha_abs': mean(alpha.abs(), active_locations),
            'ERC/alpha_mean': mean(alpha, active_locations),
            'ERC/alpha_std': std(alpha, active_locations),
            'ERC/reliability_cls_mean': mean(r_cls, active_locations),
            'ERC/reliability_cont_mean': mean(r_cont, active_locations),
            'ERC/reliability_sum_error': mean(
                (r_cls + r_cont - 1.0).abs(), active_locations),
            'ERC/preference_mae': mean(
                (predicted_preference - target).abs(), preference_valid),
            'ERC/preference_sign_accuracy': mean(
                ((r_cls - r_cont).sign() == quality_difference.sign()).float(),
                preference_distinct),
            'ERC/topk_valid_fraction': action['valid'].float().mean(),
            'ERC/topk_corrective_fraction': action['corrective'].float().mean(),
            'ERC/topk_preserve_fraction': action['baseline_is_best'].float().mean(),
            'ERC/topk_expected_response_gain': mean(
                action['expected_gain'], action['valid']),
            'ERC/selection_change_rate': mean(changed.float(), active),
            'ERC/changed_response_target_gain': mean(response_gain, changed),
            'ERC/changed_response_negative_rate': mean(
                (response_gain < -1e-6).float(), changed),
        })

        # Detached geometry diagnostics only.  This block cannot contribute to
        # any loss and does not define an ERC target.
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
            'ERC/changed_candidate_count': (
                changed & valid_box).sum().float(),
            'ERC/changed_candidate_negative_rate': mean(
                (iou_gain < -1e-6).float(), changed & valid_box),
        })
    return loss, status

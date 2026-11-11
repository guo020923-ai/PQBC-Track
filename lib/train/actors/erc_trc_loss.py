"""TRC response-only loss and detached diagnostics (IoU is metrics ONLY)."""

import torch

from lib.models.heads.target_reference_erc import local_rank_reliability_targets, trc_losses
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy


def compute_trc_loss_and_metrics(pred, gt_heatmap, gt_bbox, cfg):
    packet = pred['erc_trc_output']
    evidence = packet['evidence']
    labels = local_rank_reliability_targets(
        evidence['log_c'], evidence['log_t'], gt_heatmap,
        packet['support'], packet['active'])
    losses = trc_losses(packet, labels)
    loss = (cfg.TRAIN.ERC_TRC_RELIABILITY_WEIGHT * losses['reliability']
            + cfg.TRAIN.ERC_TRC_FUSION_WEIGHT * losses['fusion'])
    status = {'Loss/erc_trc_reliability': losses['reliability'].detach(),
              'Loss/erc_trc_fusion': losses['fusion'].detach(),
              'Loss/erc_total': loss.detach()}
    # This block must never feed labels, decisions or differentiable loss.
    with torch.no_grad():
        active = packet['active']
        active_locations = active[:, None, None] & packet['support']
        zero = loss.detach().new_zeros(())

        def mean(value, mask):
            return value[mask].mean() if mask.any() else zero

        def std(value, mask):
            return value[mask].std(unbiased=False) if mask.any() else zero

        r_c, r_t = packet['reliability'][:, 0], packet['reliability'][:, 1]
        alpha = packet['alpha']
        base, score = packet['base'].flatten(1), packet['score'].flatten(1)
        base_i, full_i = base.argmax(-1), score.argmax(-1)
        changed = (base_i != full_i) & active
        bid = torch.arange(base.shape[0], device=base.device)
        top = base.topk(min(16, base.shape[-1]), -1).indices
        d_clip = evidence['d_unclipped'].abs() >= cfg.MODEL.HEAD.ERC.TRC_D_MAX
        top_clip = d_clip.flatten(1).gather(1, top).float().mean(-1)
        valid = labels['valid']
        target_c, target_t = labels['targets'][:, 0], labels['targets'][:, 1]
        relative_target = target_c - target_t
        distinct = valid & (relative_target.abs() > 1e-6)
        status.update({
            'ERC/active_fraction': active.float().mean(),
            'ERC/alpha_abs': mean(alpha.abs(), active_locations),
            'ERC/alpha_mean': mean(alpha, active_locations),
            'ERC/alpha_std': std(alpha, active_locations),
            'ERC/alpha_saturation': mean((alpha.abs() >= .95 * packet['alpha_limit']).float(), active_locations),
            'ERC/reliability_cls_mean': mean(r_c, active_locations),
            'ERC/reliability_cont_mean': mean(r_t, active_locations),
            'ERC/reliability_mean': mean((r_c + r_t) / 2, active_locations),
            'ERC/reliability_std': std((r_c + r_t) / 2, active_locations),
            'ERC/reliability_cls_mae': mean((r_c - target_c).abs(), valid),
            'ERC/reliability_cont_mae': mean((r_t - target_t).abs(), valid),
            'ERC/relative_reliability_sign_accuracy': mean(
                ((r_c - r_t).sign() == relative_target.sign()).float(), distinct),
            'ERC/reliability_label_valid_fraction': valid.float().mean(),
            'ERC/selection_change_rate': mean(changed.float(), active),
            'ERC/d_clip_rate': mean(d_clip.float(), active_locations),
            'ERC/d_top16_clip_rate': mean(top_clip, active),
        })
        for branch, values, targets in (('cls', r_c, target_c), ('cont', r_t, target_t)):
            if valid.any():
                a, b = values[valid], targets[valid]
                a, b = a - a.mean(), b - b.mean()
                correlation = (a * b).mean() / (a.square().mean() * b.square().mean()).sqrt().clamp_min(1e-8)
            else:
                correlation = zero
            status['ERC/reliability_%s_correlation' % branch] = correlation

        # Geometry is used only AFTER the response decisions, with no gradients.
        boxes = box_cxcywh_to_xyxy(pred['bbox_map'].detach()).clamp(0, 1)
        gt = box_xywh_to_xyxy(gt_bbox.detach()).clamp(0, 1)[:, None]
        intersection_wh = (torch.minimum(boxes[..., 2:], gt[..., 2:])
                           - torch.maximum(boxes[..., :2], gt[..., :2])).clamp_min(0)
        intersection = intersection_wh.prod(-1)
        union = ((boxes[..., 2:] - boxes[..., :2]).clamp_min(0).prod(-1)
                 + (gt[..., 2:] - gt[..., :2]).clamp_min(0).prod(-1) - intersection)
        iou = intersection / union.clamp_min(1e-6)
        iou_gain = iou[bid, full_i] - iou[bid, base_i]
        valid_box = active & torch.isfinite(gt_bbox).all(-1) & (gt_bbox[:, 2:] > 0).all(-1)
        status.update({
            'ERC/candidate_iou_gain': mean(iou_gain, valid_box),
            'ERC/changed_candidate_iou_gain': mean(iou_gain, changed & valid_box),
            'ERC/changed_candidate_count': (changed & valid_box).sum().float(),
            'ERC/changed_candidate_negative_rate': mean((iou_gain < -1e-6).float(), changed & valid_box),
        })
    return loss, status

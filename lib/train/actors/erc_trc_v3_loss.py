"""Full-map three-action supervision for decision-selective TRC-v3.

The loss uses only response maps and the official GT centre heatmap.  Bbox/IoU
is detached diagnostic information and never defines an ERC target.  Native
Head responses are detached from every ERC auxiliary term.
"""

import torch
from torch.nn import functional as F

from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy


def masked_average(values, mask):
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _response_quality(log_score, relevance, support, temperature):
    log_score = log_score.masked_fill(~support, -1e4)
    probability = F.softmax(log_score / max(float(temperature), 1e-6), -1)
    return (probability * relevance).sum(-1)


@torch.no_grad()
def full_map_action_oracle(packet, relevance, steps, score_temperature,
                           target_temperature):
    """Reachable preserve/C/T targets over the complete Hann-aware map."""
    relevance = relevance.detach().flatten(1)
    base_log = packet['log_base'].detach().flatten(1)
    difference = packet['evidence']['d'].detach().flatten(1)
    support = packet['support'].detach().flatten(1)
    active = packet['active'].detach()
    count = max(int(steps), 2)
    # alpha=0 belongs exclusively to KEEP.  Allowing it inside the C/T
    # families would let both intervention actions masquerade as preservation
    # and make their risk identically zero.
    magnitude = torch.linspace(
        float(packet['alpha_limit']) / count,
        float(packet['alpha_limit']), count,
        device=base_log.device, dtype=base_log.dtype,
    )
    base = base_log.masked_fill(~support, -1e4)
    cls_scores = base[:, None] + magnitude[None, :, None] * difference[:, None]
    cont_scores = base[:, None] - magnitude[None, :, None] * difference[:, None]
    temperature = max(float(score_temperature), 1e-6)
    cls_probability = F.softmax(cls_scores / temperature, -1)
    cont_probability = F.softmax(cont_scores / temperature, -1)
    cls_soft = (cls_probability * relevance[:, None]).sum(-1)
    cont_soft = (cont_probability * relevance[:, None]).sum(-1)
    cls_hard = relevance.gather(1, cls_scores.argmax(-1))
    cont_hard = relevance.gather(1, cont_scores.argmax(-1))
    cls_index = cls_hard.argmax(-1)
    cont_index = cont_hard.argmax(-1)
    batch_index = torch.arange(base.shape[0], device=base.device)
    keep_hard = relevance[batch_index, base.argmax(-1)]
    keep_soft = _response_quality(base, relevance, support, temperature)
    action_hard = torch.stack([
        keep_hard,
        cls_hard[batch_index, cls_index],
        cont_hard[batch_index, cont_index],
    ], -1)
    action_soft = torch.stack([
        keep_soft,
        cls_soft[batch_index, cls_index],
        cont_soft[batch_index, cont_index],
    ], -1)
    # Hard decision quality is primary.  Soft quality breaks exact hard ties
    # without imposing a hand-written intervention threshold.
    utility = action_hard + 0.05 * action_soft
    target = F.softmax(
        utility / max(float(target_temperature), 1e-6), -1
    )
    valid = active & support.any(-1) & (
        relevance.max(-1).values - relevance.min(-1).values > 1e-6
    )
    best = utility.argmax(-1)
    best_quality = action_soft.max(-1).values
    return {
        'target': target,
        'utility': utility,
        'hard_quality': action_hard,
        'soft_quality': action_soft,
        'best_action': best,
        'best_quality': best_quality,
        'valid': valid,
        'cls_magnitude': magnitude[cls_index],
        'cont_magnitude': magnitude[cont_index],
    }


def compute_trc_v3_loss_and_metrics(pred, gt_heatmap, gt_bbox, cfg):
    packet = pred['erc_trc_output']
    relevance = gt_heatmap.detach().flatten(1)
    oracle = full_map_action_oracle(
        packet, gt_heatmap, cfg.TRAIN.ERC_TRC_V3_ORACLE_STEPS,
        cfg.TRAIN.ERC_TRC_V3_SCORE_TEMPERATURE,
        cfg.TRAIN.ERC_TRC_V3_TARGET_TEMPERATURE,
    )
    valid = oracle['valid']
    action_probability = packet['action_probability']
    action_kl = F.kl_div(
        action_probability.clamp_min(1e-8).log(), oracle['target'],
        reduction='none',
    ).sum(-1)
    action_loss = masked_average(action_kl, valid)

    base_log = packet['log_base'].detach().flatten(1)
    model_log = (
        base_log
        + packet['alpha'].flatten(1)
        * packet['evidence']['d'].detach().flatten(1)
    )
    support = packet['support'].detach().flatten(1)
    base_quality = _response_quality(
        base_log, relevance, support,
        cfg.TRAIN.ERC_TRC_V3_SCORE_TEMPERATURE,
    )
    model_quality = _response_quality(
        model_log, relevance, support,
        cfg.TRAIN.ERC_TRC_V3_SCORE_TEMPERATURE,
    )
    decision = masked_average(
        F.relu(oracle['best_quality'] - model_quality), valid
    )
    noharm = masked_average(F.relu(base_quality - model_quality), valid)
    action_risk = F.relu(
        oracle['soft_quality'][:, :1] - oracle['soft_quality']
    )
    risk = masked_average((action_probability * action_risk).sum(-1), valid)
    gate = packet['gate']
    gate_tv = (
        (gate[:, 1:] - gate[:, :-1]).abs().mean()
        + (gate[:, :, 1:] - gate[:, :, :-1]).abs().mean()
    )
    loss = (
        cfg.TRAIN.ERC_TRC_V3_ACTION_WEIGHT * action_loss
        + cfg.TRAIN.ERC_TRC_V3_DECISION_WEIGHT * decision
        + cfg.TRAIN.ERC_TRC_V3_NOHARM_WEIGHT * noharm
        + cfg.TRAIN.ERC_TRC_V3_RISK_WEIGHT * risk
        + cfg.TRAIN.ERC_TRC_V3_GATE_TV_WEIGHT * gate_tv
    )
    status = {
        'Loss/erc_trc_v3_action': action_loss.detach(),
        'Loss/erc_trc_v3_decision': decision.detach(),
        'Loss/erc_trc_v3_noharm': noharm.detach(),
        'Loss/erc_trc_v3_risk': risk.detach(),
        'Loss/erc_trc_v3_gate_tv': gate_tv.detach(),
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
        response_gain = (
            relevance[batch_index, full_index]
            - relevance[batch_index, base_index]
        )
        predicted_action = action_probability.argmax(-1)
        target_action = oracle['best_action']
        advantage = (
            oracle['soft_quality'].max(-1).values
            - oracle['soft_quality'][:, 0]
        )
        status.update({
            'ERC/active_fraction': active.float().mean(),
            'ERC/alpha_abs': mean(packet['alpha'].abs(), active[:, None, None].expand_as(packet['alpha'])),
            'ERC/alpha_mean': mean(packet['alpha'], active[:, None, None].expand_as(packet['alpha'])),
            'ERC/alpha_std': std(packet['alpha'], active[:, None, None].expand_as(packet['alpha'])),
            'ERC/action_keep_mean': mean(action_probability[:, 0], active),
            'ERC/action_cls_mean': mean(action_probability[:, 1], active),
            'ERC/action_cont_mean': mean(action_probability[:, 2], active),
            'ERC/action_accuracy': mean(predicted_action.eq(target_action).float(), valid),
            'ERC/oracle_keep_fraction': mean(target_action.eq(0).float(), valid),
            'ERC/oracle_cls_fraction': mean(target_action.eq(1).float(), valid),
            'ERC/oracle_cont_fraction': mean(target_action.eq(2).float(), valid),
            'ERC/oracle_response_advantage': mean(advantage, valid),
            'ERC/beta_abs': mean(packet['beta'].abs(), active),
            'ERC/beta_mean': mean(packet['beta'], active),
            'ERC/gate_mean': mean(packet['gate'], active[:, None, None].expand_as(packet['gate'])),
            'ERC/gate_std': std(packet['gate'], active[:, None, None].expand_as(packet['gate'])),
            'ERC/selection_change_rate': mean(changed.float(), active),
            'ERC/changed_response_target_gain': mean(response_gain, changed),
            'ERC/changed_response_negative_rate': mean((response_gain < -1e-6).float(), changed),
        })

        # Detached geometry diagnostics only; never included in loss/targets.
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
                (iou_gain < -1e-6).float(), changed & valid_box),
        })
    return loss, status

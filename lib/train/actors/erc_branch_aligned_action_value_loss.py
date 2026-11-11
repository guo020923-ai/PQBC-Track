"""Dense action-utility supervision for response-level ERC-AV1.

All candidate boxes and IoU values are detached and are used only to build
training targets.  ERC-AV1 itself receives response/reference evidence only
and can only select a C/T response-calibration action.
"""

import torch
from torch.nn import functional as F

from .erc_branch_aligned_iou_utility_loss import _candidate_iou_map
from .erc_branch_aligned_tristate_multiaction_loss import _masked_mean


@torch.no_grad()
def build_action_utility_targets(pred, gt_heatmap, gt_bbox, cfg):
    """Return signed utility targets for every reachable complete action."""
    log_base = pred['erc_av1_log_base'].detach().flatten(1)
    corrections = pred['erc_av1_action_correction_maps'].detach().flatten(2)
    support = pred['erc_av1_support_mask'].detach().flatten(1).bool()
    candidate_iou = _candidate_iou_map(pred['bbox_map'], gt_bbox)
    response_target = gt_heatmap.detach().flatten(1).clamp_min(0.0)
    active = pred['erc_active_mask'].detach().reshape(-1).bool()
    valid_bbox = (
        torch.isfinite(gt_bbox).all(-1)
        & (gt_bbox[:, 2:] > 0.0).all(-1)
    )
    valid = (
        active & support.any(-1) & valid_bbox
        & (response_target * support.to(response_target)).max(-1).values.gt(0.0)
        & torch.isfinite(candidate_iou).all(-1)
    )

    scores = (log_base[:, None] + corrections).masked_fill(
        ~support[:, None], torch.finfo(log_base.dtype).min)
    hard_index = scores.argmax(-1)
    hard_iou = candidate_iou[:, None].expand_as(scores).gather(
        -1, hard_index[..., None]).squeeze(-1)
    hard_iou_gain = hard_iou - hard_iou[:, :1]

    topk = min(int(cfg.TRAIN.ERC_AV1_ORACLE_TOPK), scores.shape[-1])
    top_values, top_indices = scores.topk(topk, dim=-1)
    temperature = max(
        float(cfg.TRAIN.ERC_AV1_ORACLE_TEMPERATURE), 1e-6)
    top_probability = F.softmax(top_values / temperature, dim=-1)
    top_iou = candidate_iou[:, None].expand_as(scores).gather(
        -1, top_indices)
    soft_iou = (top_probability * top_iou).sum(-1)
    soft_iou_gain = soft_iou - soft_iou[:, :1]

    top_response = response_target[:, None].expand_as(scores).gather(
        -1, top_indices)
    soft_response = (top_probability * top_response).sum(-1)
    soft_response_gain = soft_response - soft_response[:, :1]

    hard_weight = float(cfg.TRAIN.ERC_AV1_HARD_UTILITY_WEIGHT)
    soft_weight = float(cfg.TRAIN.ERC_AV1_SOFT_UTILITY_WEIGHT)
    total_weight = max(hard_weight + soft_weight, 1e-6)
    estimated_gain = (
        hard_weight * hard_iou_gain + soft_weight * soft_iou_gain
    ) / total_weight

    magnitude_index = pred['erc_av1_action_magnitude_index'].detach().long()
    magnitudes = pred['erc_av1_magnitudes'].detach().to(estimated_gain)
    action_magnitude = estimated_gain.new_zeros(estimated_gain.shape[-1])
    nonkeep = magnitude_index.ge(0)
    action_magnitude[nonkeep] = magnitudes[magnitude_index[nonkeep]]
    magnitude_scale = action_magnitude / action_magnitude.max().clamp_min(1e-6)
    minimum_gain = float(cfg.TRAIN.ERC_AV1_MIN_ACTION_GAIN)
    # KEEP has fixed zero utility.  Every non-KEEP target is its estimated
    # candidate-IoU gain minus both action cost and the minimum worthwhile
    # gain.  Therefore target utility > 0 means exactly "intervene", matching
    # the deployment rule and avoiding contradictory KEEP/positive targets.
    raw_utility = estimated_gain - (
        float(cfg.TRAIN.ERC_AV1_ACTION_COST) * magnitude_scale[None])
    raw_utility[:, 1:] = raw_utility[:, 1:] - minimum_gain
    raw_utility[:, 0] = 0.0
    raw_utility = torch.where(
        valid[:, None], raw_utility, torch.zeros_like(raw_utility))

    utility_scale = max(float(cfg.TRAIN.ERC_AV1_UTILITY_SCALE), 1e-6)
    utility_target = (raw_utility / utility_scale).clamp(-1.0, 1.0)
    best_utility, best_action = raw_utility.max(-1)
    actionable = valid & best_utility.gt(0.0)
    target_action = torch.where(
        actionable, best_action, torch.zeros_like(best_action))

    rows = torch.arange(scores.shape[0], device=scores.device)
    return {
        'valid': valid,
        'actionable': actionable,
        'target_action': target_action,
        'raw_utility': raw_utility,
        'utility_target': utility_target,
        'scores': scores,
        'hard_index': hard_index,
        'hard_iou': hard_iou,
        'hard_iou_gain': hard_iou_gain,
        'soft_iou': soft_iou,
        'soft_iou_gain': soft_iou_gain,
        'soft_response': soft_response,
        'soft_response_gain': soft_response_gain,
        'candidate_iou': candidate_iou,
        'target_hard_iou_gain': hard_iou_gain[rows, target_action],
        'target_soft_iou_gain': soft_iou_gain[rows, target_action],
        'target_utility': raw_utility[rows, target_action],
    }


def _balanced_binary_mean(values, positive, negative, zero):
    positive_mean = _masked_mean(values, positive, zero)
    negative_mean = _masked_mean(values, negative, zero)
    if positive.any() and negative.any():
        return 0.5 * (positive_mean + negative_mean)
    if positive.any():
        return positive_mean
    if negative.any():
        return negative_mean
    return zero


def compute_action_value_loss_and_metrics(pred, gt_heatmap, gt_bbox, cfg):
    predicted_utility = pred['erc_av1_action_utility']
    zero = predicted_utility.sum() * 0.0
    oracle = build_action_utility_targets(
        pred, gt_heatmap, gt_bbox, cfg)
    valid = oracle['valid']
    if not valid.any():
        return zero, {
            'Loss/erc_total': zero.detach(),
            'ERC/av1_valid_fraction': valid.float().mean(),
        }

    target = oracle['utility_target']
    positive_action = target.gt(0.0)
    harmful_action = target.lt(0.0)
    action_weight = torch.ones_like(target)
    action_weight = torch.where(
        positive_action,
        action_weight.new_full(action_weight.shape, float(
            cfg.TRAIN.ERC_AV1_POSITIVE_UTILITY_WEIGHT)),
        action_weight,
    )
    action_weight = torch.where(
        harmful_action,
        action_weight.new_full(action_weight.shape, float(
            cfg.TRAIN.ERC_AV1_HARMFUL_UTILITY_WEIGHT)),
        action_weight,
    )
    utility_error = F.smooth_l1_loss(
        predicted_utility, target, reduction='none')
    valid_actions = valid[:, None].expand_as(target).clone()
    # KEEP utility is fixed by design, so it must not dilute the regression
    # denominator even though it remains part of listwise/ranking decisions.
    valid_actions[:, 0] = False
    utility_loss = (
        utility_error * action_weight * valid_actions
    ).sum() / (action_weight * valid_actions).sum().clamp_min(1.0)

    target_action = oracle['target_action']
    actionable = oracle['actionable']
    preserve = valid & ~actionable
    listwise_temperature = max(
        float(cfg.TRAIN.ERC_AV1_LISTWISE_TEMPERATURE), 1e-6)
    listwise_frame = F.cross_entropy(
        predicted_utility / listwise_temperature,
        target_action,
        reduction='none',
    ) * listwise_temperature
    listwise_loss = _balanced_binary_mean(
        listwise_frame, actionable, preserve, zero)

    rows = torch.arange(predicted_utility.shape[0], device=predicted_utility.device)
    target_prediction = predicted_utility[rows, target_action]
    alternatives = predicted_utility.clone()
    alternatives.scatter_(1, target_action[:, None], -1e4)
    strongest_alternative = alternatives.max(-1).values
    rank_margin = float(cfg.TRAIN.ERC_AV1_RANK_MARGIN)
    rank_frame = F.softplus(
        rank_margin - target_prediction + strongest_alternative)
    rank_loss = _masked_mean(rank_frame, actionable, zero)

    strongest_nonkeep = predicted_utility[:, 1:].max(-1).values
    keep_margin = float(cfg.TRAIN.ERC_AV1_KEEP_MARGIN)
    keep_temperature = max(
        float(cfg.TRAIN.ERC_AV1_KEEP_TEMPERATURE), 1e-6)
    keep_frame = F.softplus(
        (strongest_nonkeep + keep_margin) / keep_temperature
    ) * keep_temperature
    keep_loss = _masked_mean(keep_frame, preserve, zero)

    noharm_margin = float(cfg.TRAIN.ERC_AV1_NOHARM_MARGIN)
    noharm_frame = F.softplus(
        (predicted_utility + noharm_margin) / keep_temperature
    ) * keep_temperature
    noharm_mask = valid_actions & harmful_action
    noharm_loss = (
        noharm_frame * noharm_mask
    ).sum() / noharm_mask.sum().clamp_min(1)

    policy_temperature = max(
        float(cfg.TRAIN.ERC_AV1_POLICY_TEMPERATURE), 1e-6)
    policy = F.softmax(predicted_utility / policy_temperature, dim=-1)
    expected_iou = (policy * oracle['soft_iou'].detach()).sum(-1)
    target_iou = oracle['soft_iou'][rows, target_action].detach()
    decision_loss = _masked_mean(
        F.relu(target_iou - expected_iou), valid, zero)

    expected_response = (
        policy * oracle['soft_response'].detach()).sum(-1)
    target_response = oracle['soft_response'][
        rows, target_action].detach()
    response_aux_loss = _masked_mean(
        F.relu(target_response - expected_response), actionable, zero)

    total = (
        float(cfg.TRAIN.ERC_AV1_UTILITY_WEIGHT) * utility_loss
        + float(cfg.TRAIN.ERC_AV1_LISTWISE_WEIGHT) * listwise_loss
        + float(cfg.TRAIN.ERC_AV1_RANK_WEIGHT) * rank_loss
        + float(cfg.TRAIN.ERC_AV1_KEEP_WEIGHT) * keep_loss
        + float(cfg.TRAIN.ERC_AV1_DECISION_WEIGHT) * decision_loss
        + float(cfg.TRAIN.ERC_AV1_NOHARM_WEIGHT) * noharm_loss
        + float(cfg.TRAIN.ERC_AV1_RESPONSE_AUX_WEIGHT) * response_aux_loss
    )

    with torch.no_grad():
        raw_action = pred['erc_av1_raw_hard_action']
        predicted_intervention = raw_action.ne(0) & valid
        base_score = pred['erc_windowed_base_score'].detach().flatten(1)
        full_score = pred['erc_windowed_score'].detach().flatten(1)
        base_index = base_score.argmax(-1)
        full_index = full_score.argmax(-1)
        changed = valid & full_index.ne(base_index)
        candidate_iou = oracle['candidate_iou']
        baseline_iou = candidate_iou[rows, base_index]
        full_iou = candidate_iou[rows, full_index]
        candidate_gain = full_iou - baseline_iou
        beneficial = changed & candidate_gain.gt(1e-7)
        harmful = changed & candidate_gain.lt(-1e-7)
        true_positive = predicted_intervention & actionable
        false_positive = predicted_intervention & ~actionable

        def mean(values, mask=valid):
            return _masked_mean(values, mask, zero.detach())

        intervention_count = predicted_intervention.sum().clamp_min(1)
        actionable_count = actionable.sum().clamp_min(1)
        preserve_count = preserve.sum().clamp_min(1)
        oracle_gain = oracle['target_hard_iou_gain']
        positive_oracle_gain = mean(oracle_gain, actionable)
        learned_gain = mean(candidate_gain, valid)
        capture_ratio = learned_gain / positive_oracle_gain.clamp_min(1e-6)
        status = {
            'Loss/erc_total': total.detach(),
            'Loss/erc_av1_utility': utility_loss.detach(),
            'Loss/erc_av1_listwise': listwise_loss.detach(),
            'Loss/erc_av1_rank': rank_loss.detach(),
            'Loss/erc_av1_keep': keep_loss.detach(),
            'Loss/erc_av1_decision': decision_loss.detach(),
            'Loss/erc_av1_noharm': noharm_loss.detach(),
            'Loss/erc_av1_response_aux': response_aux_loss.detach(),
            'ERC/av1_valid_fraction': valid.float().mean(),
            'ERC/av1_oracle_actionable_rate': mean(actionable.float()),
            'ERC/av1_pred_intervention_rate': mean(
                predicted_intervention.float()),
            'ERC/av1_intervention_precision': (
                true_positive.sum().float() / intervention_count),
            'ERC/av1_intervention_recall': (
                true_positive.sum().float() / actionable_count),
            'ERC/av1_false_positive_rate': (
                false_positive.sum().float() / preserve_count),
            'ERC/av1_action_accuracy': mean(
                raw_action.eq(target_action).float()),
            'ERC/av1_best_utility_mean': mean(
                pred['erc_av1_best_utility'].detach()),
            'ERC/av1_target_utility_mean': mean(
                oracle['target_utility']),
            'ERC/av1_gate_mean': mean(
                pred['erc_av1_utility_gate'].detach()),
            'ERC/av1_gate_saturation_rate': mean(
                pred['erc_av1_utility_gate'].detach().ge(1.0).float()),
            'ERC/av1_selection_change_rate': mean(changed.float()),
            'ERC/av1_candidate_iou_gain': mean(candidate_gain),
            'ERC/av1_changed_candidate_iou_gain': mean(
                candidate_gain, changed),
            'ERC/av1_beneficial_change_rate': mean(beneficial.float()),
            'ERC/av1_harmful_change_rate': mean(harmful.float()),
            'ERC/av1_oracle_hard_iou_gain': mean(oracle_gain),
            'ERC/av1_oracle_positive_iou_gain': positive_oracle_gain,
            'ERC/av1_oracle_soft_iou_gain': mean(
                oracle['target_soft_iou_gain']),
            'ERC/av1_oracle_capture_ratio': capture_ratio,
        }
    return total, status

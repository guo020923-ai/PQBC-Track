"""Anti-collapse supervision for the unified ERC-AV1b utility head.

The model still predicts one signed utility for every complete non-KEEP
action and uses ``max utility > 0`` at deployment.  Training is decomposed on
that same vector into:

1. balanced intervention permission (best non-KEEP utility versus zero),
2. conditional action choice only on actionable frames, and
3. dense signed utility regression with positive/harmful action balancing.

No additional prediction head is introduced.  Candidate boxes and IoU remain
detached target-construction signals and never become ERC inputs.
"""

import torch
from torch.nn import functional as F

from .erc_branch_aligned_action_value_loss import (
    _balanced_binary_mean,
    build_action_utility_targets,
)
from .erc_branch_aligned_tristate_multiaction_loss import _masked_mean


def _weighted_available_mean(terms, zero):
    """Average weighted terms without diluting a missing target group."""
    numerator = zero
    denominator = 0.0
    for value, weight, available in terms:
        if available and float(weight) > 0.0:
            numerator = numerator + float(weight) * value
            denominator += float(weight)
    if denominator == 0.0:
        return zero
    return numerator / denominator


def _balanced_permission_margin_loss(
        strongest_nonkeep, actionable, preserve, margin, temperature, zero):
    """Push actionable maxima above zero and preserve maxima below zero."""
    positive = F.softplus(
        (float(margin) - strongest_nonkeep) / float(temperature)
    ) * float(temperature)
    negative = F.softplus(
        (float(margin) + strongest_nonkeep) / float(temperature)
    ) * float(temperature)
    frame = torch.where(actionable, positive, negative)
    return _balanced_binary_mean(frame, actionable, preserve, zero)


def compute_balanced_action_value_loss_and_metrics(
        pred, gt_heatmap, gt_bbox, cfg, oracle=None):
    """Train unified utilities without allowing sparse KEEP to dominate."""
    predicted_utility = pred['erc_av1_action_utility']
    nonkeep_utility = predicted_utility[:, 1:]
    zero = predicted_utility.sum() * 0.0
    if oracle is None:
        oracle = build_action_utility_targets(
            pred, gt_heatmap, gt_bbox, cfg)
    valid = oracle['valid']
    if not valid.any():
        return zero, {
            'Loss/erc_total': zero.detach(),
            'ERC/av1_valid_fraction': valid.float().mean(),
        }

    actionable = oracle['actionable']
    preserve = valid & ~actionable
    target_action = oracle['target_action']
    target = oracle['utility_target']
    rows = torch.arange(
        predicted_utility.shape[0], device=predicted_utility.device)

    # Dense utility regression is balanced by semantic target group instead
    # of by the raw number of action cells.  With 15 non-KEEP actions, a plain
    # elementwise mean otherwise rewards the trivial all-negative solution.
    valid_actions = valid[:, None].expand_as(target).clone()
    valid_actions[:, 0] = False
    positive_action = valid_actions & target.gt(0.0)
    harmful_action = valid_actions & target.lt(0.0)
    neutral_action = valid_actions & ~(positive_action | harmful_action)
    utility_error = F.smooth_l1_loss(
        predicted_utility, target, reduction='none')
    positive_utility_loss = _masked_mean(
        utility_error, positive_action, zero)
    harmful_utility_loss = _masked_mean(
        utility_error, harmful_action, zero)
    neutral_utility_loss = _masked_mean(
        utility_error, neutral_action, zero)
    utility_loss = _weighted_available_mean([
        (positive_utility_loss,
         cfg.TRAIN.ERC_AV1_POSITIVE_UTILITY_WEIGHT,
         bool(positive_action.any())),
        (harmful_utility_loss,
         cfg.TRAIN.ERC_AV1_HARMFUL_UTILITY_WEIGHT,
         bool(harmful_action.any())),
        (neutral_utility_loss, 1.0, bool(neutral_action.any())),
    ], zero)

    # This is not a second permission head.  It directly constrains the exact
    # deployment statistic max(non-KEEP utility) around the zero KEEP utility.
    strongest_nonkeep = nonkeep_utility.max(-1).values
    permission_margin = float(cfg.TRAIN.ERC_AV1B_PERMISSION_MARGIN)
    permission_temperature = max(
        float(cfg.TRAIN.ERC_AV1B_PERMISSION_TEMPERATURE), 1e-6)
    permission_loss = _balanced_permission_margin_loss(
        strongest_nonkeep, actionable, preserve,
        permission_margin, permission_temperature, zero)

    # Once a frame is known to be actionable, classify only among the 15
    # deployable correction actions.  KEEP is handled by the sign constraint
    # above and no longer overwhelms fine-grained action learning.
    choice_temperature = max(
        float(cfg.TRAIN.ERC_AV1B_ACTION_TEMPERATURE), 1e-6)
    action_choice_frame = F.cross_entropy(
        nonkeep_utility / choice_temperature,
        (target_action - 1).clamp_min(0), reduction='none',
    ) * choice_temperature
    action_choice_loss = _masked_mean(
        action_choice_frame, actionable, zero)

    target_prediction = predicted_utility[rows, target_action]
    alternatives = predicted_utility.clone()
    alternatives.scatter_(1, target_action[:, None], -1e4)
    strongest_alternative = alternatives.max(-1).values
    rank_margin = float(cfg.TRAIN.ERC_AV1_RANK_MARGIN)
    rank_temperature = max(
        float(cfg.TRAIN.ERC_AV1B_RANK_TEMPERATURE), 1e-6)
    rank_frame = F.softplus((
        rank_margin - target_prediction + strongest_alternative
    ) / rank_temperature) * rank_temperature
    rank_loss = _masked_mean(rank_frame, actionable, zero)

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
    noharm_loss = _masked_mean(
        noharm_frame, harmful_action, zero)

    policy_temperature = max(
        float(cfg.TRAIN.ERC_AV1_POLICY_TEMPERATURE), 1e-6)
    policy = F.softmax(predicted_utility / policy_temperature, dim=-1)
    expected_iou = (policy * oracle['soft_iou'].detach()).sum(-1)
    target_iou = oracle['soft_iou'][rows, target_action].detach()
    decision_loss = _masked_mean(
        F.relu(target_iou - expected_iou), valid, zero)

    expected_response = (
        policy * oracle['soft_response'].detach()).sum(-1)
    target_response = oracle['soft_response'][rows, target_action].detach()
    response_aux_loss = _masked_mean(
        F.relu(target_response - expected_response), actionable, zero)

    total = (
        float(cfg.TRAIN.ERC_AV1_UTILITY_WEIGHT) * utility_loss
        + float(cfg.TRAIN.ERC_AV1B_PERMISSION_WEIGHT) * permission_loss
        + float(cfg.TRAIN.ERC_AV1B_ACTION_CHOICE_WEIGHT) * action_choice_loss
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
        false_positive = predicted_intervention & preserve
        nonkeep_correct = raw_action.eq(target_action) & actionable

        def mean(values, mask=valid):
            return _masked_mean(values, mask, zero.detach())

        intervention_count = predicted_intervention.sum().clamp_min(1)
        actionable_count = actionable.sum().clamp_min(1)
        preserve_count = preserve.sum().clamp_min(1)
        oracle_gain = oracle['target_hard_iou_gain']
        positive_oracle_gain = mean(oracle_gain, actionable)
        learned_gain = mean(candidate_gain, valid)
        actionable_best = mean(strongest_nonkeep, actionable)
        preserve_best = mean(strongest_nonkeep, preserve)
        status = {
            'Loss/erc_total': total.detach(),
            'Loss/erc_av1_utility': utility_loss.detach(),
            'Loss/erc_av1b_permission': permission_loss.detach(),
            'Loss/erc_av1b_action_choice': action_choice_loss.detach(),
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
            'ERC/av1b_actionable_action_accuracy': (
                nonkeep_correct.sum().float() / actionable_count),
            'ERC/av1_best_utility_mean': mean(strongest_nonkeep),
            'ERC/av1b_actionable_best_utility_mean': actionable_best,
            'ERC/av1b_preserve_best_utility_mean': preserve_best,
            'ERC/av1b_permission_separation': (
                actionable_best - preserve_best),
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
            'ERC/av1_oracle_capture_ratio': (
                learned_gain / positive_oracle_gain.clamp_min(1e-6)),
        }
    return total, status

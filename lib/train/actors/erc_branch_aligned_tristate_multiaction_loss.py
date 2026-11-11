"""Independent response-action Oracle and losses for FER-v5.

The Oracle never reads the learned action, a learned spatial mask, bbox_map or
candidate IoU.  It evaluates the exact deployable response actions on
``Hann*C*T`` and supervises one of KEEP / favour-C / favour-T (with fixed
magnitudes).  This keeps ERC at the evidence layer and leaves box-quality
estimation to the later RQC module.
"""

import torch
from torch.nn import functional as F


def _masked_mean(values, mask, zero):
    if mask.any():
        return values[mask].mean()
    return zero


def _balanced_mean(loss, target, class_frequency, max_weight, zero):
    if target.numel() == 0:
        return zero
    frequency = class_frequency.to(loss).clamp_min(1e-4)
    weight = frequency.reciprocal().clamp(max=float(max_weight))
    weight = weight / weight.mean().clamp_min(1e-6)
    sample_weight = weight[target]
    return (loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)


@torch.no_grad()
def build_independent_multiaction_oracle(pred, gt_heatmap, cfg):
    """Return exact hard-action labels and detached action utilities."""
    log_base = pred['erc_v5_log_base'].detach().flatten(1)
    difference = pred['erc_v5_correction_basis'].detach().flatten(1)
    support = pred['erc_v5_support_mask'].detach().flatten(1).bool()
    actions = pred['erc_v5_action_values'].detach().to(log_base)
    gt = gt_heatmap.detach().flatten(1).clamp_min(0.0)
    active = pred['erc_active_mask'].detach().reshape(-1).bool()
    valid = active & support.any(-1) & gt.max(-1).values.gt(0.0)

    scores = log_base[:, None, :] + actions[None, :, None] * difference[:, None, :]
    scores = scores.masked_fill(
        ~support[:, None, :], torch.finfo(scores.dtype).min)
    hard_index = scores.argmax(-1)
    rows = torch.arange(scores.shape[0], device=scores.device)[:, None]
    hard_relevance = gt[rows, hard_index]
    hard_gain = hard_relevance - hard_relevance[:, :1]
    changed = hard_index.ne(hard_index[:, :1])

    topk = min(int(cfg.TRAIN.ERC_V5_ORACLE_TOPK), scores.shape[-1])
    top_values, top_indices = scores.topk(topk, dim=-1)
    top_relevance = gt[:, None, :].expand_as(scores).gather(-1, top_indices)
    temperature = max(float(cfg.TRAIN.ERC_V5_ORACLE_SCORE_TEMPERATURE), 1e-6)
    soft_relevance = (
        F.softmax(top_values / temperature, dim=-1) * top_relevance
    ).sum(-1)
    soft_gain = soft_relevance - soft_relevance[:, :1]

    eligible = (
        changed
        & hard_gain.ge(float(cfg.TRAIN.ERC_V5_ORACLE_MIN_HARD_GAIN))
        & soft_gain.ge(float(cfg.TRAIN.ERC_V5_ORACLE_MIN_SOFT_GAIN))
        & actions[None].ne(0.0)
        & valid[:, None]
    )
    target = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
    # Lexicographic choice: hard gain, then soft gain, then smaller action.
    for batch_index in valid.nonzero(as_tuple=False).flatten().tolist():
        candidates = eligible[batch_index].nonzero(as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        values = hard_gain[batch_index, candidates]
        candidates = candidates[values.ge(values.max() - 1e-7)]
        values = soft_gain[batch_index, candidates]
        candidates = candidates[values.ge(values.max() - 1e-7)]
        target[batch_index] = candidates[
            actions[candidates].abs().argmin()]

    action_cost = float(cfg.TRAIN.ERC_V5_ORACLE_ACTION_COST)
    raw_utility = hard_gain + soft_gain - action_cost * actions.abs()[None]
    action_utility = torch.full_like(raw_utility, -1e4)
    action_utility[:, 0] = 0.0
    action_utility = torch.where(eligible, raw_utility, action_utility)
    return {
        'valid': valid,
        'target': target,
        'scores': scores,
        'hard_index': hard_index,
        'hard_relevance': hard_relevance,
        'hard_gain': hard_gain,
        'soft_relevance': soft_relevance,
        'soft_gain': soft_gain,
        'action_utility': action_utility,
        'eligible': eligible,
    }


def compute_branch_aligned_tristate_multiaction_loss_and_metrics(
    pred, gt_heatmap, cfg,
):
    logits = pred['erc_v5_action_logits']
    zero = logits.sum() * 0.0
    oracle = build_independent_multiaction_oracle(pred, gt_heatmap, cfg)
    valid = oracle['valid']
    if not valid.any():
        return zero, {
            'Loss/erc_total': zero.detach(),
            'ERC/v5_valid_fraction': valid.float().mean(),
        }

    target = oracle['target']
    actions = pred['erc_v5_action_values'].to(logits)
    keep_mask = actions.eq(0.0)
    cls_mask = actions.gt(0.0)
    cont_mask = actions.lt(0.0)
    nonkeep_mask = ~keep_mask
    target_intervene = target.ne(0)

    # A slowly moving, bounded class prior prevents majority KEEP collapse
    # without allowing a rare mini-batch class to receive an unbounded weight.
    ema = pred['erc_v5_action_class_ema']
    with torch.no_grad():
        batch_frequency = torch.bincount(
            target[valid], minlength=actions.numel()).to(ema)
        batch_frequency = batch_frequency / batch_frequency.sum().clamp_min(1.0)
        momentum = float(cfg.TRAIN.ERC_V5_CLASS_EMA_MOMENTUM)
        if logits.requires_grad:
            ema.mul_(momentum).add_(batch_frequency, alpha=1.0 - momentum)
            ema.div_(ema.sum().clamp_min(1e-6))
    max_weight = float(cfg.TRAIN.ERC_V5_MAX_CLASS_WEIGHT)

    permission_target = target_intervene.long()
    permission_logits = pred['erc_v5_permission_logit'].reshape(-1)
    permission_frame = F.binary_cross_entropy_with_logits(
        permission_logits, permission_target.to(logits.dtype), reduction='none')
    permission_frequency = torch.stack([
        ema[keep_mask].sum(), ema[nonkeep_mask].sum()])
    permission_loss = _balanced_mean(
        permission_frame[valid], permission_target[valid],
        permission_frequency, max_weight, zero)

    intervention = valid & target_intervene
    target_is_cont = actions[target].lt(0.0).long()
    direction_frame = F.cross_entropy(
        pred['erc_v5_direction_logits'], target_is_cont, reduction='none')
    direction_frequency = torch.stack([
        ema[cls_mask].sum(), ema[cont_mask].sum()])
    direction_loss = _balanced_mean(
        direction_frame[intervention], target_is_cont[intervention],
        direction_frequency, max_weight, zero)

    magnitude_losses = []
    magnitude_targets = []
    magnitude_frequencies = []
    for sign_mask in (cls_mask, cont_mask):
        indices = sign_mask.nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        selected = intervention & sign_mask[target]
        if not selected.any():
            continue
        local_target = (target[selected][:, None] == indices[None]).long().argmax(-1)
        magnitude_losses.append(F.cross_entropy(
            logits[selected][:, indices], local_target, reduction='none'))
        magnitude_targets.append(local_target)
        magnitude_frequencies.append(ema[indices] / ema[indices].sum().clamp_min(1e-6))
    if magnitude_losses:
        magnitude_loss = torch.stack([
            _balanced_mean(loss, local_target, frequency, max_weight, zero)
            for loss, local_target, frequency in zip(
                magnitude_losses, magnitude_targets, magnitude_frequencies)
        ]).mean()
    else:
        magnitude_loss = zero

    utility_temperature = max(
        float(cfg.TRAIN.ERC_V5_ORACLE_SCORE_TEMPERATURE), 1e-6)
    utility_target = F.softmax(
        oracle['action_utility'].detach() / utility_temperature, dim=-1)
    utility_frame = F.kl_div(
        F.log_softmax(logits, dim=-1), utility_target,
        reduction='none').sum(-1)
    utility_loss = _masked_mean(utility_frame, valid, zero)

    log_base = pred['erc_v5_log_base'].detach().flatten(1)
    difference = pred['erc_v5_correction_basis'].detach().flatten(1)
    learned_alpha = pred['erc_v5_predicted_alpha_scalar']
    learned_score = log_base + learned_alpha[:, None] * difference
    support = pred['erc_v5_support_mask'].flatten(1).bool()
    learned_score = learned_score.masked_fill(
        ~support, torch.finfo(learned_score.dtype).min)
    gt = gt_heatmap.detach().flatten(1).clamp_min(0.0)
    topk = min(int(cfg.TRAIN.ERC_V5_ORACLE_TOPK), learned_score.shape[-1])
    learned_values, learned_indices = learned_score.topk(topk, dim=-1)
    learned_relevance = gt.gather(-1, learned_indices)
    learned_expectation = (
        F.softmax(learned_values / utility_temperature, dim=-1)
        * learned_relevance).sum(-1)
    oracle_expectation = oracle['soft_relevance'].gather(
        1, target[:, None]).squeeze(1).detach()
    decision_frame = F.relu(
        oracle_expectation - learned_expectation
        + float(cfg.TRAIN.ERC_V5_DECISION_MARGIN))
    decision_loss = _masked_mean(decision_frame, intervention, zero)

    keep = valid & ~target_intervene
    baseline_expectation = oracle['soft_relevance'][:, 0].detach()
    soft_noharm = F.relu(baseline_expectation - learned_expectation)
    rows = torch.arange(logits.shape[0], device=logits.device)
    base_index = log_base.argmax(-1)
    learned_index = learned_score.argmax(-1)
    changed = learned_index.ne(base_index)
    harmful = keep & changed & gt[rows, learned_index].lt(gt[rows, base_index])
    hard_noharm = F.relu(
        learned_score[rows, learned_index]
        - learned_score[rows, base_index]
        + float(cfg.TRAIN.ERC_V5_DECISION_MARGIN))
    noharm_loss = (
        _masked_mean(soft_noharm, keep, zero)
        + _masked_mean(hard_noharm, harmful, zero)
    )

    total = (
        float(cfg.TRAIN.ERC_V5_PERMISSION_WEIGHT) * permission_loss
        + float(cfg.TRAIN.ERC_V5_DIRECTION_WEIGHT) * direction_loss
        + float(cfg.TRAIN.ERC_V5_MAGNITUDE_WEIGHT) * magnitude_loss
        + float(cfg.TRAIN.ERC_V5_UTILITY_WEIGHT) * utility_loss
        + float(cfg.TRAIN.ERC_V5_DECISION_WEIGHT) * decision_loss
        + float(cfg.TRAIN.ERC_V5_NOHARM_WEIGHT) * noharm_loss
    )

    with torch.no_grad():
        predicted = pred['erc_v5_raw_hard_action']
        learned_gain = gt[rows, learned_index] - gt[rows, base_index]
        semantic_target = torch.where(
            actions[target].gt(0), torch.ones_like(target),
            torch.where(actions[target].lt(0), torch.full_like(target, 2), target))
        semantic_pred = torch.where(
            actions[predicted].gt(0), torch.ones_like(predicted),
            torch.where(actions[predicted].lt(0), torch.full_like(predicted, 2), predicted))

        def mean(values, mask=valid):
            return _masked_mean(values, mask, zero.detach())

        status = {
            'Loss/erc_total': total.detach(),
            'Loss/erc_v5_permission': permission_loss.detach(),
            'Loss/erc_v5_direction': direction_loss.detach(),
            'Loss/erc_v5_magnitude': magnitude_loss.detach(),
            'Loss/erc_v5_utility': utility_loss.detach(),
            'Loss/erc_v5_decision': decision_loss.detach(),
            'Loss/erc_v5_noharm': noharm_loss.detach(),
            'ERC/v5_valid_fraction': valid.float().mean(),
            'ERC/v5_oracle_keep_rate': mean(target.eq(0).float()),
            'ERC/v5_oracle_cls_rate': mean(actions[target].gt(0).float()),
            'ERC/v5_oracle_cont_rate': mean(actions[target].lt(0).float()),
            'ERC/v5_pred_keep_rate': mean(predicted.eq(0).float()),
            'ERC/v5_pred_cls_rate': mean(actions[predicted].gt(0).float()),
            'ERC/v5_pred_cont_rate': mean(actions[predicted].lt(0).float()),
            'ERC/v5_action_accuracy': mean(predicted.eq(target).float()),
            'ERC/v5_semantic_accuracy': mean(
                semantic_pred.eq(semantic_target).float()),
            'ERC/v5_selection_change_rate': mean(changed.float()),
            'ERC/v5_target_relevance_gain': mean(learned_gain),
            'ERC/v5_changed_target_relevance_gain': mean(
                learned_gain, valid & changed),
            'ERC/v5_beneficial_change_rate': mean(
                (changed & learned_gain.gt(0)).float()),
            'ERC/v5_harmful_change_rate': mean(
                (changed & learned_gain.lt(0)).float()),
            'ERC/v5_alpha_abs_mean': mean(learned_alpha.abs()),
            'ERC/v5_alpha_positive_rate': mean(learned_alpha.gt(0).float()),
            'ERC/v5_alpha_negative_rate': mean(learned_alpha.lt(0).float()),
            'ERC/v5_oracle_hard_gain': mean(
                oracle['hard_gain'].gather(1, target[:, None]).squeeze(1)),
            'ERC/v5_oracle_soft_gain': mean(
                oracle['soft_gain'].gather(1, target[:, None]).squeeze(1)),
        }
        for action_index in range(actions.numel()):
            status['ERC/v5_oracle_action_%d_rate' % action_index] = mean(
                target.eq(action_index).float())
            status['ERC/v5_pred_action_%d_rate' % action_index] = mean(
                predicted.eq(action_index).float())
    return total, status

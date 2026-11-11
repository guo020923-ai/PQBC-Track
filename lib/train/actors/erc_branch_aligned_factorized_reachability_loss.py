"""Independent minimum-sufficient action supervision for FER-v6."""

import torch
from torch.nn import functional as F

from .erc_branch_aligned_tristate_multiaction_loss import (
    _balanced_mean, _masked_mean,
)


def _sparse_batch_balanced_permission_loss(
        permission_logit, permission_target, valid, empty_negative_weight,
        zero):
    """Balance KEEP/INTERVENE after reduction, not through a large pos_weight.

    INTERVENE is rare enough that many mini-batches contain no positive frame.
    Letting each of those batches contribute a full negative BCE overwhelms the
    much rarer mixed batches after per-step gradient clipping.  A mixed batch
    therefore gives the positive and negative classes equal aggregate weight;
    an all-negative batch contributes only the explicitly configured weak
    regularizer (zero by default).  This preserves the class balance through
    gradient clipping instead of relying on a pre-clipping scalar multiplier.
    """
    positive = valid & permission_target.bool()
    negative = valid & ~permission_target.bool()
    positive_loss = _masked_mean(
        F.softplus(-permission_logit), positive, zero)
    negative_loss = _masked_mean(
        F.softplus(permission_logit), negative, zero)
    positive_batch = positive.any()
    if positive_batch:
        loss = 0.5 * (positive_loss + negative_loss)
    else:
        loss = float(empty_negative_weight) * negative_loss
    return loss, positive_loss, negative_loss, positive_batch


@torch.no_grad()
def build_factorized_reachability_oracle(pred, gt_heatmap, cfg):
    log_base = pred['erc_v6_log_base'].detach().flatten(1)
    corrections = pred['erc_v6_action_correction_maps'].detach().flatten(2)
    support = pred['erc_v6_support_mask'].detach().flatten(1).bool()
    gt = gt_heatmap.detach().flatten(1).clamp_min(0.0)
    active = pred['erc_active_mask'].detach().reshape(-1).bool()
    valid = active & support.any(-1) & gt.max(-1).values.gt(0)
    scores = (log_base[:, None] + corrections).masked_fill(
        ~support[:, None], torch.finfo(log_base.dtype).min)
    hard_index = scores.argmax(-1)
    rows = torch.arange(scores.shape[0], device=scores.device)[:, None]
    hard_relevance = gt[rows, hard_index]
    hard_gain = hard_relevance - hard_relevance[:, :1]
    changed = hard_index.ne(hard_index[:, :1])
    topk = min(int(cfg.TRAIN.ERC_V6_ORACLE_TOPK), scores.shape[-1])
    top_values, top_indices = scores.topk(topk, dim=-1)
    top_relevance = gt[:, None].expand_as(scores).gather(-1, top_indices)
    temperature = max(float(cfg.TRAIN.ERC_V6_ORACLE_TEMPERATURE), 1e-6)
    soft_relevance = (
        F.softmax(top_values / temperature, -1) * top_relevance).sum(-1)
    soft_gain = soft_relevance - soft_relevance[:, :1]
    eligible = (
        changed
        & hard_gain.ge(float(cfg.TRAIN.ERC_V6_ORACLE_MIN_HARD_GAIN))
        & soft_gain.ge(float(cfg.TRAIN.ERC_V6_ORACLE_MIN_SOFT_GAIN))
        & valid[:, None]
    )
    eligible[:, 0] = False
    magnitude_index = pred['erc_v6_action_magnitude_index'].detach()
    magnitudes = pred['erc_v6_magnitudes'].detach().to(scores)
    action_magnitude = torch.zeros(scores.shape[1], device=scores.device)
    nonkeep = magnitude_index.ge(0)
    action_magnitude[nonkeep] = magnitudes[magnitude_index[nonkeep]]
    target = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
    for batch_index in valid.nonzero(as_tuple=False).flatten().tolist():
        candidates = eligible[batch_index].nonzero(as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        gain = hard_gain[batch_index, candidates]
        candidates = candidates[gain.ge(gain.max() - 1e-7)]
        size = action_magnitude[candidates]
        candidates = candidates[size.le(size.min() + 1e-7)]
        gain = soft_gain[batch_index, candidates]
        target[batch_index] = candidates[gain.argmax()]
    return {
        'valid': valid, 'target': target, 'scores': scores,
        'hard_index': hard_index, 'hard_gain': hard_gain,
        'soft_relevance': soft_relevance, 'soft_gain': soft_gain,
        'changed': changed, 'eligible': eligible,
    }


def compute_factorized_reachability_loss_and_metrics(pred, gt_heatmap, cfg):
    permission_logit = pred['erc_v6_permission_logit']
    zero = permission_logit.sum() * 0.0
    oracle = build_factorized_reachability_oracle(pred, gt_heatmap, cfg)
    valid = oracle['valid']
    if not valid.any():
        return zero, {'Loss/erc_total': zero.detach(),
                      'ERC/v6_valid_fraction': valid.float().mean()}
    target = oracle['target']
    direction = pred['erc_v6_action_direction'].long()
    scope = pred['erc_v6_action_scope'].long()
    magnitude_index = pred['erc_v6_action_magnitude_index'].long()
    number_actions = direction.numel()
    intervene = target.ne(0)
    target_direction = (direction[target] - 1).clamp_min(0)
    target_scope = scope[target]
    target_magnitude = magnitude_index[target].clamp_min(0)

    ema = pred['erc_v6_action_class_ema']
    permission_balance_mode = str(
        cfg.TRAIN.ERC_V6_PERMISSION_BALANCE_MODE).lower()
    with torch.no_grad():
        if permission_balance_mode == 'binary_ratio':
            uniform = torch.full_like(ema, 1.0 / number_actions)
            if torch.allclose(ema, uniform, atol=1e-7, rtol=0.0):
                keep_prior = float(cfg.TRAIN.ERC_V6_PERMISSION_KEEP_PRIOR)
                if not 0.0 < keep_prior < 1.0:
                    raise ValueError(
                        'ERC_V6_PERMISSION_KEEP_PRIOR must be in (0, 1)')
                ema[0] = keep_prior
                ema[1:] = (1.0 - keep_prior) / (number_actions - 1)
        frequency = torch.bincount(
            target[valid], minlength=number_actions).to(ema)
        frequency = frequency / frequency.sum().clamp_min(1)
        if permission_logit.requires_grad:
            momentum = float(cfg.TRAIN.ERC_V6_CLASS_EMA_MOMENTUM)
            ema.mul_(momentum).add_(frequency, alpha=1 - momentum)
            ema.div_(ema.sum().clamp_min(1e-6))
    max_weight = float(cfg.TRAIN.ERC_V6_MAX_CLASS_WEIGHT)

    permission_frequency = torch.stack([ema[0], ema[1:].sum()])
    permission_target = intervene.long()
    permission_positive_loss = _masked_mean(
        F.softplus(-permission_logit), valid & intervene, zero)
    permission_negative_loss = _masked_mean(
        F.softplus(permission_logit), valid & ~intervene, zero)
    permission_positive_batch = (valid & intervene).any()
    if permission_balance_mode == 'binary_ratio':
        # Permission is the only highly imbalanced binary decision.  Balance
        # KEEP/INTERVENE directly; applying the 16-way cap here caused the
        # zero-initialized policy to collapse to KEEP before it could learn
        # separability.  Conditional direction/scope/magnitude heads retain
        # their independent capped balancing below.
        max_pos_weight = float(cfg.TRAIN.ERC_V6_PERMISSION_MAX_POS_WEIGHT)
        positive_weight = (
            permission_frequency[0]
            / permission_frequency[1].clamp_min(1e-4)
        ).clamp(min=1.0, max=max_pos_weight).detach()
        permission_frame = F.binary_cross_entropy_with_logits(
            permission_logit,
            permission_target.to(permission_logit),
            pos_weight=positive_weight,
            reduction='none')
        permission_loss = _masked_mean(permission_frame, valid, zero)
    elif permission_balance_mode == 'sparse_batch_balanced':
        positive_weight = permission_logit.new_tensor(1.0)
        (
            permission_loss,
            permission_positive_loss,
            permission_negative_loss,
            permission_positive_batch,
        ) = _sparse_batch_balanced_permission_loss(
            permission_logit,
            permission_target,
            valid,
            cfg.TRAIN.ERC_V6_PERMISSION_EMPTY_NEGATIVE_WEIGHT,
            zero,
        )
    elif permission_balance_mode == 'legacy_capped':
        positive_weight = permission_logit.new_tensor(1.0)
        permission_frame = F.binary_cross_entropy_with_logits(
            permission_logit, permission_target.to(permission_logit),
            reduction='none')
        permission_loss = _balanced_mean(
            permission_frame[valid], permission_target[valid],
            permission_frequency, max_weight, zero)
    else:
        raise ValueError(
            'Unknown ERC_V6_PERMISSION_BALANCE_MODE: %s'
            % permission_balance_mode)

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
        for index in range(pred['erc_v6_magnitudes'].numel())])
    magnitude_loss = _balanced_mean(
        magnitude_frame[valid & intervene], target_magnitude[valid & intervene],
        magnitude_frequency, max_weight, zero)

    joint_probability = pred['erc_v6_joint_probability'].clamp_min(1e-8)
    action_frame = -joint_probability.log().gather(
        1, target[:, None]).squeeze(1)
    action_loss = _balanced_mean(
        action_frame[valid], target[valid], ema, max_weight, zero)

    log_base = pred['erc_v6_log_base'].detach().flatten(1)
    learned_score = log_base + pred[
        'erc_v6_predicted_correction_map'].flatten(1)
    support = pred['erc_v6_support_mask'].flatten(1).bool()
    learned_score = learned_score.masked_fill(
        ~support, torch.finfo(learned_score.dtype).min)
    gt = gt_heatmap.detach().flatten(1).clamp_min(0)
    topk = min(int(cfg.TRAIN.ERC_V6_ORACLE_TOPK), learned_score.shape[-1])
    temperature = max(float(cfg.TRAIN.ERC_V6_ORACLE_TEMPERATURE), 1e-6)
    learned_values, learned_indices = learned_score.topk(topk, -1)
    learned_expectation = (
        F.softmax(learned_values / temperature, -1)
        * gt.gather(-1, learned_indices)).sum(-1)
    oracle_expectation = oracle['soft_relevance'].gather(
        1, target[:, None]).squeeze(1).detach()
    decision_loss = _masked_mean(
        F.relu(oracle_expectation - learned_expectation),
        valid & intervene, zero)

    keep = valid & ~intervene
    baseline_expectation = oracle['soft_relevance'][:, 0].detach()
    soft_noharm = F.relu(baseline_expectation - learned_expectation)
    rows = torch.arange(log_base.shape[0], device=log_base.device)
    base_index = log_base.argmax(-1)
    learned_index = learned_score.argmax(-1)
    changed = learned_index.ne(base_index)
    relevance_gain = gt[rows, learned_index] - gt[rows, base_index]
    harmful = keep & changed & relevance_gain.lt(0)
    hard_noharm = F.relu(
        learned_score[rows, learned_index] - learned_score[rows, base_index])
    noharm_loss = (
        _masked_mean(soft_noharm, keep, zero)
        + _masked_mean(hard_noharm, harmful, zero))

    total = (
        float(cfg.TRAIN.ERC_V6_PERMISSION_WEIGHT) * permission_loss
        + float(cfg.TRAIN.ERC_V6_DIRECTION_WEIGHT) * direction_loss
        + float(cfg.TRAIN.ERC_V6_SCOPE_WEIGHT) * scope_loss
        + float(cfg.TRAIN.ERC_V6_MAGNITUDE_WEIGHT) * magnitude_loss
        + float(cfg.TRAIN.ERC_V6_ACTION_WEIGHT) * action_loss
        + float(cfg.TRAIN.ERC_V6_DECISION_WEIGHT) * decision_loss
        + float(cfg.TRAIN.ERC_V6_NOHARM_WEIGHT) * noharm_loss)

    with torch.no_grad():
        predicted = pred['erc_v6_raw_hard_action']
        predicted_permission = permission_logit.gt(0)

        def mean(values, mask=valid):
            return _masked_mean(values, mask, zero.detach())

        status = {
            'Loss/erc_total': total.detach(),
            'Loss/erc_v6_permission': permission_loss.detach(),
            'Loss/erc_v6_permission_positive': (
                permission_positive_loss.detach()),
            'Loss/erc_v6_permission_negative': (
                permission_negative_loss.detach()),
            'Loss/erc_v6_direction': direction_loss.detach(),
            'Loss/erc_v6_scope': scope_loss.detach(),
            'Loss/erc_v6_magnitude': magnitude_loss.detach(),
            'Loss/erc_v6_action': action_loss.detach(),
            'Loss/erc_v6_decision': decision_loss.detach(),
            'Loss/erc_v6_noharm': noharm_loss.detach(),
            'ERC/v6_valid_fraction': valid.float().mean(),
            'ERC/v6_oracle_keep_rate': mean(target.eq(0).float()),
            'ERC/v6_oracle_cls_rate': mean(direction[target].eq(1).float()),
            'ERC/v6_oracle_cont_rate': mean(direction[target].eq(2).float()),
            'ERC/v6_oracle_regional_rate': mean(scope[target].eq(1).float()),
            'ERC/v6_pred_keep_rate': mean(predicted.eq(0).float()),
            'ERC/v6_pred_cls_rate': mean(direction[predicted].eq(1).float()),
            'ERC/v6_pred_cont_rate': mean(direction[predicted].eq(2).float()),
            'ERC/v6_pred_regional_rate': mean(scope[predicted].eq(1).float()),
            'ERC/v6_action_accuracy': mean(predicted.eq(target).float()),
            'ERC/v6_selection_change_rate': mean(changed.float()),
            'ERC/v6_target_relevance_gain': mean(relevance_gain),
            'ERC/v6_changed_target_relevance_gain': mean(
                relevance_gain, valid & changed),
            'ERC/v6_beneficial_change_rate': mean(
                (changed & relevance_gain.gt(0)).float()),
            'ERC/v6_harmful_change_rate': mean(
                (changed & relevance_gain.lt(0)).float()),
            'ERC/v6_permission_mean': mean(
                pred['erc_v6_permission_probability']),
            'ERC/v6_permission_positive_batch': (
                permission_positive_batch.to(permission_logit).detach()),
            'ERC/v6_permission_positive_probability': mean(
                pred['erc_v6_permission_probability'], valid & intervene),
            'ERC/v6_permission_negative_probability': mean(
                pred['erc_v6_permission_probability'], valid & ~intervene),
            'ERC/v6_permission_recall': mean(
                predicted_permission.float(), valid & intervene),
            'ERC/v6_permission_false_positive_rate': mean(
                predicted_permission.float(), valid & ~intervene),
            'ERC/v6_permission_pos_weight': positive_weight.detach(),
            'ERC/v6_oracle_hard_gain': mean(
                oracle['hard_gain'].gather(1, target[:, None]).squeeze(1)),
            'ERC/v6_oracle_soft_gain': mean(
                oracle['soft_gain'].gather(1, target[:, None]).squeeze(1)),
        }
    return total, status

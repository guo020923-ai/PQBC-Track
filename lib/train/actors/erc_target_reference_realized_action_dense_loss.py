"""Realized-action and region-ranking objective for dense ERC DR3.

DR2 labelled a frame actionable whenever a better response location was
theoretically reachable.  That label did not guarantee that the residual
currently proposed by ERC moved the deployed decision in the right direction.
DR3 removes this mismatch:

* the residual proposal is trained with a safe-region versus hard-background
  listwise ranking objective;
* the gate target is selected by evaluating the actual path
  ``S(q) = S0 + q * Delta_raw`` for a small set of continuous gate actions;
* ties prefer abstention and non-positive realized gain maps to q=0;
* the same causal temporal gate used at inference is simulated in training.

All quality targets are derived from the official center-response heatmap.
The loss never consumes candidate boxes, bbox geometry, IoU, or dataset names.
"""

import torch
from torch.nn import functional as F

from .erc_target_reference_trajectory_safe_dense_loss import (
    _causal_deploy_gate,
    _domain_mean,
    _temporal_robust_gate_target,
)


def _masked_soft_peak(values, mask, temperature):
    """Low-temperature differentiable maximum over a non-empty mask."""
    minimum = torch.finfo(values.dtype).min
    masked = values.masked_fill(~mask, minimum)
    return float(temperature) * torch.logsumexp(
        masked / float(temperature), dim=-1)


def _actual_action_oracle(
        base_log, raw_correction, target_flat, action_levels,
        valid, min_gain):
    """Return the smallest best action along the current residual path.

    Actions are ordered from zero to one.  ``torch.max`` returns the first
    maximum, so equal-quality actions deterministically prefer less
    intervention.  The result is detached before supervising the gate.
    """
    levels = torch.as_tensor(
        action_levels, device=base_log.device, dtype=base_log.dtype)
    if levels.ndim != 1 or levels.numel() < 2:
        raise ValueError('DR3 requires at least two gate action levels')
    if not torch.all(levels[1:] >= levels[:-1]):
        raise ValueError('DR3 gate action levels must be sorted')
    if float(levels[0]) != 0.0 or float(levels[-1]) != 1.0:
        raise ValueError('DR3 gate action levels must start at 0 and end at 1')

    action_log = (
        base_log.unsqueeze(0)
        + levels[:, None, None] * raw_correction.flatten(1).unsqueeze(0)
    )
    action_index = action_log.argmax(-1)
    rows = torch.arange(base_log.shape[0], device=base_log.device)
    action_quality = target_flat[rows[None, :], action_index]
    best_quality, best_action = action_quality.max(0)
    baseline_quality = action_quality[0]
    best_gain = best_quality - baseline_quality
    gate_target = levels[best_action]
    actionable = valid & best_gain.gt(float(min_gain)) & gate_target.gt(0.0)
    gate_target = torch.where(
        actionable, gate_target, torch.zeros_like(gate_target))
    return (
        gate_target.detach(), actionable, best_gain.detach(),
        best_action.detach(), action_quality.detach(), action_index.detach(),
    )


def compute_target_reference_realized_action_dense_loss_and_metrics(
        pred, gt_heatmap, cfg, num_search_frames=1, dataset_ids=None):
    raw_correction = pred['erc_dr3_raw_correction']
    gate = pred['erc_dr3_current_gate'].reshape(-1)
    active = pred['erc_active_mask'].reshape(-1).bool()
    support = pred['erc_dr1_support_mask'].bool().flatten(1)
    deploy_gate = _causal_deploy_gate(
        gate, active, num_search_frames,
        cfg.MODEL.HEAD.ERC.DR2_TEMPORAL_EMA,
        cfg.MODEL.HEAD.ERC.DR2_ENABLE_MEMORY,
    )
    correction = raw_correction * deploy_gate[:, None, None]
    zero = correction.sum() * 0.0

    target = gt_heatmap.detach().to(correction).clamp_min(0.0)
    target = target / target.flatten(1).amax(-1).clamp_min(1e-6)[:, None, None]
    target_flat = target.flatten(1)
    batch = target.shape[0]
    rows = torch.arange(batch, device=target.device)

    tiny = torch.finfo(correction.dtype).tiny
    base_log = pred['erc_windowed_base_score'].detach().clamp_min(tiny).log()
    minimum = torch.finfo(base_log.dtype).min
    base_log = base_log.masked_fill(~support, minimum)
    baseline_index = base_log.argmax(-1)
    baseline_quality = target_flat[rows, baseline_index]
    valid = active & target_flat.amax(-1).gt(0.0) & support.any(-1)

    (local_gate_target, actionable, oracle_gain, best_action,
     action_quality, action_index) = (
        _actual_action_oracle(
            base_log, raw_correction, target_flat,
            cfg.TRAIN.ERC_DR3_ACTION_LEVELS, valid,
            cfg.TRAIN.ERC_DR3_MIN_REALIZED_GAIN,
        )
    )
    action_changed = action_index.ne(action_index[0:1]).any(0)
    gate_supervised = valid & action_changed
    gate_target, pair_valid, robust_target = _temporal_robust_gate_target(
        local_gate_target, valid, num_search_frames,
        cfg.TRAIN.ERC_DR3_TEMPORAL_RISK_WEIGHT,
        cfg.TRAIN.ERC_DR3_TEMPORAL_CONSENSUS_MIX,
    )
    gate_target = gate_target.detach()

    predicted_log = base_log + correction.flatten(1)
    predicted_index = predicted_log.argmax(-1)
    predicted_quality = target_flat[rows, predicted_index]
    hard_gain = predicted_quality - baseline_quality

    # Proposal learning is independent of the current gate target.  This
    # prevents the zero-initialized residual/gate pair from collapsing into a
    # self-confirming abstention solution.
    safe_threshold = float(cfg.TRAIN.ERC_DR3_SAFE_TARGET_THRESHOLD)
    safe_mask = target_flat.ge(safe_threshold) & support
    unsafe_mask = ~safe_mask & support
    safe_exists = safe_mask.any(-1)
    unsafe_exists = unsafe_mask.any(-1)
    proposal_train = (
        valid & baseline_quality.lt(safe_threshold)
        & safe_exists & unsafe_exists)
    raw_proposal_log = base_log + raw_correction.flatten(1)
    temperature = max(float(cfg.TRAIN.ERC_DR3_REGION_TEMPERATURE), 1e-4)
    safe_peak = _masked_soft_peak(raw_proposal_log, safe_mask, temperature)
    unsafe_peak = _masked_soft_peak(raw_proposal_log, unsafe_mask, temperature)
    proposal_rank_element = F.relu(
        float(cfg.TRAIN.ERC_DR3_REGION_MARGIN) - safe_peak + unsafe_peak)

    base_probability = torch.softmax(base_log, dim=-1)
    raw_probability = torch.softmax(raw_proposal_log, dim=-1)
    predicted_probability = torch.softmax(predicted_log, dim=-1)
    base_expected_quality = (base_probability * target_flat).sum(-1)
    raw_expected_quality = (raw_probability * target_flat).sum(-1)
    predicted_expected_quality = (predicted_probability * target_flat).sum(-1)
    proposal_utility_element = F.relu(
        float(cfg.TRAIN.ERC_DR3_MIN_REALIZED_GAIN)
        - (raw_expected_quality - base_expected_quality))
    action_utility_element = F.relu(
        gate_target * oracle_gain
        - (predicted_expected_quality - base_expected_quality))
    noharm_element = F.relu(
        base_expected_quality - predicted_expected_quality)

    log_predicted_probability = F.log_softmax(predicted_log, dim=-1)
    preserve_kl = F.kl_div(
        log_predicted_probability, base_probability,
        reduction='none').sum(-1)
    preserve_element = (
        preserve_kl + correction.abs().flatten(1).mean(-1))
    minimal_element = raw_correction.abs().flatten(1).mean(-1)
    gate_element = F.smooth_l1_loss(
        gate, gate_target, reduction='none')

    worst = float(cfg.TRAIN.ERC_DR3_DOMAIN_WORST_WEIGHT)
    reduce = lambda values, mask: _domain_mean(
        values, mask, dataset_ids, worst, zero)
    proposal_rank_loss = reduce(proposal_rank_element, proposal_train)
    proposal_utility_loss = reduce(proposal_utility_element, proposal_train)
    gate_loss = reduce(gate_element, gate_supervised)
    action_utility_loss = reduce(action_utility_element, actionable)
    noharm_loss = reduce(noharm_element, valid)
    preserve_loss = reduce(
        preserve_element, gate_supervised & ~actionable)
    minimal_loss = reduce(minimal_element, proposal_train)

    total = (
        float(cfg.TRAIN.ERC_DR3_GATE_WEIGHT) * gate_loss
        + float(cfg.TRAIN.ERC_DR3_PROPOSAL_RANK_WEIGHT) * proposal_rank_loss
        + float(cfg.TRAIN.ERC_DR3_PROPOSAL_UTILITY_WEIGHT)
        * proposal_utility_loss
        + float(cfg.TRAIN.ERC_DR3_ACTION_UTILITY_WEIGHT)
        * action_utility_loss
        + float(cfg.TRAIN.ERC_DR3_NOHARM_WEIGHT) * noharm_loss
        + float(cfg.TRAIN.ERC_DR3_PRESERVE_WEIGHT) * preserve_loss
        + float(cfg.TRAIN.ERC_DR3_MINIMAL_WEIGHT) * minimal_loss
    )

    with torch.no_grad():
        changed = valid & predicted_index.ne(baseline_index)
        beneficial = changed & hard_gain.gt(1e-8)
        harmful = changed & hard_gain.lt(-1e-8)
        valid_count = valid.float().sum().clamp_min(1.0)
        changed_count = changed.float().sum().clamp_min(1.0)
        actionable_count = actionable.float().sum().clamp_min(1.0)
        proposal_count = proposal_train.float().sum().clamp_min(1.0)
        status = {
            'Loss/erc_total': total.detach(),
            'Loss/erc_dr3_gate': gate_loss.detach(),
            'Loss/erc_dr3_proposal_rank': proposal_rank_loss.detach(),
            'Loss/erc_dr3_proposal_utility': proposal_utility_loss.detach(),
            'Loss/erc_dr3_action_utility': action_utility_loss.detach(),
            'Loss/erc_dr3_noharm': noharm_loss.detach(),
            'Loss/erc_dr3_preserve': preserve_loss.detach(),
            'Loss/erc_dr3_minimal': minimal_loss.detach(),
            'ERC/dr3_valid_fraction': valid.float().mean(),
            'ERC/dr3_proposal_train_rate': (
                proposal_train.float().sum() / valid_count),
            'ERC/dr3_actionable_rate': (
                actionable.float().sum() / valid_count),
            'ERC/dr3_gate_supervised_rate': (
                gate_supervised.float().sum() / valid_count),
            'ERC/dr3_oracle_gain': (
                oracle_gain[actionable].mean()
                if actionable.any() else zero.detach()),
            'ERC/dr3_oracle_gate_target': (
                local_gate_target[actionable].mean()
                if actionable.any() else zero.detach()),
            'ERC/dr3_gate_mean': (
                gate[valid].mean() if valid.any() else zero.detach()),
            'ERC/dr3_deploy_gate_mean': (
                deploy_gate[valid].mean() if valid.any() else zero.detach()),
            'ERC/dr3_gate_actionable_mean': (
                gate[actionable].mean() if actionable.any() else zero.detach()),
            'ERC/dr3_gate_preserve_mean': (
                gate[valid & ~actionable].mean()
                if (valid & ~actionable).any() else zero.detach()),
            'ERC/dr3_selection_change_rate': (
                changed.float().sum() / valid_count),
            'ERC/dr3_response_gain_per_valid': (
                (hard_gain * valid).sum() / valid_count),
            'ERC/dr3_changed_gain': (
                (hard_gain * changed).sum() / changed_count),
            'ERC/dr3_beneficial_change_rate': (
                beneficial.float().sum() / valid_count),
            'ERC/dr3_harmful_change_rate': (
                harmful.float().sum() / valid_count),
            'ERC/dr3_actionable_hit_rate': (
                (beneficial & actionable).float().sum() / actionable_count),
            'ERC/dr3_raw_expected_gain': (
                (raw_expected_quality - base_expected_quality)[proposal_train].mean()
                if proposal_train.any() else zero.detach()),
            'ERC/dr3_region_margin': (
                (safe_peak - unsafe_peak)[proposal_train].mean()
                if proposal_train.any() else zero.detach()),
            'ERC/dr3_raw_correction_abs': (
                raw_correction[valid].abs().mean()
                if valid.any() else zero.detach()),
            'ERC/dr3_correction_abs': (
                correction[valid].abs().mean()
                if valid.any() else zero.detach()),
            'ERC/dr3_correction_saturation': (
                correction[valid].abs().gt(
                    0.95 * float(cfg.MODEL.HEAD.ERC.MAX_ALPHA)
                ).float().mean() if valid.any() else zero.detach()),
            'ERC/dr3_best_action_index': (
                best_action[actionable].float().mean()
                if actionable.any() else zero.detach()),
            'ERC/dr3_best_full_action_rate': (
                best_action[actionable].eq(action_quality.shape[0] - 1)
                .float().mean() if actionable.any() else zero.detach()),
            'ERC/dr3_proposal_count': proposal_count.detach(),
        }
        if pair_valid is not None:
            status['ERC/dr3_temporal_pair_valid_rate'] = pair_valid.float().mean()
            status['ERC/dr3_temporal_robust_target_mean'] = (
                robust_target[pair_valid].mean()
                if pair_valid.any() else zero.detach())
        else:
            status['ERC/dr3_temporal_pair_valid_rate'] = zero.detach()
            status['ERC/dr3_temporal_robust_target_mean'] = zero.detach()
        if dataset_ids is not None:
            ids = dataset_ids.reshape(-1).to(valid.device)
            for dataset_id in ids[valid].unique(sorted=True):
                mask = valid & ids.eq(dataset_id)
                count = mask.float().sum().clamp_min(1.0)
                suffix = str(int(dataset_id.item()))
                status['ERC/dr3_domain_' + suffix + '_response_gain'] = (
                    (hard_gain * mask).sum() / count)
                status['ERC/dr3_domain_' + suffix + '_harmful_rate'] = (
                    harmful[mask].float().sum() / count)
                status['ERC/dr3_domain_' + suffix + '_gate_mean'] = (
                    gate[mask].mean())
    return total, status

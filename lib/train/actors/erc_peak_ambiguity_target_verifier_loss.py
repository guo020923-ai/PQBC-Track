"""Decision-aligned supervision for asymmetric peak-verifier ERC (PQ4)."""

import torch
from torch.nn import functional as F


def _domain_mean(values, mask, dataset_ids, worst_weight, zero):
    if not mask.any():
        return zero
    if dataset_ids is None:
        return values[mask].float().mean()
    ids = dataset_ids.reshape(-1).to(values.device)
    groups = []
    for dataset_id in ids[mask].unique(sorted=True):
        selected = mask & ids.eq(dataset_id)
        if selected.any():
            groups.append(values[selected].float().mean())
    if not groups:
        return zero
    groups = torch.stack(groups)
    weight = max(float(worst_weight), 0.0)
    return (groups.mean() + weight * groups.max()) / (1.0 + weight)


def compute_peak_ambiguity_target_verifier_loss_and_metrics(
        pred, gt_heatmap, cfg, dataset_ids=None):
    correction = pred['erc_pq4_raw_peak_correction']
    zero = correction.sum() * 0.0
    active = pred['erc_active_mask'].reshape(-1).bool()
    candidate_index = pred['erc_pq4_candidate_indices'].long()
    candidate_valid = pred['erc_pq4_candidate_valid'].bool()
    target = gt_heatmap.detach().flatten(1).to(correction).clamp_min(0.0)
    target = target / target.amax(-1, keepdim=True).clamp_min(1e-6)
    candidate_target = target.gather(1, candidate_index)
    candidate_target = candidate_target.masked_fill(~candidate_valid, -1.0)
    valid = active & candidate_valid.sum(-1).ge(2) & target.amax(-1).gt(0.0)
    if not valid.any():
        return zero, {
            'Loss/erc_total': zero.detach(),
            'ERC/pq4_valid_fraction': valid.float().mean(),
        }

    oracle_quality, oracle_slot = candidate_target.max(-1)
    base_quality = candidate_target[:, 0]
    oracle_gain = oracle_quality - base_quality
    actionable = (
        valid
        & oracle_slot.ne(0)
        & oracle_gain.gt(float(cfg.TRAIN.ERC_PQ4_MIN_HARD_GAIN))
    )
    preserve = valid & ~actionable
    target_slot = torch.where(
        actionable, oracle_slot, torch.zeros_like(oracle_slot))

    tiny = torch.finfo(correction.dtype).tiny
    windowed_base = pred['erc_windowed_base_score'].detach().clamp_min(tiny)
    candidate_log_base = windowed_base.log().gather(1, candidate_index)
    predicted_log = candidate_log_base + correction
    predicted_log = predicted_log.masked_fill(
        ~candidate_valid, torch.finfo(predicted_log.dtype).min)

    decision_element = F.cross_entropy(
        predicted_log, target_slot, reduction='none')
    rows = torch.arange(correction.shape[0], device=correction.device)
    oracle_score = predicted_log[rows, oracle_slot]
    baseline_score = predicted_log[:, 0]
    rank_element = F.relu(
        float(cfg.TRAIN.ERC_PQ4_RANK_MARGIN)
        - oracle_score + baseline_score)
    preserve_element = correction.abs().sum(-1) / candidate_valid.float().sum(
        -1).clamp_min(1.0)

    worst = float(cfg.TRAIN.ERC_PQ4_DOMAIN_WORST_WEIGHT)
    reduce = lambda values, mask: _domain_mean(
        values, mask, dataset_ids, worst, zero)
    decision_loss = reduce(decision_element, actionable)
    rank_loss = reduce(rank_element, actionable)
    preserve_decision_loss = reduce(decision_element, preserve)
    preserve_residual_loss = reduce(preserve_element, preserve)
    total = (
        float(cfg.TRAIN.ERC_PQ4_DECISION_WEIGHT) * decision_loss
        + float(cfg.TRAIN.ERC_PQ4_RANK_WEIGHT) * rank_loss
        + float(cfg.TRAIN.ERC_PQ4_PRESERVE_WEIGHT)
        * (preserve_decision_loss + preserve_residual_loss)
    )

    with torch.no_grad():
        predicted_slot = predicted_log.argmax(-1)
        changed = valid & predicted_slot.ne(0)
        predicted_quality = candidate_target.gather(
            1, predicted_slot[:, None]).squeeze(1)
        hard_gain = predicted_quality - base_quality
        beneficial = changed & hard_gain.gt(1e-8)
        harmful = changed & hard_gain.lt(-1e-8)
        valid_count = valid.float().sum().clamp_min(1.0)
        actionable_count = actionable.float().sum().clamp_min(1.0)
        ambiguity = pred['erc_pq4_cls_ambiguity'].reshape(-1)
        status = {
            'Loss/erc_total': total.detach(),
            'Loss/erc_pq4_decision': decision_loss.detach(),
            'Loss/erc_pq4_rank': rank_loss.detach(),
            'Loss/erc_pq4_preserve': (
                preserve_decision_loss + preserve_residual_loss).detach(),
            'ERC/pq4_valid_fraction': valid.float().mean(),
            'ERC/pq4_oracle_actionable_rate': (
                actionable.float().sum() / valid_count),
            'ERC/pq4_oracle_gain_per_valid': (
                (oracle_gain * actionable).sum() / valid_count),
            'ERC/pq4_oracle_gain_per_actionable': (
                (oracle_gain * actionable).sum() / actionable_count),
            'ERC/pq4_cls_ambiguity_mean': ambiguity[valid].mean(),
            'ERC/pq4_verifier_residual_abs': correction[valid].abs().mean(),
            'ERC/pq4_selection_change_rate': (
                changed.float().sum() / valid_count),
            'ERC/pq4_response_gain_per_valid': (
                (hard_gain * valid).sum() / valid_count),
            'ERC/pq4_changed_gain_numerator': (
                (hard_gain * changed).sum() / valid_count),
            'ERC/pq4_beneficial_change_rate': (
                beneficial.float().sum() / valid_count),
            'ERC/pq4_harmful_change_rate': (
                harmful.float().sum() / valid_count),
        }
        if dataset_ids is not None:
            ids = dataset_ids.reshape(-1).to(valid.device)
            for dataset_id in ids[valid].unique(sorted=True):
                mask = valid & ids.eq(dataset_id)
                count = mask.float().sum().clamp_min(1.0)
                suffix = str(int(dataset_id.item()))
                status['ERC/pq4_domain_' + suffix + '_response_gain'] = (
                    (hard_gain * mask).sum() / count)
                status['ERC/pq4_domain_' + suffix + '_harmful_rate'] = (
                    harmful[mask].float().mean())
    return total, status

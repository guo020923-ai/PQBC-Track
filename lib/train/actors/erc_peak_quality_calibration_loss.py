"""Response-only supervision for peak-quality calibrated ERC (PQ-ERC).

The target assigned to each branch is the official centre heatmap value at
that branch's own Hann-aware response peak.  No bbox geometry, candidate IoU,
action search or dataset identity is exposed to the inference module.
"""

import torch
from torch.nn import functional as F


def _masked_mean(values, mask, zero):
    if mask.any():
        return values[mask].float().mean()
    return zero


def _domain_robust_mean(values, valid, dataset_ids, worst_weight, zero):
    """Give each source domain equal weight and retain a worst-domain term."""
    if not valid.any():
        return zero
    if dataset_ids is None:
        return values[valid].float().mean()
    dataset_ids = dataset_ids.reshape(-1).to(values.device)
    if dataset_ids.numel() != values.numel():
        raise ValueError('PQ-ERC dataset_id batch shape mismatch')
    groups = []
    for dataset_id in dataset_ids[valid].unique(sorted=True):
        selected = valid & dataset_ids.eq(dataset_id)
        if selected.any():
            groups.append(values[selected].float().mean())
    if not groups:
        return zero
    groups = torch.stack(groups)
    weight = max(float(worst_weight), 0.0)
    return (groups.mean() + weight * groups.max()) / (1.0 + weight)


def compute_peak_quality_calibration_loss_and_metrics(
        pred, gt_heatmap, cfg, dataset_ids=None):
    reliability = pred['erc_pq_reliability']
    zero = reliability.sum() * 0.0
    active = pred['erc_active_mask'].reshape(-1).bool()
    support = pred['erc_pq_support_mask'].detach().flatten(1).bool()
    target = gt_heatmap.detach().flatten(1).to(
        device=reliability.device, dtype=reliability.dtype).clamp_min(0.0)
    target = target / target.amax(-1, keepdim=True).clamp_min(1e-6)
    valid = active & support.any(-1) & target.amax(-1).gt(0.0)
    if not valid.any():
        return zero, {
            'Loss/erc_total': zero.detach(),
            'ERC/pq_valid_fraction': valid.float().mean(),
        }

    cls_index = pred['erc_pq_cls_peak_index'].detach().long()
    cont_index = pred['erc_pq_cont_peak_index'].detach().long()
    cls_target = target.gather(1, cls_index[:, None]).squeeze(1)
    cont_target = target.gather(1, cont_index[:, None]).squeeze(1)
    quality_target = torch.stack([cls_target, cont_target], dim=-1)

    quality_element = F.smooth_l1_loss(
        reliability, quality_target, reduction='none').mean(-1)
    calibration_element = F.binary_cross_entropy(
        reliability.clamp(1e-6, 1.0 - 1e-6),
        quality_target, reduction='none').mean(-1)

    eps = max(float(cfg.MODEL.HEAD.ERC.EPS), 1e-12)
    base_log = pred['erc_windowed_base_score'].detach().clamp_min(eps).log()
    raw_correction = pred['erc_pq_raw_correction'].flatten(1)
    calibrated_log = base_log + raw_correction
    cls_log = calibrated_log.gather(1, cls_index[:, None]).squeeze(1)
    cont_log = calibrated_log.gather(1, cont_index[:, None]).squeeze(1)
    target_gap = cls_target - cont_target
    tolerance = float(cfg.TRAIN.ERC_PQ_PRESERVE_TOLERANCE)
    informative = valid & target_gap.abs().gt(tolerance)
    preserve = valid & ~informative
    signed_margin = target_gap.sign() * (cls_log - cont_log)
    fusion_margin = float(cfg.TRAIN.ERC_PQ_FUSION_MARGIN)
    fusion_temperature = max(
        float(cfg.TRAIN.ERC_PQ_FUSION_TEMPERATURE), 1e-6)
    fusion_element = F.softplus(
        (fusion_margin - signed_margin) / fusion_temperature
    ) * fusion_temperature * target_gap.abs().detach()
    preserve_element = pred['erc_pq_raw_alpha'].abs()

    worst_weight = float(cfg.TRAIN.ERC_PQ_DOMAIN_WORST_WEIGHT)
    reduce = lambda values, mask: _domain_robust_mean(
        values, mask, dataset_ids, worst_weight, zero)
    quality_loss = reduce(quality_element, valid)
    calibration_loss = reduce(calibration_element, valid)
    fusion_loss = reduce(fusion_element, informative)
    preserve_loss = reduce(preserve_element, preserve)
    total = (
        float(cfg.TRAIN.ERC_PQ_QUALITY_WEIGHT) * quality_loss
        + float(cfg.TRAIN.ERC_PQ_CALIBRATION_WEIGHT) * calibration_loss
        + float(cfg.TRAIN.ERC_PQ_FUSION_WEIGHT) * fusion_loss
        + float(cfg.TRAIN.ERC_PQ_PRESERVE_WEIGHT) * preserve_loss
    )

    with torch.no_grad():
        base_index = pred['erc_windowed_base_score'].detach().argmax(-1)
        full_score = (
            pred['erc_windowed_base_score'].detach()
            * pred['erc_pq_raw_correction'].detach().flatten(1).exp())
        full_index = full_score.argmax(-1)
        changed = valid & base_index.ne(full_index)
        base_target = target.gather(1, base_index[:, None]).squeeze(1)
        full_target = target.gather(1, full_index[:, None]).squeeze(1)
        response_gain = full_target - base_target
        beneficial = changed & response_gain.gt(1e-8)
        harmful = changed & response_gain.lt(-1e-8)
        predicted_direction = pred['erc_pq_raw_alpha'].detach().sign()
        target_direction = target_gap.sign()

        def mean(values, mask=valid):
            return _masked_mean(values, mask, zero.detach())

        status = {
            'Loss/erc_total': total.detach(),
            'Loss/erc_pq_quality': quality_loss.detach(),
            'Loss/erc_pq_calibration': calibration_loss.detach(),
            'Loss/erc_pq_fusion': fusion_loss.detach(),
            'Loss/erc_pq_preserve': preserve_loss.detach(),
            'ERC/pq_valid_fraction': valid.float().mean(),
            'ERC/pq_informative_fraction': mean(informative.float()),
            'ERC/pq_preserve_fraction': mean(preserve.float()),
            'ERC/pq_cls_target_mean': mean(cls_target),
            'ERC/pq_cont_target_mean': mean(cont_target),
            'ERC/pq_cls_reliability_mean': mean(reliability[:, 0]),
            'ERC/pq_cont_reliability_mean': mean(reliability[:, 1]),
            'ERC/pq_quality_mae': mean(
                (reliability - quality_target).abs().mean(-1)),
            'ERC/pq_direction_accuracy': mean(
                predicted_direction.eq(target_direction).float(), informative),
            'ERC/pq_disagreement_mean': mean(pred['erc_pq_disagreement']),
            'ERC/pq_peak_distance_mean': mean(pred['erc_pq_peak_distance']),
            'ERC/pq_js_divergence_mean': mean(
                pred['erc_pq_distribution_disagreement']),
            'ERC/pq_alpha_mean': mean(pred['erc_pq_raw_alpha']),
            'ERC/pq_alpha_abs': mean(pred['erc_pq_raw_alpha'].abs()),
            'ERC/pq_alpha_std': pred['erc_pq_raw_alpha'][valid].std(
                unbiased=False),
            'ERC/pq_alpha_saturation_rate': mean((
                pred['erc_pq_raw_alpha'].abs()
                >= 0.95 * float(cfg.MODEL.HEAD.ERC.MAX_ALPHA)).float()),
            'ERC/pq_selection_change_rate': mean(changed.float()),
            'ERC/pq_response_gain': mean(response_gain),
            'ERC/pq_changed_response_gain': mean(response_gain, changed),
            'ERC/pq_beneficial_change_rate': mean(beneficial.float()),
            'ERC/pq_harmful_change_rate': mean(harmful.float()),
        }
        if dataset_ids is not None:
            ids = dataset_ids.reshape(-1).to(valid.device)
            for dataset_id in ids[valid].unique(sorted=True):
                mask = valid & ids.eq(dataset_id)
                suffix = str(int(dataset_id.item()))
                status['ERC/pq_domain_' + suffix + '_response_gain'] = mean(
                    response_gain, mask)
                status['ERC/pq_domain_' + suffix + '_harmful_rate'] = mean(
                    harmful.float(), mask)
    return total, status

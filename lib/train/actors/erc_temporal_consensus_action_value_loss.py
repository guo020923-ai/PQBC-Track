"""Cross-frame robust utility supervision for temporal ERC-AV2.

The two search views in one training sample belong to the same sequence.  We
retain each frame's local oracle but blend it with a risk-adjusted utility
shared by the views.  An action is therefore rewarded most when it is useful
across target appearances, rather than only at one sampled instant.
"""

import torch
from torch.nn import functional as F

from .erc_branch_aligned_action_value_loss import (
    build_action_utility_targets,
)
from .erc_branch_aligned_action_value_balanced_loss import (
    compute_balanced_action_value_loss_and_metrics,
)


@torch.no_grad()
def build_temporal_consensus_targets(
        pred, gt_heatmap, gt_bbox, cfg, num_search_frames):
    local = build_action_utility_targets(pred, gt_heatmap, gt_bbox, cfg)
    total_rows, action_count = local['raw_utility'].shape
    frame_count = int(num_search_frames)
    metadata = {
        'frame_count': frame_count,
        'pair_valid': None,
        'local_target_action': None,
        'temporal_target_action': None,
        'stable_local_target': None,
        'target_temporal_std': None,
    }
    if frame_count < 2 or total_rows % frame_count != 0:
        return local, metadata

    batch = total_rows // frame_count
    raw = local['raw_utility'].reshape(
        frame_count, batch, action_count)
    valid = local['valid'].reshape(frame_count, batch)
    pair_valid = valid.all(dim=0)

    mean_utility = raw.mean(dim=0)
    temporal_std = raw.std(dim=0, unbiased=False)
    robust_utility = mean_utility - (
        float(cfg.TRAIN.ERC_AV2_TEMPORAL_RISK_WEIGHT) * temporal_std)
    robust_utility[:, 0] = 0.0
    consensus_mix = min(max(
        float(cfg.TRAIN.ERC_AV2_CONSENSUS_MIX), 0.0), 1.0)
    blended = (
        (1.0 - consensus_mix) * raw
        + consensus_mix * robust_utility.unsqueeze(0)
    )
    temporal_raw = torch.where(
        pair_valid[None, :, None], blended, raw)
    temporal_raw[:, :, 0] = 0.0
    temporal_raw = temporal_raw.reshape(total_rows, action_count)
    temporal_raw = torch.where(
        local['valid'][:, None], temporal_raw,
        torch.zeros_like(temporal_raw),
    )

    best_utility, best_action = temporal_raw.max(dim=-1)
    actionable = local['valid'] & best_utility.gt(0.0)
    target_action = torch.where(
        actionable, best_action, torch.zeros_like(best_action))
    utility_scale = max(float(cfg.TRAIN.ERC_AV1_UTILITY_SCALE), 1e-6)
    utility_target = (temporal_raw / utility_scale).clamp(-1.0, 1.0)
    rows = torch.arange(total_rows, device=temporal_raw.device)

    temporal = dict(local)
    temporal.update({
        'actionable': actionable,
        'target_action': target_action,
        'raw_utility': temporal_raw,
        'utility_target': utility_target,
        'target_hard_iou_gain': local['hard_iou_gain'][rows, target_action],
        'target_soft_iou_gain': local['soft_iou_gain'][rows, target_action],
        'target_utility': temporal_raw[rows, target_action],
    })

    local_action = local['target_action'].reshape(frame_count, batch)
    temporal_action = target_action.reshape(frame_count, batch)
    metadata.update({
        'pair_valid': pair_valid,
        'local_target_action': local_action,
        'temporal_target_action': temporal_action,
        'stable_local_target': local_action.eq(local_action[:1]).all(dim=0),
        'target_temporal_std': temporal_std[:, 1:].mean(dim=-1),
    })
    return temporal, metadata


def compute_temporal_consensus_action_value_loss_and_metrics(
        pred, gt_heatmap, gt_bbox, cfg, num_search_frames=1):
    oracle, metadata = build_temporal_consensus_targets(
        pred, gt_heatmap, gt_bbox, cfg, num_search_frames)
    base_loss, status = compute_balanced_action_value_loss_and_metrics(
        pred, gt_heatmap, gt_bbox, cfg, oracle=oracle)
    predicted = pred['erc_av1_action_utility']
    zero = predicted.sum() * 0.0
    frame_count = metadata['frame_count']
    if (
        frame_count < 2
        or predicted.shape[0] % frame_count != 0
        or metadata['pair_valid'] is None
    ):
        status.update({
            'Loss/erc_av2_temporal_consistency': zero.detach(),
            'ERC/av2_pair_valid_rate': zero.detach(),
            'ERC/av2_stable_target_rate': zero.detach(),
            'ERC/av2_local_target_switch_rate': zero.detach(),
            'ERC/av2_temporal_target_switch_rate': zero.detach(),
            'ERC/av2_pred_action_switch_rate': zero.detach(),
            'ERC/av2_target_utility_temporal_std': zero.detach(),
            'ERC/av2_pred_utility_temporal_std': zero.detach(),
        })
        return base_loss, status

    batch = predicted.shape[0] // frame_count
    predicted_view = predicted.reshape(
        frame_count, batch, predicted.shape[-1])
    predicted_mean = predicted_view.mean(dim=0, keepdim=True)
    consistency_error = F.smooth_l1_loss(
        predicted_view[:, :, 1:],
        predicted_mean[:, :, 1:].expand_as(predicted_view[:, :, 1:]),
        reduction='none',
    ).mean(dim=(0, 2))
    pair_valid = metadata['pair_valid']
    stable = pair_valid & metadata['stable_local_target']
    if stable.any():
        consistency_loss = consistency_error[stable].mean()
    else:
        consistency_loss = zero
    total = base_loss + (
        float(cfg.TRAIN.ERC_AV2_TEMPORAL_CONSISTENCY_WEIGHT)
        * consistency_loss
    )

    with torch.no_grad():
        local_action = metadata['local_target_action']
        temporal_action = metadata['temporal_target_action']
        pred_best, pred_nonkeep = predicted_view[:, :, 1:].max(dim=-1)
        pred_action = torch.where(
            pred_best.gt(0.0), pred_nonkeep + 1,
            torch.zeros_like(pred_nonkeep),
        )

        def switch_rate(actions):
            switched = ~actions.eq(actions[:1]).all(dim=0)
            if pair_valid.any():
                return switched[pair_valid].float().mean()
            return zero.detach()

        def pair_mean(values):
            if pair_valid.any():
                return values[pair_valid].float().mean()
            return zero.detach()

        status.update({
            'Loss/erc_total': total.detach(),
            'Loss/erc_av2_temporal_consistency': consistency_loss.detach(),
            'ERC/av2_pair_valid_rate': pair_valid.float().mean(),
            'ERC/av2_stable_target_rate': pair_mean(
                metadata['stable_local_target']),
            'ERC/av2_local_target_switch_rate': switch_rate(local_action),
            'ERC/av2_temporal_target_switch_rate': switch_rate(
                temporal_action),
            'ERC/av2_pred_action_switch_rate': switch_rate(pred_action),
            'ERC/av2_target_utility_temporal_std': pair_mean(
                metadata['target_temporal_std']),
            'ERC/av2_pred_utility_temporal_std': pair_mean(
                predicted_view[:, :, 1:].std(
                    dim=0, unbiased=False).mean(dim=-1)),
        })
        if 'erc_av3_action_family' in pred:
            family = pred['erc_av3_action_family'].detach().long()
            valid = oracle['valid']
            actionable = oracle['actionable']
            predicted_action = pred['erc_av1_raw_hard_action'].detach().long()
            target_family = family[oracle['target_action']]
            predicted_family = family[predicted_action]

            def valid_rate(mask):
                if valid.any():
                    return mask[valid].float().mean()
                return zero.detach()

            def actionable_rate(mask):
                if actionable.any():
                    return mask[actionable].float().mean()
                return zero.detach()

            status.update({
                'ERC/av3_oracle_global_rate': actionable_rate(
                    target_family.eq(1)),
                'ERC/av3_oracle_regional_rate': actionable_rate(
                    target_family.eq(2)),
                'ERC/av3_oracle_veto_rate': actionable_rate(
                    target_family.eq(3)),
                'ERC/av3_pred_keep_rate': valid_rate(
                    predicted_family.eq(0)),
                'ERC/av3_pred_global_rate': valid_rate(
                    predicted_family.eq(1)),
                'ERC/av3_pred_regional_rate': valid_rate(
                    predicted_family.eq(2)),
                'ERC/av3_pred_veto_rate': valid_rate(
                    predicted_family.eq(3)),
                'ERC/av3_reference_agreement': pred[
                    'erc_av3_reference_agreement'].detach().mean(),
                'ERC/av3_effective_strength': pred[
                    'erc_av3_effective_strength'].detach()[valid].mean(),
            })
    return total, status

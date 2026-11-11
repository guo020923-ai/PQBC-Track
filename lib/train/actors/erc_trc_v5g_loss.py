"""Branch-conditional permission supervision for TRC-v5g."""

import torch

from lib.train.actors.erc_trc_v4_loss import (
    hard_decision_action_oracle,
    masked_average,
)
from lib.train.actors.erc_trc_v5_loss import permission_losses
from lib.train.actors.erc_trc_v5c_loss import permission_boundary_losses
from lib.train.actors.erc_trc_v5d_loss import compute_trc_v5d_loss_and_metrics


def compute_trc_v5g_loss_and_metrics(pred, gt_heatmap, gt_bbox, cfg):
    base_loss, status = compute_trc_v5d_loss_and_metrics(
        pred, gt_heatmap, gt_bbox, cfg
    )
    packet = pred['erc_trc_output']
    oracle = hard_decision_action_oracle(
        packet, gt_heatmap,
        cfg.TRAIN.ERC_TRC_V5D_ORACLE_STEPS,
        cfg.TRAIN.ERC_TRC_V5D_SCORE_TEMPERATURE,
        cfg.TRAIN.ERC_TRC_V5D_MIN_TARGET_GAIN,
    )
    valid = oracle['valid']
    target_action = oracle['target_action']
    logits = packet['branch_permission_logits']
    probability = packet['branch_permission_probability']
    calibration_losses, ranking_losses = [], []
    positive_boundaries, negative_boundaries = [], []
    for branch in range(2):
        target = target_action.eq(branch + 1).to(logits)
        calibration, ranking = permission_losses(
            logits[:, branch], target, valid,
            cfg.TRAIN.ERC_TRC_V5G_BRANCH_PERMISSION_RANK_MARGIN,
        )
        positive, negative = permission_boundary_losses(
            logits[:, branch], target, valid,
            packet['branch_permission_threshold'],
            packet['permission_temperature'],
            cfg.TRAIN.ERC_TRC_V5G_BRANCH_PERMISSION_BOUNDARY_MARGIN,
            cfg.TRAIN.ERC_TRC_V5G_BRANCH_PERMISSION_BOUNDARY_TEMPERATURE,
        )
        calibration_losses.append(calibration)
        ranking_losses.append(ranking)
        positive_boundaries.append(positive)
        negative_boundaries.append(negative)
    calibration_loss = torch.stack(calibration_losses).mean()
    ranking_loss = torch.stack(ranking_losses).mean()
    positive_boundary_loss = torch.stack(positive_boundaries).mean()
    negative_boundary_loss = torch.stack(negative_boundaries).mean()
    branch_loss = (
        cfg.TRAIN.ERC_TRC_V5G_BRANCH_PERMISSION_WEIGHT * calibration_loss
        + cfg.TRAIN.ERC_TRC_V5G_BRANCH_PERMISSION_RANK_WEIGHT * ranking_loss
        + cfg.TRAIN.ERC_TRC_V5G_BRANCH_PERMISSION_POS_BOUNDARY_WEIGHT
        * positive_boundary_loss
        + cfg.TRAIN.ERC_TRC_V5G_BRANCH_PERMISSION_NEG_BOUNDARY_WEIGHT
        * negative_boundary_loss
    )
    loss = base_loss + branch_loss
    status.update({
        'Loss/erc_trc_v5g_branch_permission': calibration_loss.detach(),
        'Loss/erc_trc_v5g_branch_permission_rank': ranking_loss.detach(),
        'Loss/erc_trc_v5g_branch_permission_positive_boundary':
            positive_boundary_loss.detach(),
        'Loss/erc_trc_v5g_branch_permission_negative_boundary':
            negative_boundary_loss.detach(),
        'Loss/erc_trc_v5g_branch_total': branch_loss.detach(),
        'Loss/erc_total': loss.detach(),
    })

    with torch.no_grad():
        zero = loss.detach().new_zeros(())

        def mean(value, mask):
            return value[mask].mean() if mask.any() else zero

        for branch, name in ((0, 'cls'), (1, 'cont')):
            positive = valid & target_action.eq(branch + 1)
            negative = valid & ~target_action.eq(branch + 1)
            hard = probability[:, branch].gt(
                packet['branch_permission_threshold']
            )
            predicted = hard & valid
            true_positive = hard & positive
            precision = (
                true_positive.sum().float()
                / predicted.sum().clamp_min(1).float()
            )
            status.update({
                f'ERC/branch_{name}_permission_mean': mean(
                    probability[:, branch], valid),
                f'ERC/branch_{name}_permission_positive_mean': mean(
                    probability[:, branch], positive),
                f'ERC/branch_{name}_permission_negative_mean': mean(
                    probability[:, branch], negative),
                f'ERC/branch_{name}_permission_hard_fraction': mean(
                    hard.float(), valid),
                f'ERC/branch_{name}_permission_precision': precision,
                f'ERC/branch_{name}_permission_true_positive_rate': mean(
                    hard.float(), positive),
                f'ERC/branch_{name}_permission_false_positive_rate': mean(
                    hard.float(), negative),
            })
        status['ERC/combined_permission_fraction'] = mean(
            packet['hard_permission'], packet['active']
        )
    return loss, status

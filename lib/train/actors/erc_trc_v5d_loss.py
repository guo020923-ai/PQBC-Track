"""Balanced gate and conflict-free factorized supervision for TRC-v5d.

The deployed policy is unchanged from v5c.  Only the training graph differs:
the response decision surrogate uses the detached Oracle direction, and risk
uses detached direction probabilities.  Consequently those auxiliary losses
train permission/strength without overwhelming the rare T-direction labels.
"""

from types import SimpleNamespace

import torch

from lib.train.actors.erc_trc_v4_loss import hard_decision_action_oracle
from lib.train.actors.erc_trc_v5c_loss import compute_trc_v5c_loss_and_metrics


_V5D_FIELDS = (
    'PERMISSION_WEIGHT',
    'PERMISSION_RANK_WEIGHT',
    'PERMISSION_RANK_MARGIN',
    'PERMISSION_POS_BOUNDARY_WEIGHT',
    'PERMISSION_NEG_BOUNDARY_WEIGHT',
    'PERMISSION_BOUNDARY_MARGIN',
    'PERMISSION_BOUNDARY_TEMPERATURE',
    'DIRECTION_WEIGHT',
    'DIRECTION_CONT_WEIGHT',
    'DIRECTION_MARGIN_WEIGHT',
    'DIRECTION_MARGIN',
    'STRENGTH_WEIGHT',
    'DECISION_WEIGHT',
    'PRESERVE_WEIGHT',
    'RISK_WEIGHT',
    'ORACLE_STEPS',
    'SCORE_TEMPERATURE',
    'MIN_TARGET_GAIN',
    'DECISION_MARGIN',
    'MARGIN_TEMPERATURE',
)


def _v5c_cfg_view(cfg):
    """Present v5d settings to the audited v5c loss implementation."""
    values = {
        f'ERC_TRC_V5C_{field}': getattr(
            cfg.TRAIN, f'ERC_TRC_V5D_{field}'
        )
        for field in _V5D_FIELDS
    }
    return SimpleNamespace(TRAIN=SimpleNamespace(**values))


def teacher_decoupled_packet(packet, oracle):
    """Build loss-only soft actions without changing hard deployment output."""
    target_action = oracle['target_action']
    corrective = oracle['valid'] & target_action.ne(0)
    predicted_direction = (
        packet['direction_probability'][:, 0]
        - packet['direction_probability'][:, 1]
    ).detach()
    oracle_direction = torch.where(
        target_action.eq(1),
        predicted_direction.new_ones(predicted_direction.shape),
        -predicted_direction.new_ones(predicted_direction.shape),
    )
    loss_direction = torch.where(
        corrective, oracle_direction, predicted_direction
    )
    soft_alpha_scalar = (
        float(packet['alpha_limit'])
        * packet['permission_probability']
        * loss_direction
        * packet['strength']
    )
    soft_alpha_scalar = torch.where(
        packet['active'], soft_alpha_scalar,
        torch.zeros_like(soft_alpha_scalar),
    )
    detached_direction_probability = packet['direction_probability'].detach()
    action_probability = torch.stack([
        1.0 - packet['permission_probability'],
        packet['permission_probability']
        * detached_direction_probability[:, 0],
        packet['permission_probability']
        * detached_direction_probability[:, 1],
    ], -1)
    result = dict(packet)
    result['soft_alpha_scalar'] = soft_alpha_scalar
    result['soft_alpha'] = soft_alpha_scalar[:, None, None].expand_as(
        packet['soft_alpha']
    )
    result['action_probability'] = action_probability
    return result, corrective


def compute_trc_v5d_loss_and_metrics(pred, gt_heatmap, gt_bbox, cfg):
    packet = pred['erc_trc_output']
    oracle = hard_decision_action_oracle(
        packet, gt_heatmap,
        cfg.TRAIN.ERC_TRC_V5D_ORACLE_STEPS,
        cfg.TRAIN.ERC_TRC_V5D_SCORE_TEMPERATURE,
        cfg.TRAIN.ERC_TRC_V5D_MIN_TARGET_GAIN,
    )
    loss_packet, corrective = teacher_decoupled_packet(packet, oracle)
    loss_pred = dict(pred)
    loss_pred['erc_trc_output'] = loss_packet
    loss, status = compute_trc_v5c_loss_and_metrics(
        loss_pred, gt_heatmap, gt_bbox, _v5c_cfg_view(cfg)
    )
    status = {
        key.replace('Loss/erc_trc_v5c_', 'Loss/erc_trc_v5d_'): value
        for key, value in status.items()
    }
    status['ERC/teacher_forced_decision_fraction'] = corrective.float().mean()
    return loss, status

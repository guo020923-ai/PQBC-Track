"""Reuse the exact fixed RQC30 action/writer geometry, not an alternative policy."""
from change.erc_rqc30_dense_curve_20261020_01.geometry import (
    action_lattice, apply_size, candidates, overlap, action_targets,
)
from change.erc_rqc30_dense_curve_20261020_01.model import supported_actions
import torch


def xyxy(box):
    return torch.cat((box[..., :2], box[..., :2] + box[..., 2:]), -1)


def normalized_boxes(boxes, roi):
    start, size = roi[:, None, :2], roi[:, None, 2:]
    return torch.cat(((boxes[..., :2] - start) / size,
                      (boxes[..., :2] + boxes[..., 2:] - start) / size), -1)


def box_iou(a, b, generalized=False):
    """Broadcasting xyxy IoU; prediction gradient stays attached."""
    inter = (torch.minimum(a[..., 2:], b[..., 2:]) -
             torch.maximum(a[..., :2], b[..., :2])).clamp_min(0).prod(-1)
    aa = (a[..., 2:] - a[..., :2]).clamp_min(0).prod(-1)
    bb = (b[..., 2:] - b[..., :2]).clamp_min(0).prod(-1)
    union = (aa + bb - inter).clamp_min(1e-12)
    iou = inter / union
    if not generalized:
        return iou
    area = (torch.maximum(a[..., 2:], b[..., 2:]) -
            torch.minimum(a[..., :2], b[..., :2])).clamp_min(0).prod(-1).clamp_min(1e-12)
    return iou - (area - union) / area

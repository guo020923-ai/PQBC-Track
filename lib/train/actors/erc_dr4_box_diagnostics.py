"""Detached, same-crop diagnostics. These are NOT full-image/trajectory AUC.

Do not clamp boxes to the search crop: tracker maps boxes to the image before
clipping to image boundaries. Image dimensions/history are unavailable here.
"""
import numpy as np
import torch


@torch.no_grad()
def deployed_box_diagnostics(pred, gt_xywh, heatmap, valid, loss_index):
    c = pred['erc_cls_response'].detach().flatten(1)
    t = pred['erc_cont_response'].detach().flatten(1)
    n = c.shape[-1]
    size = int(n ** .5)
    if size * size != n:
        raise ValueError('Expected a square response map')
    h = np.hanning(size)
    window = torch.as_tensor(np.outer(h, h).reshape(1, -1), device=c.device)
    baseline = c.double() * window * t.double()
    score = baseline * pred['erc_factor'].detach().reshape_as(c).double()
    bi, fi = baseline.argmax(-1), score.argmax(-1)
    row = torch.arange(c.shape[0], device=c.device)
    boxes = pred['bbox_map'].detach().double()
    gt = gt_xywh.detach().double()
    if boxes.shape != (c.shape[0], n, 4) or gt.shape != (c.shape[0], 4):
        raise ValueError('Box/response batch alignment mismatch')
    lo, hi = boxes[..., :2] - boxes[..., 2:] / 2, boxes[..., :2] + boxes[..., 2:] / 2
    glo, ghi = gt[:, None, :2], (gt[:, :2] + gt[:, 2:])[:, None]
    inter = (torch.minimum(hi, ghi) - torch.maximum(lo, glo)).clamp_min(0).prod(-1)
    union = boxes[..., 2:].clamp_min(0).prod(-1) + gt[:, None, 2:].clamp_min(0).prod(-1) - inter
    iou = inter / union.clamp_min(1e-12)
    gain = iou[row, fi] - iou[row, bi]
    center = gt[:, :2] + gt[:, 2:] / 2
    error = (boxes[..., :2] - center[:, None]).norm(dim=-1)
    center_gain = error[row, bi] - error[row, fi]
    response = heatmap.detach().flatten(1).to(gain)
    response_gain = response[row, fi] - response[row, bi]
    changed = valid & fi.ne(bi)
    masks = {
        'valid': valid, 'changed': changed,
        'beneficial': changed & gain.gt(1e-6),
        'harmful': changed & gain.lt(-1e-6),
        'response_positive_iou_negative': changed & response_gain.gt(1e-8) & gain.lt(-1e-6),
        'loss_selection_mismatch': valid & fi.ne(loss_index),
        'base_selection_mismatch': valid & bi.ne(pred['erc_windowed_base_score'].argmax(-1)),
    }
    result = {}
    prefix = 'ERC/dr4_box_'
    count = valid.sum().clamp_min(1)
    for name, mask in masks.items():
        result[prefix + name + '_count'] = mask.sum().float()
        result[prefix + name + '_rate'] = mask.sum() / count
    result[prefix + 'iou_gain_sum'] = gain[valid].sum()
    result[prefix + 'iou_gain'] = gain[valid].sum() / count
    result[prefix + 'changed_iou_gain_sum'] = gain[changed].sum()
    result[prefix + 'changed_iou_gain'] = gain[changed].sum() / changed.sum().clamp_min(1)
    result[prefix + 'center_error_gain_sum'] = center_gain[valid].sum()
    result[prefix + 'center_error_gain'] = center_gain[valid].sum() / count
    return result

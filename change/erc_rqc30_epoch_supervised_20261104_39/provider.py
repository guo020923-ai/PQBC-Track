"""Self-contained frozen-RQC30 25-action evidence provider."""

import torch
from torch.nn import functional as F


def _visual_maps(inputs):
    search = inputs["search"].detach().float()
    head = inputs["head"].detach().float()
    mask = inputs["source_mask"].detach().bool()
    roles = inputs["template_roles"].detach().float()
    search_tokens = search.permute(0, 2, 3, 1)
    similarity = torch.einsum(
        "bhwc,brc->brhw", F.normalize(search_tokens, dim=-1),
        F.normalize(roles, dim=-1),
    )
    maps = torch.cat((
        head.mean(1, keepdim=True),
        head.std(1, keepdim=True, unbiased=False),
        head.square().mean(1, keepdim=True).sqrt(),
        head.amax(1, keepdim=True),
        similarity,
        search.square().mean(1, keepdim=True).sqrt(),
        inputs["native_extent"].detach().float().mean(1, keepdim=True),
    ), 1)
    if maps.shape[1] != 8:
        raise RuntimeError("fixed eight-map RQC30 evidence contract changed")
    return torch.where(mask, maps, torch.zeros_like(maps))


def _five_points(boxes):
    center = boxes[..., :2] + 0.5 * boxes[..., 2:]
    half = 0.5 * boxes[..., 2:]
    offsets = boxes.new_tensor([
        [0.0, 0.0], [-1.0, 0.0], [1.0, 0.0],
        [0.0, -1.0], [0.0, 1.0],
    ])
    return center[:, :, None] + offsets[None, None] * half[:, :, None]


@torch.no_grad()
def causal_candidate_provider(source, image, fixed_output, previous_state):
    del previous_state
    expected_stamp = (
        int(source.frame_id), tuple(fixed_output["target_bbox"]),
        tuple(image.shape[:2]),
    )
    if getattr(source, "_light_stamp", None) != expected_stamp:
        raise ValueError("stale or mismatched actual RQC30 evidence")
    inputs = source.last_inputs
    required = {
        "search", "head", "template_roles", "native_extent", "source_mask",
        "source_context", "candidate_boxes", "crop_boxes",
        "saved_crop_boxes", "valid",
    }
    if set(inputs) != required:
        raise ValueError("actual RQC30 lightweight schema changed")
    boxes = inputs["candidate_boxes"].detach().double()
    if boxes.shape[1:] != (25, 4):
        raise ValueError("fixed RQC30 action lattice must contain 25 boxes")
    anchor = torch.zeros(len(boxes), dtype=torch.long, device=boxes.device)
    actual = boxes.new_tensor([fixed_output["target_bbox"]])
    if not torch.allclose(boxes[:, 0], actual, rtol=0, atol=1e-8):
        raise RuntimeError("action zero no longer reproduces actual RQC30 output")
    maps = _visual_maps(inputs)
    points = _five_points(inputs["crop_boxes"].detach().float())
    grid = points.reshape(len(boxes), 25 * 5, 1, 2) * 2 - 1
    sampled = F.grid_sample(
        maps, grid, align_corners=False, padding_mode="zeros"
    ).reshape(len(boxes), 8, 25, 5).permute(0, 2, 3, 1).flatten(2)
    anchor_box = boxes[:, :1]
    relative = torch.cat((
        (boxes[..., :2] - anchor_box[..., :2])
        / anchor_box[..., 2:].clamp_min(1e-6),
        (boxes[..., 2:] / anchor_box[..., 2:].clamp_min(1e-6)).log(),
    ), -1).float().clamp(-10, 10)
    valid = inputs["valid"].detach().bool()
    valid_fraction = valid.float().flatten(2).mean(-1, keepdim=True)
    crop_area = inputs["crop_boxes"][..., 2:].detach().float().prod(
        -1, keepdim=True
    ).clamp_min(1e-12).log()
    action_delta = relative[..., 2:]
    evidence = torch.cat((
        sampled, relative, valid_fraction, crop_area, action_delta
    ), -1)
    if evidence.shape != (len(boxes), 25, 48) or not torch.isfinite(evidence).all():
        raise RuntimeError("fixed 48-D actual RQC30 evidence contract changed")
    count = valid.sum(-1)
    axis_supported = count.ge(3).all(-1).any(-1)
    executable = (action_delta.eq(0) | axis_supported).all(-1)
    executable[:, 0] = True
    return {
        "boxes": boxes,
        "evidence": evidence,
        "executable": executable,
        "anchor_index": anchor,
    }

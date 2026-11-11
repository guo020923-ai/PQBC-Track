"""TRAIN/deployment-identical quarter-action basis."""

import torch


def quarter_mapping(evidence, executable):
    if evidence.shape[1:] != (25, 48) or executable.shape != evidence.shape[:2]:
        raise ValueError("action schema")
    relative = evidence[..., 40:44]
    target = relative * 0.25
    distance = (relative[:, None, :, :] - target[:, :, None, :]).square().sum(-1)
    distance = distance.masked_fill(~executable[:, None, :], torch.inf)
    mapped = distance.argmin(-1)
    mapped[:, 0] = 0
    return mapped


def execution_mask(evidence, executable):
    """Return exactly the action identities reachable after quarter mapping."""
    mapped = quarter_mapping(evidence, executable)
    mask = torch.zeros_like(executable)
    rows = torch.arange(len(mask), device=mask.device)
    for raw in range(25):
        mask[rows, mapped[:, raw]] |= executable[:, raw]
    mask[:, 0] = True
    return mask

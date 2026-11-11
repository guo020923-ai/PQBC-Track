"""Strict numeric export/load for the fixed Epoch-100 joint-utility model."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .config import PROTOCOL
from .model import HarmAwareJointUtilityERC, VERSION


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def export_numeric(checkpoint, directory):
    checkpoint, directory = Path(checkpoint), Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    saved = torch.load(str(checkpoint), map_location="cpu")
    if saved.get("kind") != "rqc30_harm_aware_joint_utility_100ep_v1":
        raise ValueError("wrong training checkpoint kind")
    if saved.get("model_version") != VERSION or saved.get("protocol") != PROTOCOL:
        raise ValueError("training checkpoint identity mismatch")
    if saved.get("epoch") != 100 or saved.get("optimizer_steps") != 187500:
        raise ValueError("predetermined Epoch-100 checkpoint required")
    model = HarmAwareJointUtilityERC()
    model.load_state_dict(saved["model"], strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise FloatingPointError("non-finite fixed model")
    directory.mkdir(parents=False)
    arrays = {
        key: value.detach().cpu().contiguous().numpy()
        for key, value in model.state_dict().items()
    }
    numeric = directory / "erc_epoch100_numeric.npz"
    with numeric.open("xb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    manifest = {
        "format": "numpy_npz_numeric_only_no_pickle",
        "kind": "rqc30_harm_aware_joint_utility_100ep_v1",
        "model_version": VERSION,
        "epoch": 100,
        "optimizer_steps": 187500,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": digest(checkpoint),
        "numeric_npz_sha256": digest(numeric),
        "tensor_keys": sorted(arrays),
        "tensor_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "tensor_dtypes": {key: str(value.dtype) for key, value in arrays.items()},
        "protocol": PROTOCOL,
    }
    with (directory / "manifest.json").open("x", encoding="utf8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    return manifest


def load_numeric(directory, device="cpu"):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    numeric = directory / "erc_epoch100_numeric.npz"
    if manifest.get("format") != "numpy_npz_numeric_only_no_pickle":
        raise ValueError("unexpected checkpoint format")
    if manifest.get("model_version") != VERSION or manifest.get("protocol") != PROTOCOL:
        raise ValueError("numeric checkpoint identity mismatch")
    if digest(numeric) != manifest.get("numeric_npz_sha256"):
        raise ValueError("numeric checkpoint hash mismatch")
    model = HarmAwareJointUtilityERC().to(device)
    expected = model.state_dict()
    with np.load(numeric, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValueError("numeric tensor key mismatch")
        state = {}
        for key, reference in expected.items():
            array = archive[key]
            if list(array.shape) != list(reference.shape):
                raise ValueError("numeric tensor shape mismatch: " + key)
            state[key] = torch.from_numpy(array.copy()).to(
                device=device, dtype=reference.dtype
            )
    model.load_state_dict(state, strict=True)
    return model, manifest

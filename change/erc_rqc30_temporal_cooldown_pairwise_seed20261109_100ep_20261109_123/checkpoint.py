import json
import os
from pathlib import Path

import numpy as np
import torch

from .config import PROTOCOL
from .io_utils import digest
from .model import ExecutionAlignedPairwiseERC, VERSION


def export_numeric(checkpoint, destination):
    checkpoint, destination = Path(checkpoint), Path(destination)
    saved = torch.load(str(checkpoint), map_location="cpu")
    if (saved.get("kind") != VERSION or saved.get("protocol") != PROTOCOL or
            saved.get("epoch") != 100 or saved.get("optimizer_steps") != 187500):
        raise ValueError("checkpoint identity")
    model = ExecutionAlignedPairwiseERC(); model.load_state_dict(saved["model"])
    arrays = {key: value.numpy() for key, value in model.state_dict().items()}
    destination.mkdir()
    numeric = destination / "erc_epoch100_numeric.npz"
    with numeric.open("xb") as handle:
        np.savez(handle, **arrays); handle.flush(); os.fsync(handle.fileno())
    manifest = {
        "format": "numpy_npz_numeric_only_no_pickle", "kind": VERSION,
        "model_version": VERSION, "epoch": 100, "optimizer_steps": 187500,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": digest(checkpoint),
        "cache_manifest_sha256": saved["cache_manifest_sha256"],
        "numeric_npz_sha256": digest(numeric), "tensor_keys": sorted(arrays),
        "protocol": PROTOCOL,
    }
    with (destination / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True); handle.flush(); os.fsync(handle.fileno())
    return manifest


def load_numeric(destination, device="cpu"):
    destination = Path(destination)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    numeric = destination / "erc_epoch100_numeric.npz"
    if (manifest.get("model_version") != VERSION or manifest.get("protocol") != PROTOCOL or
            digest(numeric) != manifest.get("numeric_npz_sha256")):
        raise ValueError("numeric identity")
    model = ExecutionAlignedPairwiseERC().to(device)
    reference, state = model.state_dict(), {}
    with np.load(numeric, allow_pickle=False) as arrays:
        if set(arrays.files) != set(reference):
            raise ValueError("numeric keys")
        for key, value in reference.items():
            state[key] = torch.from_numpy(arrays[key].copy()).to(device=device, dtype=value.dtype)
    model.load_state_dict(state)
    return model, manifest

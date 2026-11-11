"""Thirty-epoch supervised optimization over the fixed causal TRAIN cache."""

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .config import PROTOCOL, validate_protocol
from .io_utils import atomic_json, digest, write_new_json
from .model import RQC30EpochERC, VERSION
from .objective import structured_regret_loss


def _load_cache(cache_dir):
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest["schema"] != "rqc30_epoch_erc_cache_v1":
        raise ValueError("wrong cache schema")
    arrays = {key: [] for key in ("evidence", "executable", "utility", "source")}
    for shard in manifest["shards"]:
        path = cache_dir / shard["name"]
        if digest(path) != shard["sha256"]:
            raise ValueError("cache shard hash changed: " + path.name)
        with np.load(path, allow_pickle=False) as packet:
            for key in arrays:
                arrays[key].append(packet[key])
    arrays = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    if len(arrays["evidence"]) != PROTOCOL["cache_examples"]:
        raise ValueError("cache size mismatch")
    return manifest, arrays


def _save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_epochs(output_dir, resume=None, cache_dir=None):
    validate_protocol()
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else output_dir / "cache"
    manifest, arrays = _load_cache(cache_dir)
    random.seed(PROTOCOL["seed"])
    np.random.seed(PROTOCOL["seed"])
    torch.manual_seed(PROTOCOL["seed"])
    torch.cuda.manual_seed_all(PROTOCOL["seed"])
    device = torch.device("cuda")
    model = RQC30EpochERC().to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PROTOCOL["learning_rate"],
        weight_decay=PROTOCOL["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PROTOCOL["epochs"], eta_min=0.0
    )
    first_epoch = 1
    if resume is not None:
        saved = torch.load(str(resume), map_location="cpu")
        if saved["protocol"] != PROTOCOL or saved["model_version"] != VERSION:
            raise ValueError("resume identity mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        first_epoch = int(saved["epoch"]) + 1
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=resume is not None)
    stats_path = output_dir / "stats.jsonl"
    if resume is None and stats_path.exists():
        raise FileExistsError(stats_path)
    total_steps = PROTOCOL["epochs"] * PROTOCOL["steps_per_epoch"]
    completed_steps = (first_epoch - 1) * PROTOCOL["steps_per_epoch"]
    started = time.monotonic()
    recent = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(PROTOCOL["seed"])
    with stats_path.open("a" if resume else "x", encoding="utf8") as stats_file:
        for epoch in range(first_epoch, PROTOCOL["epochs"] + 1):
            epoch_started = time.monotonic()
            order = torch.randperm(PROTOCOL["cache_examples"], generator=generator).numpy()
            sums = {key: 0.0 for key in (
                "loss", "chosen_regret", "actionable_rate",
                "label_keep_rate", "predicted_keep_rate",
            )}
            for step in range(PROTOCOL["steps_per_epoch"]):
                step_started = time.monotonic()
                indices = order[
                    step * PROTOCOL["batch_size"]:(step + 1) * PROTOCOL["batch_size"]
                ]
                evidence = torch.from_numpy(arrays["evidence"][indices]).to(
                    device=device, dtype=torch.float32
                )
                executable = torch.from_numpy(arrays["executable"][indices]).to(device)
                utility = torch.from_numpy(arrays["utility"][indices]).to(
                    device=device, dtype=torch.float32
                )
                anchor = torch.zeros(len(indices), dtype=torch.long, device=device)
                prediction = model(evidence, executable, anchor)
                loss, row = structured_regret_loss(
                    prediction["relative_score"], executable, anchor, utility
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), PROTOCOL["gradient_clip"]
                )
                if not torch.isfinite(norm):
                    raise FloatingPointError("non-finite gradient")
                optimizer.step()
                completed_steps += 1
                for key in sums:
                    sums[key] += float(row[key])
                recent.append(time.monotonic() - step_started)
                recent = recent[-200:]
                if step == 0 or (step + 1) % 25 == 0:
                    seconds_per_step = sum(recent) / len(recent)
                    low_step = float(np.quantile(recent, 0.2))
                    high_step = float(np.quantile(recent, 0.8))
                    remaining = total_steps - completed_steps
                    atomic_json(output_dir / "progress.json", {
                        "phase": "EPOCH_TRAINING",
                        "epoch": epoch,
                        "total_epochs": PROTOCOL["epochs"],
                        "step_in_epoch": step + 1,
                        "steps_per_epoch": PROTOCOL["steps_per_epoch"],
                        "completed_steps": completed_steps,
                        "total_steps": total_steps,
                        "recent_seconds_per_step": seconds_per_step,
                        "eta_seconds": remaining * seconds_per_step,
                        "eta_seconds_low": remaining * low_step,
                        "eta_seconds_high": remaining * high_step,
                        "eta_scope": "remaining_epoch_training",
                        "updated_unix": time.time(),
                    })
            scheduler.step()
            record = {
                "epoch": epoch,
                "seconds": time.monotonic() - epoch_started,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "stats": {key: value / PROTOCOL["steps_per_epoch"] for key, value in sums.items()},
            }
            stats_file.write(json.dumps(record, allow_nan=False) + "\n")
            stats_file.flush()
            os.fsync(stats_file.fileno())
            _save_checkpoint(checkpoints / ("ERC_epoch_%04d.pth.tar" % epoch), {
                "kind": "rqc30_epoch_supervised_erc_v2_actionable_gain",
                "model_version": VERSION,
                "protocol": PROTOCOL,
                "cache_manifest_sha256": digest(cache_dir / "manifest.json"),
                "epoch": epoch,
                "optimizer_steps": completed_steps,
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            })
    final = checkpoints / "ERC_epoch_0030.pth.tar"
    completed = {
        "status": "TRAIN_COMPLETE_PENDING_READONLY_AUDIT",
        "fixed_epoch": 30,
        "optimizer_steps": total_steps,
        "checkpoint": str(final),
        "checkpoint_sha256": digest(final),
        "training_seconds": time.monotonic() - started,
        "official_auc": None,
        "cache_manifest_sha256": digest(cache_dir / "manifest.json"),
        "automatic_test": False,
    }
    write_new_json(output_dir / "completed.json", completed)
    atomic_json(output_dir / "progress.json", dict(completed, phase="TRAINING_COMPLETE"))
    return completed

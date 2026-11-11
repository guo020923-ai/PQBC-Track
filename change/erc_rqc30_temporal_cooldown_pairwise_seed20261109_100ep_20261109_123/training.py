import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .config import PROTOCOL, calibration_mask, validate_protocol
from .io_utils import atomic_json, digest
from .actions import execution_mask
from .model import ExecutionAlignedPairwiseERC, VERSION
from .objective import pairwise_utility_loss
from .runtime import deterministic_torch


def load_cache(cache_dir):
    from change.erc_rqc30_feedback_pairwise_100ep_20261107_70.training import (
        load_cache as load_source_cache,
    )
    cache_dir = Path(cache_dir)
    if digest(cache_dir / "manifest.json") != PROTOCOL["source_cache_manifest_sha256"]:
        raise ValueError("fixed feedback cache identity")
    return load_source_cache(cache_dir)


def _learning_rate(epoch):
    if epoch <= PROTOCOL["warmup_epochs"]:
        return PROTOCOL["peak_learning_rate"] * epoch / PROTOCOL["warmup_epochs"]
    fraction = (epoch - PROTOCOL["warmup_epochs"]) / (PROTOCOL["epochs"] - PROTOCOL["warmup_epochs"])
    return PROTOCOL["minimum_learning_rate"] + 0.5 * (
        PROTOCOL["peak_learning_rate"] - PROTOCOL["minimum_learning_rate"]
    ) * (1 + math.cos(math.pi * fraction))


def train_epochs(output_dir, cache_dir):
    validate_protocol(); deterministic_torch()
    output_dir, cache_dir = Path(output_dir), Path(cache_dir)
    _, arrays = load_cache(cache_dir)
    held = calibration_mask(np.arange(30000)); training_rows = np.arange(30000)[~held]
    seed = PROTOCOL["training_seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = ExecutionAlignedPairwiseERC().cuda().train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PROTOCOL["peak_learning_rate"],
        weight_decay=PROTOCOL["weight_decay"],
    )
    checkpoints = output_dir / "checkpoints"; checkpoints.mkdir()
    generator = torch.Generator().manual_seed(seed)
    completed_steps = 0; started = time.monotonic(); recent = []
    stat_keys = (
        "loss", "predicted_keep_rate", "harmful_deploy_rate",
        "mean_deployed_utility", "chosen_regret", "pair_accuracy", "mean_score",
    )
    with (output_dir / "stats.jsonl").open("x", encoding="utf-8") as stats_file:
        for epoch in range(1, 101):
            rate = _learning_rate(epoch)
            for group in optimizer.param_groups:
                group["lr"] = rate
            pieces, remaining = [], PROTOCOL["steps_per_epoch"] * PROTOCOL["batch_size"]
            while remaining:
                order = torch.randperm(len(training_rows), generator=generator).numpy()
                take = min(remaining, len(order)); pieces.append(training_rows[order[:take]]); remaining -= take
            order = np.concatenate(pieces); sums = {key: 0.0 for key in stat_keys}
            epoch_started = time.monotonic()
            for step in range(PROTOCOL["steps_per_epoch"]):
                step_started = time.monotonic()
                index = order[step * 16:(step + 1) * 16]
                evidence = torch.from_numpy(arrays["evidence"][index]).cuda().float()
                source_executable = torch.from_numpy(arrays["executable"][index]).cuda().bool()
                utility = torch.from_numpy(arrays["utility"][index]).cuda().float()
                anchor = torch.zeros(len(index), dtype=torch.long, device="cuda")
                executable = execution_mask(evidence, source_executable)
                prediction = model(evidence, executable, anchor)
                loss, row = pairwise_utility_loss(prediction, executable, anchor, utility)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), PROTOCOL["gradient_clip"])
                if not torch.isfinite(norm):
                    raise FloatingPointError("gradient")
                optimizer.step(); completed_steps += 1
                for key in stat_keys:
                    sums[key] += float(row[key])
                recent.append(time.monotonic() - step_started); recent = recent[-200:]
                if step == 0 or (step + 1) % 25 == 0:
                    seconds = sum(recent) / len(recent)
                    atomic_json(output_dir / "progress.json", {
                        "phase": "EPOCH_TRAINING", "epoch": epoch, "total_epochs": 100,
                        "step_in_epoch": step + 1, "steps_per_epoch": 1875,
                        "completed_steps": completed_steps, "total_steps": 187500,
                        "learning_rate": rate, "recent_seconds_per_step": seconds,
                        "eta_seconds": (187500 - completed_steps) * seconds,
                        "updated_unix": time.time(),
                    })
            record = {
                "epoch": epoch, "seconds": time.monotonic() - epoch_started,
                "learning_rate": rate,
                "stats": {key: value / 1875 for key, value in sums.items()},
            }
            stats_file.write(json.dumps(record, allow_nan=False) + "\n")
            stats_file.flush(); os.fsync(stats_file.fileno())
            final = checkpoints / f"ERC_epoch_{epoch:04d}.pth.tar"
            temporary = final.with_suffix(final.suffix + ".partial")
            torch.save({
                "kind": VERSION, "model_version": VERSION, "protocol": PROTOCOL,
                "cache_manifest_sha256": digest(cache_dir / "manifest.json"),
                "epoch": epoch, "optimizer_steps": completed_steps,
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
            }, temporary)
            os.replace(temporary, final)
    final = checkpoints / "ERC_epoch_0100.pth.tar"
    result = {
        "status": "TRAIN_COMPLETE_PENDING_READONLY_AUDIT", "fixed_epoch": 100,
        "optimizer_steps": 187500, "checkpoint": str(final),
        "checkpoint_sha256": digest(final), "training_seconds": time.monotonic() - started,
        "train_rows": int((~held).sum()), "calibration_rows": int(held.sum()),
        "cache_manifest_sha256": digest(cache_dir / "manifest.json"),
    }
    with (output_dir / "completed.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2); handle.flush(); os.fsync(handle.fileno())
    atomic_json(output_dir / "progress.json", dict(result, phase="TRAINING_COMPLETE"))
    return result


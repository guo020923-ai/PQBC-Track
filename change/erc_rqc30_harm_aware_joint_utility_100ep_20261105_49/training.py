"""Fixed Epoch-100 training entry for the sealed 30k frozen-RQC cache."""

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .config import PROTOCOL, validate_protocol
from .model import HarmAwareJointUtilityERC, VERSION
from .objective import harm_aware_utility_loss


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_new(path, value):
    with Path(path).open("x", encoding="utf8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def load_cache(cache_dir):
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    if digest(manifest_path) != PROTOCOL["cache_manifest_sha256"]:
        raise ValueError("sealed cache manifest identity changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "rqc30_epoch_erc_cache_v1" or manifest.get("examples") != 30000:
        raise ValueError("wrong cache")
    arrays = {key: [] for key in ("evidence", "executable", "utility", "source")}
    for shard in manifest["shards"]:
        path = cache_dir / shard["name"]
        if digest(path) != shard["sha256"]:
            raise ValueError("cache shard changed")
        with np.load(path, allow_pickle=False) as archive:
            for key in arrays:
                arrays[key].append(archive[key].copy())
    return manifest, {key: np.concatenate(value) for key, value in arrays.items()}


def learning_rate(epoch):
    if epoch <= 5:
        return PROTOCOL["peak_learning_rate"] * epoch / 5
    progress = (epoch - 5) / 95
    return PROTOCOL["minimum_learning_rate"] + 0.5 * (
        PROTOCOL["peak_learning_rate"] - PROTOCOL["minimum_learning_rate"]
    ) * (1 + math.cos(math.pi * progress))


def train_epochs(output_dir, cache_dir):
    validate_protocol()
    output_dir, cache_dir = Path(output_dir), Path(cache_dir)
    _, arrays = load_cache(cache_dir)
    seed = PROTOCOL["training_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = HarmAwareJointUtilityERC().cuda().train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PROTOCOL["peak_learning_rate"],
        weight_decay=PROTOCOL["weight_decay"],
    )
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir()
    stats_path = output_dir / "stats.jsonl"
    completed_steps = 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    started = time.monotonic()
    recent = []
    keys = (
        "loss", "regression_loss", "distribution_loss",
        "false_positive_harm_loss", "actionable_rate",
        "predicted_keep_rate", "actionable_recall", "harmful_deploy_rate",
        "mean_deployed_utility", "chosen_regret",
        "predicted_action24_share_nonkeep", "oracle_action24_share_actionable",
    )
    with stats_path.open("x", encoding="utf8") as stats_file:
        for epoch in range(1, 101):
            lr = learning_rate(epoch)
            for group in optimizer.param_groups:
                group["lr"] = lr
            epoch_started = time.monotonic()
            order = torch.randperm(30000, generator=generator).numpy()
            sums = {key: 0.0 for key in keys}
            for step in range(1875):
                step_started = time.monotonic()
                indices = order[step * 16:(step + 1) * 16]
                evidence = torch.from_numpy(arrays["evidence"][indices]).cuda().float()
                executable = torch.from_numpy(arrays["executable"][indices]).cuda().bool()
                utility = torch.from_numpy(arrays["utility"][indices]).cuda().float()
                anchor = torch.zeros(len(indices), dtype=torch.long, device="cuda")
                prediction = model(evidence, executable, anchor)
                loss, row = harm_aware_utility_loss(
                    prediction["predicted_gain"], executable, anchor, utility
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("non-finite gradient")
                optimizer.step()
                completed_steps += 1
                for key in keys:
                    sums[key] += float(row[key])
                recent.append(time.monotonic() - step_started)
                recent = recent[-200:]
                if step == 0 or (step + 1) % 25 == 0:
                    seconds_per_step = sum(recent) / len(recent)
                    atomic_json(output_dir / "progress.json", {
                        "phase": "EPOCH_TRAINING", "epoch": epoch,
                        "total_epochs": 100, "step_in_epoch": step + 1,
                        "steps_per_epoch": 1875,
                        "completed_steps": completed_steps,
                        "total_steps": 187500, "learning_rate": lr,
                        "recent_seconds_per_step": seconds_per_step,
                        "eta_seconds": (187500 - completed_steps) * seconds_per_step,
                        "updated_unix": time.time(),
                    })
            record = {
                "epoch": epoch, "seconds": time.monotonic() - epoch_started,
                "learning_rate": lr,
                "stats": {key: value / 1875 for key, value in sums.items()},
            }
            stats_file.write(json.dumps(record, allow_nan=False) + "\n")
            stats_file.flush()
            os.fsync(stats_file.fileno())
            checkpoint = checkpoints / f"ERC_epoch_{epoch:04d}.pth.tar"
            temporary = checkpoint.with_suffix(checkpoint.suffix + ".partial")
            torch.save({
                "kind": "rqc30_harm_aware_joint_utility_100ep_v1",
                "model_version": VERSION, "protocol": PROTOCOL,
                "cache_manifest_sha256": digest(cache_dir / "manifest.json"),
                "epoch": epoch, "optimizer_steps": completed_steps,
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
            }, temporary)
            os.replace(temporary, checkpoint)
    final = checkpoints / "ERC_epoch_0100.pth.tar"
    completed = {
        "status": "TRAIN_COMPLETE_PENDING_READONLY_AUDIT",
        "fixed_epoch": 100, "optimizer_steps": 187500,
        "checkpoint": str(final), "checkpoint_sha256": digest(final),
        "training_seconds": time.monotonic() - started,
        "cache_manifest_sha256": digest(cache_dir / "manifest.json"),
        "automatic_test": True,
    }
    write_new(output_dir / "completed.json", completed)
    atomic_json(output_dir / "progress.json", dict(completed, phase="TRAINING_COMPLETE"))
    return completed


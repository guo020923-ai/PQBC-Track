"""Build one fixed 30k causal TRAIN cache from frozen RQC30 trajectories."""

import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import PROTOCOL, validate_protocol
from .io_utils import atomic_json, digest, write_new_json
from .objective import action_utility


def _read_rgb(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("missing TRAIN frame: " + str(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class FrozenRQCStream:
    def __init__(self, pools, params, seed):
        from change.erc_natural_reuse_fresh_20261021_01.bridge import collector_class

        self.pools = pools
        self.params = params
        self.rng = random.Random(seed)
        self.collector = collector_class()(params, "tnl2k")
        self.clip = None
        self.cursor = 0

    def _sample_clip(self):
        warmup = PROTOCOL["cache_warmup_frames"]
        sample = PROTOCOL["cache_frames_per_clip"]
        total = 1 + warmup + sample
        for _ in range(2000):
            source = self.rng.choices(
                self.pools.vl,
                [self.pools.weights[index] for index in self.pools.vl],
            )[0]
            pool = self.pools.datasets[source]
            sid = self.rng.randrange(pool.get_num_sequences())
            info = pool.get_sequence_info(sid)
            valid = (info["valid"].bool() & info["visible"].bool()).cpu()
            if len(valid) < total:
                continue
            starts = [
                begin for begin in range(len(valid) - total + 1)
                if bool(valid[begin:begin + total].all())
            ]
            if not starts:
                continue
            begin = self.rng.choice(starts)
            frames, annotation, meta = pool.get_frames(sid, [begin], info)
            language = meta.get("language")
            if not isinstance(language, str) or not language.strip():
                continue
            sequence_path = Path(pool._get_sequence_path(sid)).resolve()
            frame_dir = sequence_path / ("imgs" if pool.get_name() == "tnl2k" else "img")
            paths = sorted(
                path for path in frame_dir.iterdir()
                if path.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            if len(paths) < begin + total:
                continue
            return {
                "source": int(source),
                "domain": pool.get_name(),
                "sequence": pool.sequence_list[sid],
                "begin": begin,
                "paths": paths,
                "annotations": info["bbox"],
                "initial_image": frames[0],
                "initial_box": annotation["bbox"][0].tolist(),
                "language": language,
            }
        raise RuntimeError("no contiguous original-TRAIN cache clip after 2000 draws")

    def _new_clip(self):
        self.clip = self._sample_clip()
        self.collector.initialize(
            self.clip["initial_image"],
            {"init_bbox": self.clip["initial_box"], "language": self.clip["language"]},
        )
        warmup = PROTOCOL["cache_warmup_frames"]
        for offset in range(1, warmup + 1):
            image = _read_rgb(self.clip["paths"][self.clip["begin"] + offset])
            self.collector.track(image, None)
        self.cursor = self.clip["begin"] + 1 + warmup

    @torch.no_grad()
    def next_example(self):
        end = None if self.clip is None else (
            self.clip["begin"] + 1 + PROTOCOL["cache_warmup_frames"]
            + PROTOCOL["cache_frames_per_clip"]
        )
        if self.clip is None or self.cursor >= end:
            self._new_clip()
        image = _read_rgb(self.clip["paths"][self.cursor])
        previous = list(self.collector.state)
        fixed_output = self.collector.track(image, None)
        from .provider import causal_candidate_provider

        packet = causal_candidate_provider(self.collector, image, fixed_output, previous)
        truth = self.clip["annotations"][self.cursor].to(
            packet["boxes"].device, dtype=torch.double
        )[None]
        utility = action_utility(packet["boxes"], truth)
        row = {
            "evidence": packet["evidence"][0].float().cpu().numpy(),
            "executable": packet["executable"][0].cpu().numpy(),
            "utility": utility[0].float().cpu().numpy(),
            "source": self.clip["source"],
        }
        self.cursor += 1
        return row

    def close(self):
        close = getattr(self.collector, "close", None)
        if close is not None:
            close()


def _write_shard(cache_dir, index, rows):
    final = cache_dir / ("shard_%04d.npz" % index)
    temporary = cache_dir / ("shard_%04d.partial" % index)
    arrays = {
        "evidence": np.stack([row["evidence"] for row in rows]).astype(np.float16),
        "executable": np.stack([row["executable"] for row in rows]).astype(np.bool_),
        "utility": np.stack([row["utility"] for row in rows]).astype(np.float32),
        "source": np.asarray([row["source"] for row in rows], dtype=np.int16),
    }
    with temporary.open("xb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, final)
    return {"name": final.name, "examples": len(rows), "sha256": digest(final)}


def build_cache(output_dir):
    validate_protocol()
    output_dir = Path(output_dir)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=False)
    from change.erc_natural_reuse_fresh_20261021_01.bridge import original_training_components

    legacy, _ = original_training_components()
    legacy.idle()
    spec, cfg, plan, params = legacy.settings()
    pools = legacy.FitPools(cfg, plan)
    stream = FrozenRQCStream(pools, params, PROTOCOL["seed"])
    started = time.monotonic()
    total = PROTOCOL["cache_examples"]
    shards, rows = [], []
    intervals = []
    interval_started, interval_completed = started, 0
    try:
        for completed in range(1, total + 1):
            rows.append(stream.next_example())
            if len(rows) == 1024 or completed == total:
                shards.append(_write_shard(cache_dir, len(shards), rows))
                rows = []
            if completed == 1 or completed % 100 == 0 or completed == total:
                elapsed = time.monotonic() - started
                rate = completed / max(elapsed, 1e-9)
                now = time.monotonic()
                if completed > interval_completed:
                    intervals.append((now - interval_started) / (completed - interval_completed))
                    intervals = intervals[-20:]
                    interval_started, interval_completed = now, completed
                low_step = float(np.quantile(intervals, 0.2))
                high_step = float(np.quantile(intervals, 0.8))
                remaining = total - completed
                atomic_json(output_dir / "progress.json", {
                    "phase": "CACHE_EXTRACTION",
                    "completed_examples": completed,
                    "total_examples": total,
                    "elapsed_seconds": elapsed,
                    "examples_per_second": rate,
                    "eta_seconds": remaining / max(rate, 1e-9),
                    "eta_seconds_low": remaining * low_step,
                    "eta_seconds_high": remaining * high_step,
                    "eta_scope": "cache_extraction_only",
                    "updated_unix": time.time(),
                })
        manifest = {
            "schema": "rqc30_epoch_erc_cache_v1",
            "protocol": PROTOCOL,
            "examples": total,
            "shards": shards,
            "seconds": time.monotonic() - started,
            "train_only": True,
            "fixed_rqc_trajectory": True,
        }
        write_new_json(cache_dir / "manifest.json", manifest)
        return manifest
    finally:
        stream.close()

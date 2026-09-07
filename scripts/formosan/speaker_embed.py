#!/usr/bin/env python3
# coding=utf-8
"""Stage 2 (only needed for manifests with NO *usable* `speaker` column --
see `common.has_usable_speaker_column`): compute a ReDimNet2-B6
(multilingual) speaker embedding per kept utterance, cached to
`<manifest_stem>_spk_emb.pt` for pair_ref_audio.py's HDBSCAN clustering.

Model choice: selected via an explicit experiment (VoiceEncoder vs
ECAPA-TDNN vs CAM++ vs ReDimNet2-English vs ReDimNet2-Multilingual),
validated against the real `speaker` labels in ithuan_formosan/
nchc_formosan (the two datasets that DO have one). ReDimNet2-Multilingual
won cleanly on both clustering ARI (1.000 at full 19,814-row scale) and
verification EER/minDCF (0.07% EER, minDCF@0.01=0.0000) -- see the
handoff doc for the full comparison table.

ReDimNet2-B6 needs a GPU to be practical at this data volume (~1.2s/sample
on CPU vs ~0.02-0.1s/sample batched on GPU). Batches are formed by sorting
utterances by duration and bucketing similar-length ones together, so
padding waste stays small; very short trailing padding has negligible
effect on the model's pooled embedding.

Usage:
    python speaker_embed.py --manifest /mnt/md1/user_wayne/formosan_final/manifests/ntu_formosan_corpus/ami-x-pswl_train.jsonl
    python speaker_embed.py --dataset formospeech/klokah --config all --device cuda:0
    python speaker_embed.py --dataset formospeech/klokah --config all --device cuda:0 --shard 0 --num-shards 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchcodec.decoders import AudioDecoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    REDIMNET_DATASET,
    REDIMNET_MODEL_NAME,
    REDIMNET_SR,
    REDIMNET_TORCH_HUB_REPO,
    REDIMNET_TRAIN_TYPE,
    TORCH_HUB_DIR,
    has_usable_speaker_column,
    list_configs,
    load_manifest_rows,
    manifest_dir,
    resolve_splits,
)

# Without expandable segments the caching allocator fragments badly here:
# batches vary in padded length, so it ends up unable to find a contiguous
# block and OOMs with several GB still free (repeatedly, on torch 2.14).
# Setting it in-process keeps the fix with the code rather than relying on
# the caller's environment. It also cut steady-state usage 11GB -> 5GB.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

torch.hub.set_dir(TORCH_HUB_DIR)

# Batches are capped by *padded* audio duration, not by a fixed row count.
# Utterances are sorted by duration, so a fixed count made the trailing
# (longest) batches enormous: this data tops out near 24s, and 64 x 24s =
# ~1550s of padded audio in one forward pass, which OOMs a 24GB card even
# though the same count of ~3s median clips is fine. MAX_BATCH_SECONDS is
# the real memory knob; MAX_BATCH_ROWS just avoids absurd row counts on
# very short clips. 250s was fine on torch 2.9 but OOM-retried repeatedly on
# torch 2.14 (same 24GB card), so it is halved -- the retry keeps results
# correct either way, it just wastes a forward pass each time it fires.
MAX_BATCH_SECONDS = 120.0
MAX_BATCH_ROWS = 64


def manifest_work_seconds(path: Path) -> float:
    """Total audio duration in a manifest -- the cost proxy for sharding.

    Batches are bounded by padded duration, not row count, so seconds of audio
    predicts GPU time far better than the number of rows.
    """
    total = 0.0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                total += json.loads(line).get("duration") or 0.0
    return total


def assign_shard(targets: list[Path], shard: int, num_shards: int) -> list[Path]:
    """Split `targets` across shards by longest-processing-time-first packing.

    A plain `targets[shard::num_shards]` stride splits by manifest *count*,
    which is badly wrong here: manifests are named `<config>_<split>.jsonl` and
    sort into an alternating eval/train sequence, while a klokah train manifest
    can hold ~70x more audio than its eval counterpart. Striding by 6 over that
    period-2 list handed shards 0/2/4 nothing but tiny eval manifests and
    shards 1/3/5 all the large train ones -- measured 5.6h vs 133.4h of audio,
    a 23.8x imbalance that left half the GPUs idle for most of the run.

    Greedy LPT (sort by descending cost, repeatedly assign to the least-loaded
    shard) brings that to 63.32h vs 63.47h -- 1.002x. Every shard computes the
    same assignment from the same inputs, so this needs no coordination between
    the worker processes and stays a pure static split.
    """
    if num_shards <= 1:
        return targets
    weights = {p: manifest_work_seconds(p) for p in targets}
    loads = [0.0] * num_shards
    buckets: list[list[Path]] = [[] for _ in range(num_shards)]
    for path in sorted(targets, key=lambda p: -weights[p]):
        i = min(range(num_shards), key=lambda k: loads[k])
        buckets[i].append(path)
        loads[i] += weights[path]
    return sorted(buckets[shard])


def load_model(device: str):
    model = torch.hub.load(
        REDIMNET_TORCH_HUB_REPO,
        "redimnet2",
        model_name=REDIMNET_MODEL_NAME,
        train_type=REDIMNET_TRAIN_TYPE,
        dataset=REDIMNET_DATASET,
        pretrained=True,
        trust_repo=True,
    ).eval().to(device)
    return model


def load_wav(path: str) -> np.ndarray:
    # torchcodec decodes, downmixes to mono and resamples to REDIMNET_SR in a
    # single step (all source audio here is already mono, so num_channels=1 is
    # a no-op rather than a downmix -- note FFmpeg's stereo downmix applies a
    # gain normalisation and is not the plain channel mean).
    samples = AudioDecoder(path, sample_rate=REDIMNET_SR, num_channels=1).get_all_samples()
    return samples.data.squeeze(0).numpy().astype(np.float32)


def _duration_batches(rows: list[dict], order: list[int]) -> list[list[int]]:
    """Group duration-sorted indices into batches of bounded padded duration.

    Every row in a batch is padded to the batch's longest utterance, so the
    cost of a batch is len(batch) * max_duration -- that product, not the
    row count, is what has to stay bounded.
    """
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0.0
    for i in order:
        dur = float(rows[i].get("duration") or 0.0)
        cand_max = max(current_max, dur)
        if current and (
            len(current) + 1 > MAX_BATCH_ROWS
            or (len(current) + 1) * cand_max > MAX_BATCH_SECONDS
        ):
            batches.append(current)
            current, current_max = [i], dur
        else:
            current.append(i)
            current_max = cand_max
    if current:
        batches.append(current)
    return batches


@torch.inference_mode()
def _forward_batch(rows: list[dict], batch_idx: list[int], model, device: str) -> np.ndarray:
    wavs = [load_wav(rows[i]["audio_filepath"]) for i in batch_idx]
    max_len = max(w.shape[0] for w in wavs)
    padded = np.zeros((len(wavs), max_len), dtype=np.float32)
    for j, w in enumerate(wavs):
        padded[j, : w.shape[0]] = w
    batch_t = torch.from_numpy(padded).to(device)
    return model(batch_t).cpu().numpy().astype(np.float32)  # [B, D]


def _forward_with_oom_retry(rows: list[dict], batch_idx: list[int], model, device: str):
    """Yield (row_index, embedding) pairs, halving the batch on CUDA OOM.

    MAX_BATCH_SECONDS is tuned for a 24GB card; retrying instead of failing
    keeps a long offline run alive on a smaller/busier GPU rather than losing
    hours of completed work to one oversized batch.
    """
    try:
        out = _forward_batch(rows, batch_idx, model, device)
        for j, i in enumerate(batch_idx):
            yield i, out[j]
        return
    except torch.OutOfMemoryError:
        if len(batch_idx) == 1:
            raise
        torch.cuda.empty_cache()
        print(f"  [oom] splitting batch of {len(batch_idx)}", flush=True)
    mid = len(batch_idx) // 2
    for half in (batch_idx[:mid], batch_idx[mid:]):
        yield from _forward_with_oom_retry(rows, half, model, device)


def embed_rows(rows: list[dict], model, device: str) -> dict[str, np.ndarray]:
    # Sort by duration so similar-length utterances batch together (minimal
    # padding waste), then bound each batch by padded duration.
    order = sorted(range(len(rows)), key=lambda i: rows[i].get("duration") or 0.0)
    batches = _duration_batches(rows, order)
    embs: dict[str, np.ndarray] = {}
    t0 = time.time()
    n_done = 0
    for b, batch_idx in enumerate(batches):
        for i, emb in _forward_with_oom_retry(rows, batch_idx, model, device):
            embs[rows[i]["id"]] = emb
        n_done += len(batch_idx)
        if b % 10 == 0:
            print(f"  {n_done}/{len(rows)} ({time.time()-t0:.1f}s)", flush=True)
    return embs


def process_manifest(manifest_path: Path, model, device: str, overwrite: bool) -> None:
    out_path = manifest_path.with_name(manifest_path.stem + "_spk_emb.pt")
    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path} exists")
        return

    rows = load_manifest_rows(manifest_path)
    if not rows:
        print(f"[skip] {manifest_path} is empty")
        return
    if has_usable_speaker_column(rows):
        print(f"[skip] {manifest_path} already has a usable `speaker` column -- no embedding needed")
        return

    print(f"[{manifest_path.name}] {len(rows)} rows, computing embeddings on {device}")
    embs = embed_rows(rows, model, device)
    torch.save(embs, out_path)
    print(f"  -> {out_path} ({len(embs)} embeddings)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--config", default="all")
    ap.add_argument("--splits", default="train,eval")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--shard", type=int, default=0, help="for manual multi-GPU parallelism: this shard's index")
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    targets: list[Path] = []
    if args.manifest:
        targets.append(Path(args.manifest))
    elif args.dataset:
        configs = list_configs(args.dataset) if args.config == "all" else [args.config]
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        mdir = manifest_dir(args.dataset)
        for config in configs:
            for split in resolve_splits(args.dataset, config, splits):
                p = mdir / f"{config}_{split}.jsonl"
                if p.exists():
                    targets.append(p)
                else:
                    print(f"[missing] {p} (run materialize.py first)")
    else:
        raise SystemExit("Pass either --manifest or --dataset")

    targets = assign_shard(targets, args.shard, args.num_shards)
    if not targets:
        print("no targets for this shard")
        return

    model = load_model(args.device)
    for path in targets:
        process_manifest(path, model, args.device, args.overwrite)


if __name__ == "__main__":
    main()

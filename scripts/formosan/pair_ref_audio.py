#!/usr/bin/env python3
# coding=utf-8
"""Stage 3: assign a voice-cloning reference (ref_audio/ref_text) to every
row of a materialized manifest -- strictly within its own
(dataset, lang_code, split), so a train-split row can never be paired with
an eval-split row (eval doubles as the held-out test set, so this would be
a leakage bug, not just a quality issue).

Two strategies, chosen automatically per manifest:
  - Usable `speaker` column (see common.has_usable_speaker_column --
    ithuan_formosan, nchc_formosan, except 4 nchc_formosan configs where
    `speaker` is just a copy of `lang_code`): group by speaker, sample a
    *different* utterance from the same speaker. A speaker with only one
    utterance in this manifest gets no ref (we do not force a
    cross-speaker match).
  - No usable `speaker` column (ntu_formosan_corpus, klokah, and those 4
    nchc_formosan configs): HDBSCAN clustering (metric=precomputed cosine
    distance, min_cluster_size=5, min_samples=1 -- see common.py for the
    rationale) on the cached ReDimNet2 embeddings from speaker_embed.py.
    Sample a random utterance from the same cluster. Rows HDBSCAN calls
    noise (-1), or whose cluster has no other member, get no ref.

Input:  <manifest_dir>/<config>_<split>.jsonl   (from materialize.py)
Output: <manifest_dir>/<config>_<split>.reffed.jsonl

Usage:
    python pair_ref_audio.py --manifest /mnt/md1/user_wayne/formosan_final/manifests/ithuan_formosan/ami-x-skl_train.jsonl
    python pair_ref_audio.py --dataset formospeech/ithuan_formosan --config all --splits train,eval
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import HDBSCAN

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    REF_PAIR_MIN_COSINE,
    REF_PAIR_SINGLE_SPEAKER_MEAN_COSINE,
    has_usable_speaker_column,
    list_configs,
    load_manifest_rows,
    manifest_dir,
    resolve_splits,
)


def cosine_distance_matrix(embs: np.ndarray) -> np.ndarray:
    n = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    sim = n @ n.T
    sim = np.clip(sim, -1.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return (dist + dist.T) / 2.0


def pair_by_speaker(rows: list[dict], seed: int) -> None:
    rng = random.Random(seed)
    by_speaker: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        # A null speaker is *unknown* identity, not a shared one -- bucketing
        # every such row together would pair genuinely different speakers.
        if r.get("speaker") is not None:
            by_speaker[r["speaker"]].append(i)

    n_none = 0
    for i, r in enumerate(rows):
        if r.get("speaker") is None:
            r["ref_audio_filepath"] = None
            r["ref_text"] = None
            r["ref_pairing"] = "none_missing_speaker"
            n_none += 1
            continue
        pool = [j for j in by_speaker[r["speaker"]] if j != i]
        if not pool:
            r["ref_audio_filepath"] = None
            r["ref_text"] = None
            r["ref_pairing"] = "none_singleton_speaker"
            n_none += 1
            continue
        j = rng.choice(pool)
        ref = rows[j]
        r["ref_audio_filepath"] = ref["audio_filepath"]
        r["ref_text"] = ref["ipa"]
        r["ref_pairing"] = "speaker_match"
    print(f"  speaker-based pairing: {n_none}/{len(rows)} rows had no ref (singleton or null speaker)")


def pair_by_embedding(rows: list[dict], emb_cache: dict[str, torch.Tensor], seed: int) -> None:
    rng = random.Random(seed)
    ids = [r["id"] for r in rows if r["id"] in emb_cache]
    id_set = set(ids)
    n_missing = len(rows) - len(ids)
    if n_missing:
        print(f"  WARNING: {n_missing}/{len(rows)} rows had no cached embedding -- no ref for those")

    row_by_id = {r["id"]: r for r in rows}
    for r in rows:
        if r["id"] not in id_set:
            r["ref_audio_filepath"] = None
            r["ref_text"] = None
            r["ref_pairing"] = "none_missing_embedding"

    if len(ids) < 2:
        for i in ids:
            r = row_by_id[i]
            r["ref_audio_filepath"] = None
            r["ref_text"] = None
            r["ref_pairing"] = "none_too_few_rows"
        return

    embs = np.stack([np.asarray(emb_cache[i]) for i in ids]).astype(np.float64)
    dist = cosine_distance_matrix(embs)
    clusterer = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="precomputed",
    )
    labels = clusterer.fit_predict(dist)

    by_cluster: dict[int, list[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        if lab != -1:
            by_cluster[lab].append(idx)

    # 1.0 - distance recovers cosine similarity (see cosine_distance_matrix).
    sim = 1.0 - dist

    # Is this manifest essentially one speaker? Only then is HDBSCAN's noise
    # label an artefact worth overriding (see REF_PAIR_SINGLE_SPEAKER_MEAN_COSINE).
    off_diagonal = ~np.eye(len(ids), dtype=bool)
    mean_cosine = float(sim[off_diagonal].mean())
    single_speaker = mean_cosine >= REF_PAIR_SINGLE_SPEAKER_MEAN_COSINE

    n_noise = int((labels == -1).sum())
    n_none = 0
    n_rescued = 0
    for idx, i in enumerate(ids):
        r = row_by_id[i]
        lab = labels[idx]
        pool = [j for j in by_cluster.get(lab, []) if j != idx] if lab != -1 else []
        pairing = "embedding_cluster"

        if not pool and single_speaker:
            # HDBSCAN found no cluster for this row, but the manifest is one
            # speaker, so that says nothing about identity -- there is simply no
            # density contrast to find. Fall back to "anyone this row is
            # confidently the same speaker as" instead of dropping the ref.
            candidates = np.flatnonzero(sim[idx] >= REF_PAIR_MIN_COSINE)
            pool = [j for j in candidates.tolist() if j != idx]
            pairing = "embedding_similarity"
            if pool:
                n_rescued += 1

        if not pool:
            r["ref_audio_filepath"] = None
            r["ref_text"] = None
            r["ref_pairing"] = "none_noise_or_singleton_cluster"
            n_none += 1
            continue
        choice = rng.choice(pool)
        ref = row_by_id[ids[choice]]
        r["ref_audio_filepath"] = ref["audio_filepath"]
        r["ref_text"] = ref["ipa"]
        r["ref_pairing"] = pairing
    rescue = (
        f"{n_rescued} of those paired by cosine >= {REF_PAIR_MIN_COSINE}"
        if single_speaker
        else "cosine fallback off (multi-speaker manifest)"
    )
    print(
        f"  embedding-cluster pairing: mean_cosine={mean_cosine:.3f} "
        f"single_speaker={single_speaker}, {n_noise}/{len(ids)} rows marked noise "
        f"by HDBSCAN, {rescue}, {n_none}/{len(rows)} rows ended up with no ref"
    )


def process_manifest(manifest_path: Path, seed: int, overwrite: bool) -> None:
    out_path = manifest_path.with_name(manifest_path.stem + ".reffed.jsonl")
    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path} exists")
        return

    rows = load_manifest_rows(manifest_path)
    if not rows:
        print(f"[skip] {manifest_path} is empty")
        return

    usable_speaker = has_usable_speaker_column(rows)
    print(f"[{manifest_path.name}] {len(rows)} rows, usable_speaker_column={usable_speaker}")

    if usable_speaker:
        pair_by_speaker(rows, seed)
    else:
        emb_path = manifest_path.with_name(manifest_path.stem + "_spk_emb.pt")
        if not emb_path.exists():
            raise FileNotFoundError(
                f"{emb_path} not found -- run speaker_embed.py for this manifest first "
                f"(no usable `speaker` column, need ReDimNet2 embeddings for HDBSCAN pairing)."
            )
        emb_cache = torch.load(emb_path, weights_only=False)
        pair_by_embedding(rows, emb_cache, seed)

    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None, help="single manifest jsonl path")
    ap.add_argument("--dataset", default=None, help="e.g. formospeech/ithuan_formosan")
    ap.add_argument("--config", default="all")
    ap.add_argument("--splits", default="train,eval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
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

    for path in targets:
        process_manifest(path, args.seed, args.overwrite)


if __name__ == "__main__":
    main()

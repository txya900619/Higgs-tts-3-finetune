#!/usr/bin/env python3
# coding=utf-8
"""Validate the ref-audio clustering against the manifests that carry real
speaker ids -- a regression test for `common.REF_CLUSTER_DISTANCE`.

Why this exists: the original clustering choice (HDBSCAN) was made on
manifests holding only 2-3 speakers and scored with ARI. Both were too narrow.
klokah has dozens of speakers per config, and ARI is not what pairing cares
about -- splitting one speaker across several clusters is harmless, since
every one of those clusters is still that speaker, while merging two speakers
into one cluster produces exactly the cross-speaker ref the pipeline is built
to avoid. So this scores:

    purity  -- share of rows in a cluster containing exactly one speaker
               (must stay 100%; this is the failure mode that matters)
    paired  -- share of rows whose cluster has at least one other member
               (a row alone in its cluster gets no ref)

Run it after changing REF_CLUSTER_DISTANCE, the embedding model, or the
linkage. It needs embeddings for the labelled manifests, which the pipeline
does not otherwise compute (they pair via their `speaker` column), so pass
--device to put that one-off pass on a GPU.

Usage:
    python validate_clustering.py --device cuda:0
    python validate_clustering.py --device cuda:0 --thresholds 0.4,0.45,0.5,0.55,0.6,0.65
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    MANIFEST_ROOT,
    REF_CLUSTER_DISTANCE,
    load_manifest_rows,
)
from pair_ref_audio import cosine_distance_matrix  # noqa: E402

# Every manifest with real speaker ids. nchc's four pyu-* configs are excluded
# on purpose: their `speaker` is a copy of `lang_code`, not an identity.
LABELLED = [
    ("ithuan_formosan", "ami-x-skl_train"),
    ("ithuan_formosan", "trv-x-tgdy_train"),
    ("ithuan_formosan", "trv-x-truku_train"),
    ("nchc_formosan", "ckv_train"),
    ("nchc_formosan", "trv-x-truku_train"),
]


def score(dist: np.ndarray, speakers: np.ndarray, threshold: float) -> tuple[float, float, int]:
    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="precomputed",
        linkage="average",
    ).fit_predict(dist)
    pure = paired = 0
    for c in set(labels):
        member = labels == c
        if len(set(speakers[member])) == 1:
            pure += int(member.sum())
        if member.sum() >= 2:
            paired += int(member.sum())
    n = len(speakers)
    return 100.0 * pure / n, 100.0 * paired / n, len(set(labels))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--thresholds", default=None, help="comma-separated; default brackets REF_CLUSTER_DISTANCE")
    args = ap.parse_args()

    if args.thresholds:
        thresholds = [float(t) for t in args.thresholds.split(",")]
    else:
        thresholds = sorted({round(REF_CLUSTER_DISTANCE + d, 2) for d in (-0.10, -0.05, 0.0, 0.05, 0.10, 0.15)})

    import speaker_embed as SE

    model = SE.load_model(args.device)

    print(f"REF_CLUSTER_DISTANCE = {REF_CLUSTER_DISTANCE}")
    header = " ".join(f"d={t}".rjust(18) for t in thresholds)
    print(f"  {'manifest':30s} {'spk':>4s} {header}")

    totals = {t: [0, 0, 0] for t in thresholds}
    for ds, stem in LABELLED:
        rows = load_manifest_rows(MANIFEST_ROOT / ds / f"{stem}.jsonl")
        embs = SE.embed_rows(rows, model, args.device)
        E = np.stack([embs[r["id"]] for r in rows]).astype(np.float64)
        dist = cosine_distance_matrix(E)
        speakers = np.array([r["speaker"] for r in rows])

        cells = []
        for t in thresholds:
            purity, paired, n_clusters = score(dist, speakers, t)
            cells.append(f"{purity:5.1f}%/{paired:5.1f}%({n_clusters})")
            totals[t][0] += purity * len(rows) / 100.0
            totals[t][1] += paired * len(rows) / 100.0
            totals[t][2] += len(rows)
        name = f"{ds.split('_')[0]}/{stem}"
        print(f"  {name[:30]:30s} {len(set(speakers)):4d} " + " ".join(c.rjust(18) for c in cells))

    print(f"  {'TOTAL':30s} {'':4s} " + " ".join(
        f"{100 * totals[t][0] / totals[t][2]:5.1f}%/{100 * totals[t][1] / totals[t][2]:5.1f}%".rjust(18)
        for t in thresholds
    ))
    print("\n  purity / paired (clusters). purity must be 100% -- anything less means")
    print("  a cluster mixed two speakers, which is a cross-speaker ref.")

    bad = [t for t in thresholds if totals[t][0] / totals[t][2] < 1.0]
    if REF_CLUSTER_DISTANCE in bad:
        raise SystemExit(f"\nFAIL: REF_CLUSTER_DISTANCE={REF_CLUSTER_DISTANCE} does not hold 100% purity.")
    print(f"\n  OK: REF_CLUSTER_DISTANCE={REF_CLUSTER_DISTANCE} holds 100% purity.")


if __name__ == "__main__":
    main()

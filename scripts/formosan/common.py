# coding=utf-8
"""Shared config/helpers for the Formosan (族語) data pipeline.

Pipeline stages (run in order; each script is a separate stage so any of
them can be re-run / resumed independently):

    1. materialize.py     HF parquet -> filtered wav files + raw manifest
                           jsonl. Filters: dnsmos_ovrl >= DNSMOS_OVRL_MIN
                           (null/NaN fails) and non-empty `mandarin`.
    2. speaker_embed.py    For manifests with no *usable* `speaker` column
                           (no column at all, or fewer than 2 distinct
                           non-null values across the manifest): computes
                           ReDimNet2-B6 (multilingual) speaker embeddings,
                           cached per row.
    3. pair_ref_audio.py   Assigns ref_audio/ref_text (voice-cloning
                           context) to every kept row, strictly within its
                           own (dataset, lang_code, split) -- never across
                           train/eval, to avoid leakage into the eval/test
                           set. Uses the real `speaker` column when usable
                           (group by speaker, random other utterance from
                           the same speaker); otherwise HDBSCAN clustering
                           on the cached embeddings (random utterance from
                           the same cluster). Rows with no same-speaker /
                           same-cluster partner (or classified as noise by
                           HDBSCAN) get no ref audio rather than a forced
                           mismatched one.
    4. build_jsonl.py      Assembles the final {audio,text,ref_audio,
                           ref_text} jsonl (Higgs raw schema) per split,
                           combined across all datasets/configs into one
                           multilingual train/eval/test set. `eval` rows
                           serve double duty as both the validation set and
                           the held-out test set (per user instruction) --
                           test.jsonl is a byte-identical copy of
                           eval.jsonl.

Datasets (all under the `formospeech` HF org, all gated -- the active HF
token must have access):
    - ntu_formosan_corpus : no eval split, no `speaker` column
    - klokah              : has eval split, no `speaker` column
    - ithuan_formosan     : has eval split, has `speaker` column
    - nchc_formosan       : no eval split, has `speaker` column (but 4 of
                            its configs carry a `speaker` column that is
                            just a copy of `lang_code` -- a single distinct
                            value, meaning it's not informative)

We deliberately do NOT hardcode the per-dataset config (lang_code) list or
column set here -- `list_configs()` discovers them at runtime from the repo
itself, since klokah alone has 42 configs and hand-maintaining that list
would rot. Likewise, "does this manifest have a usable speaker column" is
decided at runtime (see `has_usable_speaker_column` in pair_ref_audio.py)
rather than hardcoding the 4 known-uninformative nchc_formosan configs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("HF_HOME", os.environ.get("FORMOSAN_HF_HOME", "/mnt/md0/user_wayne/.hf_cache"))

from huggingface_hub import HfApi, HfFileSystem  # noqa: E402

DATASETS = [
    "formospeech/ntu_formosan_corpus",
    "formospeech/klokah",
    "formospeech/ithuan_formosan",
    "formospeech/nchc_formosan",
]

# Output root. Override with FORMOSAN_ROOT when moving to another machine
# (see FORMOSAN_HANDOFF.md §7) -- nothing else here is machine-specific.
ROOT = Path(os.environ.get("FORMOSAN_ROOT", "/mnt/md0/user_wayne/formosan_final"))
AUDIO_ROOT = ROOT / "audio"
MANIFEST_ROOT = ROOT / "manifests"
OUT_DIR = ROOT / "higgs_jsonl"

TARGET_SR = 24000  # matches scripts/prepare_data.py's default --sample-rate

DNSMOS_OVRL_MIN = 3.0

# --- Speaker embedding model (chosen after an explicit selection experiment:
# VoiceEncoder vs ECAPA-TDNN vs CAM++ vs ReDimNet2-EN vs ReDimNet2-Multilingual,
# validated against the real `speaker` labels in ithuan_formosan/nchc_formosan.
# ReDimNet2-Multilingual won on both clustering ARI (1.000, full 19,814-row
# scale) and verification EER/minDCF (0.07% EER, clearly best of the five). ---
REDIMNET_TORCH_HUB_REPO = "PalabraAI/redimnet2"
REDIMNET_MODEL_NAME = "b6"
REDIMNET_TRAIN_TYPE = "lm"
REDIMNET_DATASET = "vb2+vox2+cnc2_v0"  # multilingual: VoxBlink2+VoxCeleb2+CN-Celeb2
REDIMNET_SR = 16000
TORCH_HUB_DIR = os.environ.get("FORMOSAN_TORCH_HUB_DIR", str(Path(os.environ["HF_HOME"]) / "torch_hub"))

# HDBSCAN params (same selection experiment: min_cluster_size=5,
# min_samples=1 gave the best full-scale ARI with the least noise).
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 1

# Cosine-similarity floor for pairing a row HDBSCAN labelled noise (-1) with
# some other utterance anyway.
#
# Why the fallback exists: HDBSCAN is density-based, so on a manifest that is
# essentially ONE speaker there is no density contrast to find and it labels
# the sparse periphery as noise even though every row is the same person.
# That is exactly what the 4 single-speaker nchc_formosan `pyu-*` configs look
# like -- mean pairwise cosine 0.73-0.81 across the whole manifest, yet
# 15-57% of rows came back as noise and lost their ref audio.
#
# Calibrated on every manifest that carries real speaker ids (ithuan_formosan
# x3, nchc_formosan ckv + trv-x-truku): 19,761 utterances, 6 distinct speaker
# pairs, 44.5M same-speaker and 22.6M different-speaker cosine pairs.
#
#     threshold   different-speaker pairs passing   rows left with no ref
#         0.50               0.0033%                        0
#         0.60               0.0000%                        2
#         0.70               0.0000%                       10  (of 10,969)
#         0.80               0.0000%                      426
#
# The highest different-speaker cosine actually observed is 0.583, so 0.50 is
# NOT safe despite looking clean against a single config -- an early
# calibration used only nchc ckv, whose two speakers are male/female, and
# cross-gender is the easiest case to separate (max 0.481). Same-gender pairs
# run much closer: nchc trv-x-truku peaks at 0.583.
#
# 0.70 leaves ~0.12 of margin above that observed maximum and costs only 10
# rows (0.09%), because pairing samples randomly among ALL candidates over the
# bar -- a row loses its ref only if *no* partner clears it, not merely
# because some same-speaker pairs fall below. The calibration set has just 6
# speaker pairs while klokah has far more speakers, so the true cross-speaker
# maximum is likely higher than 0.583; the margin is cheap insurance.
REF_PAIR_MIN_COSINE = 0.70

# The cosine fallback above is only *justified* when a manifest really is one
# speaker, which is the case HDBSCAN cannot handle: with no density contrast
# it labels the sparse periphery as noise even though every row is the same
# person. When a manifest genuinely holds many speakers, a noise label means
# what it says, and pairing those rows by similarity would risk exactly the
# cross-speaker match the whole design avoids.
#
# Mean pairwise cosine over the manifest separates the two cases cleanly:
#     nchc_formosan pyu-x-ksvk / pyu-x-pym  (proven single-speaker)  0.810 / 0.733
#     ntu_formosan_corpus  xnb / dru-x-ngdr                          0.443 / 0.558
#     klokah, 10 sampled configs                                     0.335 - 0.472
#
# klokah is emphatically multi-speaker -- it also yields 18-89 HDBSCAN clusters
# per config with inter-cluster similarity reaching 0.969, so no fixed cosine
# bar could tell its speakers apart. It carries 252,758 of the 284,655 rows, so
# applying the fallback there would have mispaired far more than it fixed.
# 0.65 sits in the empty band between the two groups.
REF_PAIR_SINGLE_SPEAKER_MEAN_COSINE = 0.65

_HF_FS: Optional[HfFileSystem] = None
_HF_API: Optional[HfApi] = None


def hf_fs() -> HfFileSystem:
    global _HF_FS
    if _HF_FS is None:
        _HF_FS = HfFileSystem()
    return _HF_FS


def hf_api() -> HfApi:
    global _HF_API
    if _HF_API is None:
        _HF_API = HfApi()
    return _HF_API


def dataset_tag(repo: str) -> str:
    """`formospeech/klokah` -> `klokah`."""
    return repo.split("/")[-1]


def list_configs(repo: str) -> list[str]:
    """Discover config (lang_code) names from the repo's file tree.

    A config is any top-level directory containing at least one
    `train-*.parquet` file (mirrors the HF `configs:` block each of these
    dataset cards declares).
    """
    info = hf_api().dataset_info(repo)
    configs = sorted(
        {
            s.rfilename.split("/", 1)[0]
            for s in info.siblings
            if "/" in s.rfilename and "/train-" in s.rfilename
        }
    )
    if not configs:
        raise ValueError(f"No configs discovered for {repo}")
    return configs


def has_eval_split(repo: str, config: str) -> bool:
    info = hf_api().dataset_info(repo)
    return any(s.rfilename.startswith(f"{config}/eval-") for s in info.siblings)


def manifest_dir(repo: str) -> Path:
    d = MANIFEST_ROOT / dataset_tag(repo)
    d.mkdir(parents=True, exist_ok=True)
    return d

def all_manifest_dirs() -> list[Path]:
    return [manifest_dir(repo) for repo in DATASETS]


def audio_dir(repo: str, config: str) -> Path:
    d = AUDIO_ROOT / dataset_tag(repo) / config
    d.mkdir(parents=True, exist_ok=True)
    return d


def strip_ipa_dashes(ipa: str) -> str:
    """`"t-a-ɾ-a-ʦ-o-w-a k-i-s-o?"` -> `"tarat͡sowa kiso?"`.

    The `ipa` column is a NeMo-phone-tokenizer format with "-" joining
    individual phones -- meaningless to a byte-level BPE tokenizer (Higgs'
    Qwen3 backbone), so we strip the delimiters and keep the phone sequence
    itself plus the inter-word spaces. Confirmed against the old Chatterbox
    Formosan pipeline (`chatterbox/dataset.py:_raw_text`), which did the
    same thing for the same reason (its own grapheme tokenizer).
    """
    return ipa.replace("-", "")


def resolve_splits(repo: str, config: str, requested: list[str]) -> list[str]:
    splits = list(requested)
    if "eval" in splits and not has_eval_split(repo, config):
        splits.remove("eval")
    return splits


def load_manifest_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def has_usable_speaker_column(rows: list[dict]) -> bool:
    """Is this manifest's `speaker` column real speaker identity?

    Unusable in exactly two cases:
      - no non-null values at all (ntu_formosan_corpus, klokah);
      - every value is just a copy of the row's `lang_code` -- the 4
        nchc_formosan configs (pyu-x-ksvk/ktrp/mkzy/pym) whose `speaker`
        field is a constant equal to the config name.

    Anything else is usable, *including a single distinct value*. A manifest
    where every row shares one real speaker id is the easiest pairing case
    (any other utterance is guaranteed same-speaker), not an unusable one --
    an earlier ">= 2 distinct values" rule sent those down the embedding
    path instead. That misrouted all three ithuan_formosan `eval` manifests,
    which are single-speaker (E-PV001 / E-SV001 / E-TV001) and only ~20 rows
    each: HDBSCAN with min_cluster_size=5 can label a large fraction of so
    few rows as noise, silently dropping ref audio from the eval/test set.

    Note this is deliberately *not* keyed on the distinct-value count:
    single-value is legitimate (ithuan eval) or degenerate (nchc pyu)
    depending only on whether the value is the lang_code.
    """
    labelled = [
        (r.get("speaker"), r.get("lang_code")) for r in rows if r.get("speaker") is not None
    ]
    return any(speaker != lang_code for speaker, lang_code in labelled)

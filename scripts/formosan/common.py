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
    - ilrdf_dicts         : no eval split, no `speaker` column; 16 configs of
                            dictionary example sentences. Column-for-column
                            identical to ntu_formosan_corpus, so it needs no
                            special handling -- it was simply missing here.

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
    "formospeech/ilrdf_dicts",
]

# Output root: audio/, manifests/ and higgs_jsonl/ live here. It sits inside
# the repo but is gitignored -- ~79GB, almost all of it 270k small wav files.
# Override with FORMOSAN_ROOT when moving to another machine (see
# FORMOSAN_HANDOFF.md §7); nothing else here is machine-specific.
ROOT = Path(os.environ.get("FORMOSAN_ROOT", "/mnt/md0/user_wayne/Higgs-tts-3-finetune/data"))
AUDIO_ROOT = ROOT / "audio"
MANIFEST_ROOT = ROOT / "manifests"
OUT_DIR = ROOT / "higgs_jsonl"

TARGET_SR = 24000  # matches scripts/prepare_data.py's default --sample-rate

DNSMOS_OVRL_MIN = 3.0

# Minimum utterance length, in seconds. Applied in pair_ref_audio.py, i.e.
# BEFORE reference pairing, which is the point that matters: filtering after
# pairing would leave rows whose target clears the bar but whose reference does
# not, and intersecting both conditions afterwards throws away far more than
# necessary (124,813 rows survive that way against 174,425 when the short
# utterances are removed from the pool first).
#
# 3.0s matches what Higgs' own documentation asks of a cloning reference
# ("as little as 3 to 5 seconds"), and a reference is just another row's
# target here, so the same bar has to apply to both.
MIN_DURATION_SEC = 3.0

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

# Clustering for ref-audio pairing: agglomerative, average linkage, on the
# cosine DISTANCE matrix (1 - cosine similarity). Merge the two nearest
# clusters repeatedly and stop once the nearest pair is further apart than
# REF_CLUSTER_DISTANCE.
#
# This replaced HDBSCAN, which the original model-selection experiment picked.
# That experiment was sound but scoped too narrowly: it only ever measured
# manifests with 2-3 speakers (ithuan_formosan, nchc_formosan), and it scored
# them with ARI. Neither transfers to klokah, which holds 252,758 of the
# 284,655 rows and has dozens of speakers per config:
#
#   - HDBSCAN is density-based, so a speaker whose recordings vary a lot reads
#     as "low density" and gets fragmented or labelled noise even though those
#     utterances sit close together. On klokah it left 5-36% of rows as noise,
#     i.e. with no ref at all, and lowering min_cluster_size made it *worse*
#     (sxr: 11% noise at 5, 45% at 3). Agglomerative leaves 0.2-0.4% isolated
#     on the same manifests.
#   - ARI is the wrong objective here. Splitting one speaker across several
#     clusters is harmless for pairing (every cluster is still that speaker);
#     only *merging* two speakers does damage. ARI punishes the harmless case,
#     which is what made agglomerative look threshold-sensitive. Scored on
#     cluster purity instead, agglomerative holds 100% purity and 100% pairing
#     across every labelled manifest for the whole band d=0.45..0.60, and only
#     fails at 0.65 (ithuan ami-x-skl's two speakers merge).
#
# 0.50 sits mid-band, and independently at the valley of klokah's bimodal
# pairwise-cosine distribution (modes near 0.25 and 0.70, trough at 0.45-0.55).
#
# Nothing else is needed: the earlier REF_PAIR_MIN_COSINE fallback and its
# single-speaker gate existed only to repair HDBSCAN's noise labels, and both
# are gone with it.
REF_CLUSTER_DISTANCE = 0.40

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

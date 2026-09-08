#!/usr/bin/env python3
# coding=utf-8
"""Stage 1: HF parquet rows -> filtered 24kHz mono wav files + raw manifest jsonl.

Filters applied (per user spec):
    - dnsmos_ovrl present (not null/NaN) and >= common.DNSMOS_OVRL_MIN
    - mandarin present (not null, not empty/whitespace-only after strip)

Reads the dataset's parquet files DIRECTLY with pyarrow rather than going
through `datasets.load_dataset()`. `load_dataset` "generates" a second full
Arrow copy of every config on disk before you can iterate it -- ~115GB for
klokah alone, on top of the ~47GB of parquet already in the hub cache -- and
`datasets.disable_caching()` does not prevent it (that switch only governs
`.map()` transform caches; its own docs say "caching doesn't affect
load_dataset"). The parquet already stores the encoded audio inline as
`audio: struct<bytes, path>`, so pyarrow + torchcodec reads it with no
intermediate copy. hf_hub_download still gives us the usual cached,
resumable download of the parquet itself.

Audio I/O is torchcodec, not soundfile/torchaudio:
  - `AudioDecoder(bytes, sample_rate=..., num_channels=1)` decodes, downmixes
    and resamples in one step, straight from the parquet bytes -- no temp file.
  - `AudioEncoder(...).to_file()` writes ~21x faster than `soundfile.write()`
    (measured: 551 vs 26 files/sec). libsndfile appears to force a sync per
    close, so writing hundreds of thousands of small files spent nearly all
    its time blocked in `jbd2_log_wait_commit`; FFmpeg-backed torchcodec does
    not, which is what made this stage IO-bound rather than CPU-bound.

Resumable: re-running skips any (config, split) whose manifest already
exists unless --overwrite is passed. Row-level audio writes are NOT
individually resumed within a manifest -- an interrupted run leaves a
PARTIAL manifest that a later run would wrongly treat as complete, so
delete that manifest (or pass --overwrite for its config/split) before
resuming.

The `speaker` column (when the source dataset has one -- ithuan_formosan,
nchc_formosan) is copied through verbatim. Whether it's actually *usable*
for ref-audio pairing (some nchc_formosan configs carry a `speaker` that's
just a copy of `lang_code`) is decided later, in pair_ref_audio.py, by
checking the manifest itself -- not here.

Usage:
    python materialize.py --dataset formospeech/ithuan_formosan --config all
    python materialize.py --dataset formospeech/klokah --config ami-x-frng --limit 200
    python materialize.py --dataset formospeech/klokah --config all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

# `common` must be imported BEFORE huggingface_hub: it sets HF_HOME via
# os.environ.setdefault, and huggingface_hub freezes its cache paths at import
# time. Importing hf_hub_download first silently sends every download to
# ~/.cache/huggingface instead -- which on this machine is the small root
# partition, and quietly put 60GB of klokah parquet there.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DNSMOS_OVRL_MIN,
    TARGET_SR,
    audio_dir,
    hf_api,
    list_configs,
    manifest_dir,
    resolve_splits,
    strip_ipa_dashes,
)

from huggingface_hub import hf_hub_download  # noqa: E402  (must follow `common`)

# Kept after `common` for the same reason: it is stdlib-only today, but the
# ordering above must not depend on that staying true.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from g2p import text_to_ipa  # noqa: E402

# Everything except `audio`; pulled separately so the (large) audio column is
# only materialized one row group at a time.
META_COLUMNS = [
    "id", "duration", "text", "ipa", "mandarin",
    "language", "lang_code", "dnsmos_ovrl",
]


def split_parquet_paths(repo: str, config: str, split: str) -> list[str]:
    """Local paths to this (config, split)'s parquet files, downloading as needed."""
    info = hf_api().dataset_info(repo)
    names = sorted(
        s.rfilename
        for s in info.siblings
        if s.rfilename.startswith(f"{config}/{split}-") and s.rfilename.endswith(".parquet")
    )
    return [hf_hub_download(repo, n, repo_type="dataset") for n in names]


def sha_of_file(path: Path) -> str:
    """Content fingerprint of the written clip, for de-duplication downstream.

    klokah reuses one recording across several lesson items, each with its own
    id and therefore its own `<id>.wav`. Path-based de-duplication cannot see
    that, so 576 train rows shared a recording with 425 of the 7,844 eval rows
    (5.4% of the eval set) -- verified in the source parquet, where e.g.
    ami-x-frng mode-0000906/0000907 (train) and mode-0000908 (eval) all carry
    byte-identical audio. Recording the hash here lets `build_jsonl` split on
    content instead, without re-reading the corpus.

    Hashing the file rather than `samples.data` keeps this value reproducible
    from the manifest alone: anything can re-derive it from the wav on disk.
    """
    h = hashlib.blake2b(digest_size=20)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phonemize(text: str, lang_code: str, published: str) -> tuple[str, bool]:
    """The row's IPA, with the upstream apostrophe spacing bug repaired.

    The published `ipa` column is generated with a g2p table in which 15
    Atayal/Bunun/Paiwan rows map `ʼ` to "ʔ " -- a phone with a trailing space.
    That space survives into the corpus, where it is indistinguishable from a
    word boundary, so it cannot be repaired from the string alone: it splits
    words in half (`maʼun` -> `maʔ un`, 18,768 rows) and strands punctuation
    (`taknaʼ.` -> `taknaʔ .`, 5,210 rows), and a word-final `ʼ` legitimately
    *is* followed by a boundary.

    So we regenerate from `text` instead, where `convert_to_ipa` takes word
    boundaries from `text.split()` and the ambiguity never arises. This is a
    repair, not a re-transcription: with the space left in (`strip_values=False`)
    the same code reproduces the published column byte-for-byte on all 284,606
    materialised rows, and with it removed the output differs from that column
    in whitespace alone. The tie bars are undone because the corpus writes the
    affricates as single code points and only the spacing should change.

    Returns the IPA and whether it had to fall back to the published column.
    """
    try:
        ipa = text_to_ipa(text, lang_code)
    except (KeyError, ValueError):
        return strip_ipa_dashes(published), True
    return ipa.replace("t͡s", "ʦ").replace("t͡ɕ", "ʨ").replace("d͡ʒ", "ʤ"), False


def duplicated_ids(paths: list[str]) -> set[str]:
    """Ids that appear more than once in this (config, split).

    Audio is written to `<id>.wav`, so a repeated id means the second row
    silently overwrites the first and both manifest rows then point at one
    file. ithuan_formosan does this: `trv-x-truku` alone has 48 ids carrying
    two rows each, with byte-identical audio but *different* text -- the same
    recording labelled with two transcripts. One of them is wrong and nothing
    in the data says which, so both rows are dropped rather than keeping a
    coin-flip. It is a tiny number (12 rows of 284,655 survive filtering), and
    a wrong text->audio pair is exactly what a TTS model must not be taught.
    """
    seen: set[str] = set()
    repeated: set[str] = set()
    for row_id in paths:
        if row_id in seen:
            repeated.add(row_id)
        seen.add(row_id)
    return repeated


def scan_ids(paths: list[str]) -> set[str]:
    """Ids appearing more than once across this split's parquet files."""
    ids: list[str] = []
    for path in paths:
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            ids.extend(str(x) for x in pf.read_row_group(rg, columns=["id"]).column("id").to_pylist())
    return duplicated_ids(ids)


def process_split(repo: str, config: str, split: str, limit: int | None, overwrite: bool) -> dict:
    manifest_path = manifest_dir(repo) / f"{config}_{split}.jsonl"
    if manifest_path.exists() and not overwrite and limit is None:
        return {"dataset": repo, "config": config, "split": split, "skipped": "manifest exists"}

    paths = split_parquet_paths(repo, config, split)
    out_dir = audio_dir(repo, config)
    ambiguous_ids = scan_ids(paths)

    n_seen = n_kept = 0
    n_bad_dnsmos = n_bad_mandarin = n_decode_error = n_dup_id = 0
    n_ipa_fallback = 0
    t0 = time.time()
    stop = False

    with open(manifest_path, "w", encoding="utf-8") as fh:
        for path in paths:
            if stop:
                break
            pf = pq.ParquetFile(path)
            available = set(pf.schema_arrow.names)
            columns = [c for c in (*META_COLUMNS, "speaker", "audio") if c in available]

            for rg in range(pf.metadata.num_row_groups):
                if stop:
                    break
                table = pf.read_row_group(rg, columns=columns)
                for row in table.to_pylist():
                    n_seen += 1
                    if limit is not None and n_seen > limit:
                        stop = True
                        break

                    dnsmos = row.get("dnsmos_ovrl")
                    if dnsmos is None or dnsmos != dnsmos or dnsmos < DNSMOS_OVRL_MIN:
                        n_bad_dnsmos += 1
                        continue

                    mandarin = (row.get("mandarin") or "").strip()
                    if not mandarin:
                        n_bad_mandarin += 1
                        continue

                    row_id = str(row["id"])
                    if row_id in ambiguous_ids:
                        n_dup_id += 1
                        continue
                    wav_path = out_dir / f"{row_id}.wav"
                    try:
                        samples = AudioDecoder(
                            row["audio"]["bytes"], sample_rate=TARGET_SR, num_channels=1
                        ).get_all_samples()
                        AudioEncoder(samples.data, sample_rate=TARGET_SR).to_file(str(wav_path))
                    except Exception:
                        n_decode_error += 1
                        continue

                    audio_sha = sha_of_file(wav_path)

                    ipa, fell_back = phonemize(
                        row.get("text") or "", config, row.get("ipa") or "",
                    )
                    n_ipa_fallback += fell_back

                    record = {
                        "id": row_id,
                        "source_dataset": repo,
                        "lang_code": config,
                        "language": row.get("language"),
                        "split": split,
                        "duration": row.get("duration"),
                        "text": row.get("text"),
                        "ipa": ipa,
                        "mandarin": mandarin,
                        "dnsmos_ovrl": dnsmos,
                        "audio_filepath": str(wav_path),
                        "audio_sha": audio_sha,
                    }
                    if row.get("speaker") is not None:
                        record["speaker"] = row["speaker"]
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_kept += 1

    return {
        "dataset": repo,
        "config": config,
        "split": split,
        "n_seen": n_seen,
        "n_kept": n_kept,
        "n_skipped_dnsmos": n_bad_dnsmos,
        "n_skipped_mandarin": n_bad_mandarin,
        "n_skipped_duplicate_id": n_dup_id,
        "n_decode_error": n_decode_error,
        "n_ipa_fallback": n_ipa_fallback,
        "elapsed_sec": round(time.time() - t0, 1),
        "manifest": str(manifest_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g. formospeech/klokah")
    ap.add_argument("--config", default="all", help="lang_code, or 'all'")
    ap.add_argument("--splits", default="train,eval", help="comma-separated; eval is skipped per-config if the repo has none")
    ap.add_argument("--limit", type=int, default=None, help="cap rows per (config,split) -- smoke testing")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    configs = list_configs(args.dataset) if args.config == "all" else [args.config]
    requested_splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    all_stats = []
    for config in configs:
        for split in resolve_splits(args.dataset, config, requested_splits):
            stats = process_split(args.dataset, config, split, args.limit, args.overwrite)
            print(json.dumps(stats, ensure_ascii=False))
            all_stats.append(stats)

    print("=== summary ===")
    kept = sum(s.get("n_kept", 0) for s in all_stats)
    seen = sum(s.get("n_seen", 0) for s in all_stats)
    print(f"total kept {kept} / seen {seen} across {len(all_stats)} (config,split) shards")


if __name__ == "__main__":
    main()

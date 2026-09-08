#!/usr/bin/env python3
# coding=utf-8
"""Stage 4: assemble the final Higgs-raw-schema jsonl (the format
`scripts/prepare_data.py` in the main repo expects) from every
`*.reffed.jsonl` manifest under the data root's manifests/, combined across
all datasets/configs into one multilingual train / eval / test set.

Per-row output schema (matches the Higgs-tts-3-finetune README's "Prepare
Data" section):
    {"audio": <path>, "text": <ipa>, "ref_audio": <path>, "ref_text": <ipa>}
`ref_audio`/`ref_text` are omitted entirely for rows pair_ref_audio.py
could not find a same-speaker/same-cluster partner for. Reference audio is
optional per-sample both here and in the base model -- its card documents a
"Zero-shot TTS" mode that takes no reference at all -- but `--require-ref`
drops those rows anyway, and defaults to doing so. See its help text for why.

No language/dialect tag is prefixed onto `text` (decided against it: the
`ipa` text already differs enough per language/dialect, and Higgs' Qwen3
tokenizer has no built-in language-tag special token to hook into anyway
-- confirmed by dumping the full added-tokens list from
`multimodalart/higgs-audio-v3-tts-4b-transformers`'s tokenizer.json).

`eval` rows serve double duty as both the validation set and the final
held-out test set (per user instruction) -- test.jsonl is a byte-identical
copy of eval.jsonl, kept as a separate file only so downstream configs can
name either role explicitly.

Usage:
    python build_jsonl.py                       # all datasets found on disk
    python build_jsonl.py --datasets formospeech/ithuan_formosan
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MANIFEST_ROOT, OUT_DIR, dataset_tag  # noqa: E402


def iter_reffed_manifests(dataset_tags: list[str] | None):
    for d in sorted(MANIFEST_ROOT.iterdir()):
        if not d.is_dir():
            continue
        if dataset_tags is not None and d.name not in dataset_tags:
            continue
        for p in sorted(d.glob("*.reffed.jsonl")):
            yield p


def row_to_higgs_record(row: dict) -> dict:
    record = {"audio": row["audio_filepath"], "text": row["ipa"]}
    if row.get("ref_audio_filepath"):
        record["ref_audio"] = row["ref_audio_filepath"]
        record["ref_text"] = row["ref_text"]
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=None, help="comma-separated dataset repo ids; default = all found on disk")
    ap.add_argument(
        "--require-ref", choices=("all", "eval", "none"), default="all",
        help=(
            "Drop rows that have no ref_audio. 'all' (default) drops them from "
            "train and eval, 'eval' only from eval, 'none' keeps everything.\n"
            "\n"
            "Without a reference the base model still synthesizes -- its card "
            "calls this 'Zero-shot TTS' -- but the voice comes purely from "
            "sampling: there is no speaker embedding or default speaker in the "
            "prompt. Measured on six generations of one sentence differing only "
            "by seed, pairwise speaker similarity was 0.369 (range 0.21-0.59), "
            "squarely in this project's cross-speaker band (0.25-0.35) and "
            "nowhere near its same-speaker band (0.72-0.88). Every generation is "
            "a different voice; upstream issue #14 asks how to stabilize it and "
            "is still unanswered.\n"
            "\n"
            "For a voice-cloning finetune those rows therefore teach the opposite "
            "of the goal -- invent a voice when no reference is given, rather "
            "than follow the reference -- so they are dropped by default. Keep "
            "them (--require-ref none) only if zero-shot synthesis is also "
            "wanted. Dropping is also what makes eval metrics well defined: "
            "speaker similarity has nothing to compare against without a ref."
        ),
    )
    args = ap.parse_args()

    dataset_tags = None
    if args.datasets:
        dataset_tags = [dataset_tag(d) for d in args.datasets.split(",")]

    # Some source splits are not disjoint: every one of ithuan_formosan
    # trv-x-tgdy's eval rows is a byte-for-byte duplicate of a train row (same
    # id, same audio bytes), and trv-x-truku has two more. Re-checked against
    # the current upstream revision (a1414bd9, 2026-09-02): still 20/20 and
    # 2/19 -- the overlap is not fixed at the source, so this stays. Rows
    # are written to `<id>.wav`, so a shared id is also a shared *file* -- which
    # is why those eval rows' ref_audio pointed at "train" audio. eval doubles
    # as the held-out test set, so a train row that also appears in eval has to
    # go: drop it from train and keep eval intact. Done here, at assembly, so it
    # holds no matter which manifests are combined.
    # Splitting on the file path is not enough. klokah hangs one recording off
    # several lesson items, each with its own id and so its own `<id>.wav`, and
    # the train/eval split was made by id: 576 train rows share a recording with
    # 425 of the 7,844 eval rows (5.4% of the eval set), all of them klokah.
    # Confirmed in the source parquet -- ami-x-frng mode-0000906/0000907 (train)
    # and mode-0000908 (eval) are one recording under three ids. So compare the
    # `audio_sha` materialize records, and index it by path too, since a row
    # names its reference by path.
    sha_by_path: dict[str, str] = {}
    eval_sha: set[str] = set()
    missing_sha = 0
    for manifest_path in iter_reffed_manifests(dataset_tags):
        stem = manifest_path.name[: -len(".reffed.jsonl")]
        is_eval = stem.rsplit("_", 1)[-1] == "eval"
        with open(manifest_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sha = row.get("audio_sha")
                if not sha:
                    missing_sha += 1
                    continue
                sha_by_path[row["audio_filepath"]] = sha
                if is_eval:
                    eval_sha.add(sha)
    if missing_sha:
        raise SystemExit(
            f"{missing_sha} rows have no `audio_sha`. Falling back to path "
            "comparison would silently re-admit the duplicate recordings this "
            "guards against -- re-run materialize.py, or backfill the field."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_files = {
        "train": open(OUT_DIR / "train.jsonl", "w", encoding="utf-8"),
        "eval": open(OUT_DIR / "eval.jsonl", "w", encoding="utf-8"),
    }
    counts = {"train": 0, "eval": 0}
    n_with_ref = {"train": 0, "eval": 0}
    # The same reuse that leaks a recording across splits also repeats one
    # inside a split: 5,247 train rows and 16 eval rows are re-recordings of a
    # clip already emitted. In train that silently up-weights those speakers,
    # in eval it scores the same clip twice, so keep one copy of each. Which
    # copy is arbitrary but stable -- manifests are walked in sorted order.
    seen_sha = {"train": set(), "eval": set()}
    n_dropped_repeat = {"train": 0, "eval": 0}
    n_dropped_dup = 0
    n_stripped_ref = 0
    n_dropped_no_ref = {"train": 0, "eval": 0}
    require_ref_in = {"all": {"train", "eval"}, "eval": {"eval"}, "none": set()}[args.require_ref]
    per_lang_counts: dict[str, dict[str, int]] = {}

    for manifest_path in iter_reffed_manifests(dataset_tags):
        # <config>_<split>.reffed.jsonl
        stem = manifest_path.name[: -len(".reffed.jsonl")]
        split = stem.rsplit("_", 1)[-1]
        if split not in out_files:
            print(f"[skip] unrecognized split in {manifest_path.name}")
            continue
        with open(manifest_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sha = sha_by_path[row["audio_filepath"]]
                if split == "train" and sha in eval_sha:
                    n_dropped_dup += 1
                    continue
                if sha in seen_sha[split]:
                    n_dropped_repeat[split] += 1
                    continue
                seen_sha[split].add(sha)
                record = row_to_higgs_record(row)
                if split == "train" and sha_by_path.get(record.get("ref_audio")) in eval_sha:
                    # Pairing runs before this dedup, so a train row can hold a
                    # ref pointing at audio that turned out to be eval's. eval
                    # doubles as the test set, so that would condition training
                    # on test audio. The ref is optional per-sample -- drop just
                    # the ref, keep the row.
                    record.pop("ref_audio", None)
                    record.pop("ref_text", None)
                    n_stripped_ref += 1
                if "ref_audio" not in record and split in require_ref_in:
                    n_dropped_no_ref[split] += 1
                    continue
                out_files[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split] += 1
                if "ref_audio" in record:
                    n_with_ref[split] += 1
                lc = row["lang_code"]
                per_lang_counts.setdefault(lc, {"train": 0, "eval": 0})
                per_lang_counts[lc][split] += 1

    for f in out_files.values():
        f.close()

    test_path = OUT_DIR / "test.jsonl"
    shutil.copyfile(OUT_DIR / "eval.jsonl", test_path)

    print(f"dropped {n_dropped_dup} train rows whose audio also appears in eval (source-split overlap)")
    print(f"dropped repeats of an already-emitted recording: "
          f"{n_dropped_repeat['train']} train / {n_dropped_repeat['eval']} eval")
    print(f"stripped ref_audio from {n_stripped_ref} train rows that referenced eval audio")
    print(
        f"--require-ref={args.require_ref}: dropped "
        f"{n_dropped_no_ref['train']} train / {n_dropped_no_ref['eval']} eval rows with no ref_audio"
    )
    print(f"train.jsonl: {counts['train']} rows ({n_with_ref['train']} with ref_audio) -> {OUT_DIR / 'train.jsonl'}")
    print(f"eval.jsonl:  {counts['eval']} rows ({n_with_ref['eval']} with ref_audio) -> {OUT_DIR / 'eval.jsonl'}")
    print(f"test.jsonl:  {counts['eval']} rows -> {test_path}  (copy of eval.jsonl, per user spec)")
    print(f"languages covered: {len(per_lang_counts)}")
    for lc, c in sorted(per_lang_counts.items()):
        print(f"  {lc}: train={c['train']} eval={c['eval']}")


if __name__ == "__main__":
    main()

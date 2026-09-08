# coding=utf-8
"""Stage 1.5: drop duplicate recordings, before reference pairing sees them.

klokah hangs one recording off several lesson items, each with its own id and
so its own `<id>.wav` -- `ami-x-frng` mode-0000906/0000907 (train) and
mode-0000908 (eval) are one recording under three ids, confirmed byte-identical
in the source parquet. ithuan_formosan goes further and reuses the *same* id
across train and eval. Neither is visible to a path comparison.

This has to run before `pair_ref_audio.py`, not after. Pairing draws a
reference from within the same (config, split) manifest, so a train row's
reference is a train file by definition -- but 616 of those files were
recordings that also sit in eval under another id, which would condition
training on held-out audio. Cleaning up afterwards in `build_jsonl` meant
stripping those references and then dropping 661 train rows left without one.
Deduplicating first lets pairing pick a legitimate reference instead, so those
rows survive.

Three classes are removed, all by `audio_sha`:
  - a train row whose recording also appears in eval (eval doubles as the
    held-out test set, so train yields);
  - repeats of a recording already kept within the same split, which would
    otherwise up-weight those speakers in train and double-score them in eval.

`build_jsonl` keeps its own equivalents as a safety net; after this stage they
should report zero.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MANIFEST_ROOT, dataset_tag  # noqa: E402


def raw_manifests(dataset_tags: list[str] | None) -> list[Path]:
    out = []
    for d in sorted(MANIFEST_ROOT.iterdir()):
        if not d.is_dir() or (dataset_tags and d.name not in dataset_tags):
            continue
        out += [p for p in sorted(d.glob("*.jsonl")) if ".reffed." not in p.name]
    return out


def split_of(path: Path) -> str:
    return path.name[: -len(".jsonl")].rsplit("_", 1)[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=None, help="comma-separated repo ids; default = all on disk")
    ap.add_argument("--apply", action="store_true", help="rewrite the manifests (default is a dry run)")
    args = ap.parse_args()

    tags = [dataset_tag(d) for d in args.datasets.split(",")] if args.datasets else None
    manifests = raw_manifests(tags)
    if not manifests:
        raise SystemExit("no manifests found")

    missing = 0
    eval_sha: set[str] = set()
    for path in manifests:
        if split_of(path) != "eval":
            continue
        for line in open(path, encoding="utf-8"):
            sha = json.loads(line).get("audio_sha")
            if sha:
                eval_sha.add(sha)
            else:
                missing += 1
    if missing:
        raise SystemExit(f"{missing} eval rows have no `audio_sha` -- re-run materialize.py first")
    print(f"{len(manifests)} manifests, {len(eval_sha):,} distinct eval recordings", flush=True)

    # Sorted order makes "which copy survives" stable across runs.
    seen: dict[str, set[str]] = {}
    dropped = Counter()
    per_dataset = Counter()
    for path in manifests:
        split = split_of(path)
        kept_lines = []
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            sha = row.get("audio_sha")
            if not sha:
                raise SystemExit(f"{path} has a row with no `audio_sha` -- re-run materialize.py")
            if split == "train" and sha in eval_sha:
                dropped["train row whose recording is in eval"] += 1
                per_dataset[path.parent.name] += 1
                continue
            bucket = seen.setdefault(split, set())
            if sha in bucket:
                dropped[f"repeat within {split}"] += 1
                per_dataset[path.parent.name] += 1
                continue
            bucket.add(sha)
            kept_lines.append(line)
        if args.apply:
            tmp = str(path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.writelines(kept_lines)
            os.replace(tmp, path)

    total = sum(dropped.values())
    print(f"\n{'dropped' if args.apply else '[dry run] would drop'} {total:,} rows")
    for reason, n in dropped.most_common():
        print(f"  {n:>7,}  {reason}")
    print(f"  by dataset: {dict(per_dataset)}")
    if not args.apply:
        print("\nre-run with --apply to rewrite; then re-run pair_ref_audio.py and build_jsonl.py")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# coding=utf-8
"""Transcribe a set of wav files with the Formosan ASR, for WER/CER scoring.

The same script transcribes ground-truth recordings and every synthesized
system, so the recognizer, decoding settings and normalization are identical
across the comparison -- the only thing that varies is the audio.

Decoding is greedy (`num_beams=1`). The checkpoint inherits a beam setting
from `openai/whisper-large-v2` that OOMs a 24GB card at batch 8, and beam
search would in any case only matter if it were applied unevenly.

`language="id"` is not a mistake: this model reuses Indonesian as its language
id, as its model card specifies.

Input JSONL needs `audio` (path) and `ortho` (reference orthography, since the
ASR emits orthography rather than IPA); anything else is passed through.

Usage:
    python asr_transcribe.py --input testset.jsonl --output asr_gt.jsonl
    python asr_transcribe.py --input x.jsonl --output y.jsonl --shard 0 --num-shards 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline  # noqa: E402

ASR_MODEL_ID = "ILRDF/whisper-large-v2-formosan-all"
ASR_LANGUAGE = "id"
BATCH_SIZE = 8


def row_cost(row: dict, audio_field: str) -> float:
    """Cost proxy for one utterance.

    Whisper pads every clip to 30s of mel frames, so the encoder costs the same
    regardless of length; what varies is how many tokens the decoder emits,
    which tracks transcript length. Fall back to file size when there is no
    reference text (synthesized audio scored before its transcript exists).
    """
    text = row.get("ortho") or row.get("text") or ""
    if text:
        return float(len(text))
    try:
        return os.path.getsize(row[audio_field]) / 1000.0
    except OSError:
        return 1.0


def assign_shard(rows: list[dict], shard: int, num_shards: int, audio_field: str) -> list[dict]:
    """Split rows across shards by longest-processing-time-first packing.

    A plain `rows[shard::num_shards]` stride balances the row *count*, which is
    not the same as balancing work: decode length varies several-fold between
    utterances, so one shard can finish well ahead of the other and leave a GPU
    idle. Greedy LPT on the cost proxy keeps them within a fraction of a
    percent. Every shard derives the same assignment from the same input, so
    this needs no coordination.
    """
    if num_shards <= 1:
        return rows
    costs = {id(r): row_cost(r, audio_field) for r in rows}
    loads = [0.0] * num_shards
    buckets: list[list[dict]] = [[] for _ in range(num_shards)]
    for row in sorted(rows, key=lambda r: -costs[id(r)]):
        i = min(range(num_shards), key=lambda k: loads[k])
        buckets[i].append(row)
        loads[i] += costs[id(row)]
    return buckets[shard]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--audio-field", default="audio", help="which field holds the wav path")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    rows = assign_shard(rows, args.shard, args.num_shards, args.audio_field)
    if not rows:
        raise SystemExit(f"no rows for shard {args.shard}/{args.num_shards}")

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        ASR_MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True,
    ).to(args.device)
    processor = AutoProcessor.from_pretrained(ASR_MODEL_ID)
    asr = pipeline(
        "automatic-speech-recognition",
        model=model, tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128, chunk_length_s=30,
        batch_size=args.batch_size, torch_dtype=dtype, device=args.device,
    )

    out_path = Path(args.output)
    if args.num_shards > 1:
        out_path = out_path.with_name(f"{out_path.stem}.rank{args.shard}-of-{args.num_shards}{out_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_fail = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start : start + args.batch_size]
            try:
                outputs = asr(
                    [r[args.audio_field] for r in chunk],
                    generate_kwargs={"language": ASR_LANGUAGE, "num_beams": 1},
                )
            except Exception as exc:  # keep a long run alive; record the gap
                n_fail += len(chunk)
                print(f"  [fail] batch at {start}: {type(exc).__name__}: {exc}", flush=True)
                outputs = [{"text": ""} for _ in chunk]
            for row, out in zip(chunk, outputs):
                fh.write(json.dumps({**row, "asr": out["text"].strip()}, ensure_ascii=False) + "\n")
            done = min(start + args.batch_size, len(rows))
            if (start // args.batch_size) % 20 == 0 or done == len(rows):
                rate = done / max(time.time() - t0, 1e-9)
                eta = (len(rows) - done) / max(rate, 1e-9)
                print(f"  {done}/{len(rows)} ({rate:.1f}/s, eta {eta/60:.0f}m)", flush=True)

    print(f"[asr] shard={args.shard}/{args.num_shards} rows={len(rows)} failed={n_fail} "
          f"elapsed={time.time()-t0:.0f}s output={out_path}")


if __name__ == "__main__":
    main()

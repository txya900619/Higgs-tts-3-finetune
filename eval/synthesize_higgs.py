#!/usr/bin/env python3
# coding=utf-8
"""Synthesize the evaluation set with Higgs Audio v3, optionally with a LoRA.

`scripts/infer.py` handles one utterance and never passes `reference_text`,
even though `generate_speech` accepts it and every training sample carried one.
This passes it, both because the model was trained that way and because the F5
baseline receives the reference transcript too -- withholding it here would
hand F5 an advantage that has nothing to do with either model's acoustics.

Text is the dataset's own `ipa` field, which is what this model was finetuned
on. That differs from what `eval/g2p.py` produces for F5 (different glottal
symbols, spacing and punctuation conventions); each system is fed the format it
was trained on rather than a shared one, which is the only way the comparison
measures the models instead of a format mismatch.

Usage:
    python synthesize_higgs.py --input testset.jsonl --out-dir gen/higgs \
        --lora-path output/lora_r16_lr1e-4/checkpoint-best --shard 0 --num-shards 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# eval/ sits beside the repo packages rather than inside them.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from torchcodec.encoders import AudioEncoder  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-path", default="multimodalart/higgs-audio-v3-tts-4b-transformers")
    ap.add_argument("--lora-path", default=None, help="omit to evaluate the base model")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-f5-ok", action="store_true",
                    help="restrict to rows F5 can also synthesize, for the like-for-like table")
    args = ap.parse_args()

    from model.modeling import HiggsMultimodalQwen3ForConditionalGeneration

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    if args.only_f5_ok:
        rows = [r for r in rows if r.get("f5_ok")]
    rows = rows[args.shard :: args.num_shards]
    if not rows:
        raise SystemExit(f"no rows for shard {args.shard}/{args.num_shards}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = HiggsMultimodalQwen3ForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    if args.lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path).merge_and_unload()
    model = model.to(args.device).eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / f"manifest.rank{args.shard}-of-{args.num_shards}.jsonl"

    t0 = time.time()
    n_fail = 0
    with open(manifest, "w", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            stem = Path(row["audio"]).stem
            wav_path = out_dir / f"{row['lang_code']}__{stem}.wav"
            try:
                # Seed per row so a rerun reproduces the same sample, and so two
                # systems are not compared across different random draws.
                torch.manual_seed(args.seed + i)
                decoded = AudioDecoderCache.load(row["ref_audio"])
                wav = model.generate_speech(
                    row["ipa"], tokenizer,
                    reference_audio=decoded, reference_sample_rate=24000,
                    reference_text=row["ref_ipa"],
                    temperature=args.temperature, top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                )
                AudioEncoder(wav.unsqueeze(0), sample_rate=model.config.sample_rate).to_file(str(wav_path))
            except Exception as exc:
                n_fail += 1
                print(f"  [fail] {stem}: {type(exc).__name__}: {exc}", flush=True)
                continue
            fh.write(json.dumps({**row, "audio": str(wav_path), "gt_audio": row["audio"]},
                                ensure_ascii=False) + "\n")
            if i % 25 == 0 or i == len(rows) - 1:
                rate = (i + 1) / max(time.time() - t0, 1e-9)
                print(f"  {i+1}/{len(rows)} ({rate:.2f}/s, eta {(len(rows)-i-1)/max(rate,1e-9)/60:.0f}m)", flush=True)

    print(f"[higgs] shard={args.shard}/{args.num_shards} rows={len(rows)} failed={n_fail} "
          f"lora={args.lora_path} elapsed={time.time()-t0:.0f}s out={out_dir}")


class AudioDecoderCache:
    """Decode a reference wav to the tensor `generate_speech` expects."""

    @staticmethod
    def load(path: str) -> torch.Tensor:
        from torchcodec.decoders import AudioDecoder
        return AudioDecoder(path, sample_rate=24000, num_channels=1).get_all_samples().data


if __name__ == "__main__":
    main()

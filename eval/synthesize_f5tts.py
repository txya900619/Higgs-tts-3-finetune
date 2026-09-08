#!/usr/bin/env python3
# coding=utf-8
"""Synthesize the evaluation set with the Formosan F5-TTS baseline.

Ported from the ILRDF/formosan-f5-tts Space's `app.py`. The model construction
(`load_f5tts` / `load_model`) and the inference path (`load_vocoder`,
`preprocess_ref_audio_text`, `infer_process` from `f5_tts.infer.utils_infer`)
are the Space's, unchanged, so the baseline runs the way its authors intended.

What differs, deliberately:

  - No Gradio, and no `refs.yaml`. The Space ships one fixed demo reference
    clip per dialect; using those would compare our system's cloning against a
    hand-picked prompt. Each row here supplies its OWN `ref_audio` / reference
    transcript -- the same pair our model gets -- so the two systems are given
    identical conditioning.
  - `hf_hub_download` instead of `cached_path("hf://...")`.
  - Text is converted by `eval/g2p.py`, which fixes the Space's longest-match
    bug (see its docstring). Rows whose orthography contains graphemes absent
    from the dialect's g2p table cannot be converted at all; they are marked
    `f5_ok=False` upstream and skipped here, and reported separately rather
    than quietly dropped.

Usage:
    python synthesize_f5tts.py --input testset.jsonl --out-dir gen/f5 \
        --shard 0 --num-shards 2
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

F5_REPO = "ithuan/f5-tts-formosan-all-finetune-v3"
F5_CKPT = "model_877212.safetensors"
F5_VOCAB = "vocab.txt"


def load_f5tts(device: str, dtype=torch.float32):
    """The Space's model construction, verbatim apart from how files are fetched."""
    from f5_tts.infer.utils_infer import (
        hop_length, load_checkpoint, mel_spec_type, n_fft, n_mel_channels,
        ode_method, target_sample_rate, win_length,
    )
    from f5_tts.model import CFM, DiT
    from f5_tts.model.utils import get_tokenizer

    ckpt_path = hf_hub_download(F5_REPO, F5_CKPT)
    vocab_path = hf_hub_download(F5_REPO, F5_VOCAB)

    # `old=False` in the Space's config -> text_mask_padding=True, pe_attn_head=None
    model_cfg = dict(
        dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4,
        text_mask_padding=True, pe_attn_head=None,
    )
    vocab_char_map, vocab_size = get_tokenizer(vocab_path, "custom")
    model = CFM(
        transformer=DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            n_mel_channels=n_mel_channels, target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(method=ode_method),
        vocab_char_map=vocab_char_map,
    ).to(device)
    # use_ema mirrors the Space's `old` flag (False for this checkpoint).
    return load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--nfe-step", type=int, default=32)
    ap.add_argument("--cross-fade-duration", type=float, default=0.15)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--dtype", default="float16", choices=["float32", "float16"],
                    help="The Space runs float32; float16 is ~2.6x faster and was "
                         "checked to leave WER/CER and speaker similarity unchanged.")
    args = ap.parse_args()

    from f5_tts.infer.utils_infer import infer_process, load_vocoder, preprocess_ref_audio_text

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("f5_ok")]
    rows = rows[args.shard :: args.num_shards]
    if not rows:
        raise SystemExit(f"no rows for shard {args.shard}/{args.num_shards}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / f"manifest.rank{args.shard}-of-{args.num_shards}.jsonl"

    vocoder = load_vocoder()
    model = load_f5tts(args.device, getattr(torch, args.dtype))

    t0 = time.time()
    n_fail = 0
    with open(manifest, "w", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            stem = Path(row["audio"]).stem
            wav_path = out_dir / f"{row['lang_code']}__{stem}.wav"
            try:
                ref_audio, ref_text = preprocess_ref_audio_text(
                    row["ref_audio"], row["f5_ref_text"], show_info=lambda *a, **k: None,
                )
                wave, sr, _ = infer_process(
                    ref_audio, ref_text, row["f5_text"], model, vocoder,
                    cross_fade_duration=args.cross_fade_duration,
                    nfe_step=args.nfe_step, speed=args.speed,
                    show_info=lambda *a, **k: None,
                )
                sf.write(str(wav_path), wave, sr)
            except Exception as exc:
                n_fail += 1
                print(f"  [fail] {stem}: {type(exc).__name__}: {exc}", flush=True)
                continue
            fh.write(json.dumps({**row, "audio": str(wav_path), "gt_audio": row["audio"]},
                                ensure_ascii=False) + "\n")
            if i % 25 == 0 or i == len(rows) - 1:
                rate = (i + 1) / max(time.time() - t0, 1e-9)
                print(f"  {i+1}/{len(rows)} ({rate:.2f}/s, eta {(len(rows)-i-1)/max(rate,1e-9)/60:.0f}m)", flush=True)

    print(f"[f5] shard={args.shard}/{args.num_shards} rows={len(rows)} failed={n_fail} "
          f"elapsed={time.time()-t0:.0f}s out={out_dir}")


if __name__ == "__main__":
    main()

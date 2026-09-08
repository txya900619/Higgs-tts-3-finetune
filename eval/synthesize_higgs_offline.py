#!/usr/bin/env python3
# coding=utf-8
"""Synthesize the evaluation set with vllm-omni's OFFLINE engine, in-process.

Runs inside the vllm-omni container. Preferred over the HTTP server
(`synthesize_higgs_omni.py`) because `engine.generate()` takes a *list* of
prompts -- the shipped example loops one at a time, but the API batches -- so
vLLM schedules the whole set itself instead of us guessing a concurrency, and
nothing crosses a socket. Sending reference audio over HTTP costs ~333KB of
base64 per request, 2.7GB across the full 7,844-row set.

Each prompt carries its own reference codes in `additional_information`, so
per-utterance voice cloning batches fine; the per-request reference is not a
barrier to batching the way it is for F5.

Prompt construction follows the shipped
`examples/offline_inference/text_to_speech/higgs_audio_v3/end2end.py`, with two
changes:

  - Reference codes come from `--codes`, the JSONL `prepare_data.py` already
    produced, instead of calling `encode_reference_audio` per row. Both run the
    same `bosonai/higgs-audio-v2-tokenizer`; spot-checked on real rows they
    agree on 639 of 640 codes, the one difference being numeric noise. Only the
    delay pattern still has to be applied, which is a tensor reshape.
  - Every prompt is built before a single `generate()` call. Building them in
    chunks left both talker replicas idle during each chunk's prep -- reading
    and encoding N reference clips is serial work with no generation behind it.
    One queue keeps the GPUs fed from start to finish; the codes are a few
    thousand ints per row, so holding them all costs little.

Usage (inside the container):
    python /work/eval/synthesize_higgs_offline.py \\
        --model /models/lora_r16_lr1e-4 \\
        --deploy-config /cfg/higgs_2gpu_clone.yaml \\
        --input /work/data/eval_out/testset.jsonl \\
        --out-dir /work/data/eval_out/gen_higgs
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SAMPLE_RATE = 24000


def extract_pcm(multimodal_output: dict) -> torch.Tensor:
    audio = multimodal_output.get("model_outputs")
    if audio is None:
        audio = multimodal_output.get("audio")
    if audio is None:
        raise ValueError(f"no audio key: {list(multimodal_output.keys())}")
    if isinstance(audio, list):
        valid = [torch.as_tensor(a).float().cpu().reshape(-1) for a in audio if a is not None]
        if not valid:
            raise ValueError("empty audio list")
        return torch.cat(valid, dim=0) if len(valid) > 1 else valid[0]
    return torch.as_tensor(audio).float().cpu().reshape(-1)


def to_int16(pcm: torch.Tensor) -> np.ndarray:
    arr = pcm.numpy()
    if arr.dtype.kind == "f":
        arr = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
    return arr.astype(np.int16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--deploy-config", default=None)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--audio-prefix-from", default=None,
                    help="host path prefix to rewrite, e.g. /mnt/.../data/audio")
    ap.add_argument("--audio-prefix-to", default=None, help="container path, e.g. /audio")
    ap.add_argument("--codes", default=None,
                    help="JSONL from prepare_data.py carrying ref_audio_codes; "
                         "falls back to encoding each reference on the fly when omitted")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from vllm_omni.entrypoints.omni import Omni
    from transformers import AutoTokenizer
    from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_tokenizer import (
        HiggsAudioV3TokenizerAdapter, apply_delay_pattern, encode_reference_audio,
    )

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def remap(p: str) -> str:
        if args.audio_prefix_from and args.audio_prefix_to:
            return p.replace(args.audio_prefix_from, args.audio_prefix_to)
        return p

    def wav_path(row) -> Path:
        return out_dir / f"{row['lang_code']}__{Path(row['audio']).stem}.wav"

    todo = [r for r in rows if not wav_path(r).exists()]
    print(f"[offline] {len(rows)} rows, {len(todo)} to do", flush=True)
    if not todo:
        return

    engine = Omni(model=args.model, deploy_config=args.deploy_config, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    adapter = HiggsAudioV3TokenizerAdapter(tokenizer)

    ref_codes_by_path: dict[str, list] = {}
    if args.codes:
        needed = {r["ref_audio"] for r in todo}
        # A reference here is always some other row's target, so its codes are in
        # the file twice: as that row's `audio_codes`, and as `ref_audio_codes`
        # on whichever rows point at it. Take either.
        for line in open(args.codes, encoding="utf-8"):
            if len(ref_codes_by_path) == len(needed):
                break
            row = json.loads(line)
            for path_key, codes_key in (("ref_audio", "ref_audio_codes"), ("audio", "audio_codes")):
                src = row.get(path_key)
                if src in needed and src not in ref_codes_by_path and row.get(codes_key):
                    ref_codes_by_path[src] = row[codes_key]
        missing = len(needed) - len(ref_codes_by_path)
        print(f"[offline] reference codes: {len(ref_codes_by_path)}/{len(needed)} from "
              f"{args.codes}" + (f"; {missing} will be encoded on the fly" if missing else ""),
              flush=True)

    t0 = time.time()
    n_done = n_fail = 0

    prompts, keep = [], []
    for row in todo:
        try:
            cached = ref_codes_by_path.get(row["ref_audio"])
            if cached is not None:
                raw = torch.as_tensor(cached, dtype=torch.long)
            else:
                wav, sr = sf.read(remap(row["ref_audio"]), always_2d=False)
                if wav.ndim == 2:
                    wav = wav.mean(axis=1)
                raw = encode_reference_audio(wav, int(sr))
            ref = apply_delay_pattern(raw)
            ids = adapter.build_prompt(
                row["ipa"], num_ref_tokens=int(ref.shape[0]), reference_text=row["ref_ipa"],
            )
            ids, positions = adapter.prepare_prompt_for_engine(ids)
            prompts.append({
                "prompt_token_ids": ids,
                "additional_information": {
                    "audio_input_ids": ref.to(torch.long),
                    "audio_input_ids_mask": torch.ones(ref.shape[0], dtype=torch.bool),
                    "audio_placeholder_positions": positions,
                },
            })
            keep.append(row)
        except Exception as exc:
            n_fail += 1
            print(f"  [prep-fail] {Path(row['audio']).stem}: {type(exc).__name__}: {exc}", flush=True)

    print(f"[offline] {len(prompts)} prompts built in {time.time()-t0:.1f}s; generating", flush=True)
    t_gen = time.time()
    outputs = engine.generate(prompts)
    gen_elapsed = time.time() - t_gen

    # Omni.generate() is a generator drained in *completion* order, not input
    # order (vllm_omni/entrypoints/omni.py: `while active_reqs: ... yield`), so
    # zipping it against `keep` silently pairs every clip with another row's
    # text. Request ids are minted as f"{i}_{uuid4()}", so the leading integer
    # is the index into `keep`.
    ordered: list = [None] * len(keep)
    for out in outputs:
        rid = getattr(out, "request_id", None)
        try:
            idx = int(str(rid).split("_", 1)[0])
        except (TypeError, ValueError):
            raise SystemExit(
                f"cannot recover input order: unexpected request_id {rid!r}. "
                "Refusing to fall back to positional pairing -- it would "
                "mislabel every output."
            )
        ordered[idx] = out
    if any(o is None for o in ordered):
        missing = [i for i, o in enumerate(ordered) if o is None]
        print(f"  [warn] {len(missing)} prompts returned no output", flush=True)

    for row, out in zip(keep, ordered):
        if out is None:
            n_fail += 1
            continue
        try:
            pcm = extract_pcm(out.outputs[0].multimodal_output)
            sf.write(str(wav_path(row)), to_int16(pcm), SAMPLE_RATE, format="WAV", subtype="PCM_16")
            n_done += 1
        except Exception as exc:
            n_fail += 1
            print(f"  [write-fail] {Path(row['audio']).stem}: {type(exc).__name__}: {exc}", flush=True)
    print(f"[offline] generate() {gen_elapsed:.1f}s for {len(prompts)} "
          f"({len(prompts)/max(gen_elapsed,1e-9):.2f}/s)", flush=True)

    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            p = wav_path(row)
            if p.exists():
                fh.write(json.dumps({**row, "audio": str(p), "gt_audio": row["audio"]},
                                    ensure_ascii=False) + "\n")
    print(f"[offline] done={n_done} failed={n_fail} elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

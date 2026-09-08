#!/usr/bin/env python3
# coding=utf-8
"""Synthesize the evaluation set against a vllm-omni Higgs Audio v3 server.

Roughly 6x faster than driving `model.generate_speech` directly through
transformers (0.54 vs 0.09 utterances/sec on two A5000s), which takes the full
7,844-row evaluation from ~12h to ~4h. The win is vLLM's continuous batching:
the autoregressive talker cannot be batched by hand across utterances the way a
diffusion model can, but the server batches concurrent requests for us.

Serve the model first (see eval/omni/higgs_2gpu_clone.yaml for the deploy
config and why it differs from the shipped profiles):

    docker run -d --name higgs-omni --gpus '"device=1,2"' --ipc=host \\
      -v $PWD/data/omni:/models:ro -v $PWD/eval/omni:/cfg:ro \\
      -v /mnt/md0/user_wayne/.hf_cache:/hf_cache -e HF_HOME=/hf_cache \\
      -p 8095:8095 vllm/vllm-omni:nightly-x86_64 \\
      vllm serve /models/<checkpoint> --host 0.0.0.0 --port 8095 \\
      --trust-remote-code --omni --deploy-config /cfg/higgs_2gpu_clone.yaml

Reference audio goes over the wire as a data: URL -- the endpoint rejects raw
base64 ("The URL must be either a HTTP, data or file URL"). `ref_text` is
optional for v3 but is sent anyway: every training sample carried one, and the
F5 baseline gets the reference transcript too.

Resumable: a row whose wav already exists is skipped, so an interrupted run can
be restarted without redoing work.

Usage:
    python synthesize_higgs_omni.py --input data/eval_out/testset.jsonl \\
        --out-dir data/eval_out/gen_higgs --concurrency 16
"""
from __future__ import annotations

import argparse
import base64
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def synthesize(session, url, model, row, out_path, retries):
    ref = base64.b64encode(Path(row["ref_audio"]).read_bytes()).decode()
    payload = {
        "model": model,
        "input": row["ipa"],
        "ref_audio": f"data:audio/wav;base64,{ref}",
        "ref_text": row["ref_ipa"],
    }
    last = None
    for attempt in range(retries + 1):
        try:
            resp = session.post(url, json=payload, timeout=600)
            if resp.status_code == 200 and resp.content:
                out_path.write_bytes(resp.content)
                return None
            last = f"HTTP {resp.status_code}: {resp.text[:160]}"
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(2 * (attempt + 1))
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--server", default="http://localhost:8095")
    ap.add_argument("--model", default="/models/lora_r16_lr1e-4")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def wav_path(row):
        return out_dir / f"{row['lang_code']}__{Path(row['audio']).stem}.wav"

    todo = [r for r in rows if not wav_path(r).exists()]
    print(f"[higgs-omni] {len(rows)} rows, {len(rows) - len(todo)} already done, {len(todo)} to do", flush=True)
    if not todo:
        return

    url = f"{args.server.rstrip('/')}/v1/audio/speech"
    session = requests.Session()
    session.mount("http://", requests.adapters.HTTPAdapter(
        pool_connections=args.concurrency, pool_maxsize=args.concurrency))

    lock = threading.Lock()
    state = {"done": 0, "fail": 0}
    t0 = time.time()
    failures = []

    def work(row):
        err = synthesize(session, url, args.model, row, wav_path(row), args.retries)
        with lock:
            if err:
                state["fail"] += 1
                failures.append((row["audio"], err))
                print(f"  [fail] {Path(row['audio']).stem}: {err}", flush=True)
            else:
                state["done"] += 1
            n = state["done"] + state["fail"]
            if n % 25 == 0 or n == len(todo):
                rate = n / max(time.time() - t0, 1e-9)
                print(f"  {n}/{len(todo)} ({rate:.2f}/s, eta {(len(todo)-n)/max(rate,1e-9)/60:.0f}m, "
                      f"{state['fail']} failed)", flush=True)

    with ThreadPoolExecutor(args.concurrency) as pool:
        list(pool.map(work, todo))

    manifest = out_dir / "manifest.jsonl"
    with open(manifest, "w", encoding="utf-8") as fh:
        for row in rows:
            p = wav_path(row)
            if p.exists():
                fh.write(json.dumps({**row, "audio": str(p), "gt_audio": row["audio"]},
                                    ensure_ascii=False) + "\n")

    print(f"[higgs-omni] done={state['done']} failed={state['fail']} "
          f"elapsed={time.time()-t0:.0f}s manifest={manifest}")


if __name__ == "__main__":
    main()

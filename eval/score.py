# coding=utf-8
"""Score the three-way TTS comparison: WER/CER, speaker similarity, per dialect.

Consumes the ASR hypotheses written by ``eval/asr_transcribe.py`` (sharded
outputs are merged automatically) and reports every metric per dialect as well
as overall, because dialect difficulty varies enormously here -- the ground
truth alone spans 3.5% to 58.6% WER, so a single pooled number says almost
nothing about whether a system works.

Speaker similarity is cosine on ReDimNet b6 embeddings between each synthesised
clip and the reference clip it was cloned from, using the same model and
decoding path as ``scripts/formosan/speaker_embed.py`` so the numbers stay
comparable with the corpus-side clustering work. Ground truth is scored the
same way: it is a *different utterance by the same speaker*, which makes it the
natural ceiling rather than a perfect 1.0.

Usage:
    python eval/score.py --manifest data/eval_out/s30_higgs.jsonl \
        --system gt=data/eval_out/asr_gt.jsonl \
        --system f5=data/eval_out/asr_f5.jsonl \
        --system higgs=data/eval_out/asr_higgs.jsonl \
        --out data/eval_out/scores.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "formosan"))

from normalize import normalize  # noqa: E402


def load_asr(spec: str) -> list[dict]:
    """Read one system's ASR output, merging ``rank<i>-of-<n>`` shards."""
    path = Path(spec)
    if path.exists():
        files = [path]
    else:
        stem, suffix = path.stem, path.suffix
        files = sorted(path.parent.glob(f"{stem}.rank*-of-*{suffix}"))
        if not files:
            raise SystemExit(f"no ASR output at {spec} (and no rank shards beside it)")
    rows: list[dict] = []
    for f in files:
        rows += [json.loads(line) for line in open(f, encoding="utf-8")]
    return rows


def edit_distance(a: list, b: list) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def rate(errors: int, total: int) -> float:
    return 100.0 * errors / total if total else float("nan")


def speaker_similarity(pairs: list[tuple[str, str]], device: str) -> dict[tuple[str, str], float]:
    """Cosine between each (generated, reference) pair, one ReDimNet pass each."""
    import numpy as np
    import torch
    from speaker_embed import load_model, load_wav  # reuse the corpus-side loader

    model = load_model(device)
    cache: dict[str, "np.ndarray"] = {}

    def embed(path: str) -> "np.ndarray":
        if path not in cache:
            wav = torch.from_numpy(load_wav(path)).unsqueeze(0).to(device)
            with torch.inference_mode():
                vec = model(wav).squeeze(0).float().cpu().numpy()
            cache[path] = vec / (np.linalg.norm(vec) + 1e-12)
        return cache[path]

    out = {}
    for i, (gen, ref) in enumerate(pairs, 1):
        out[(gen, ref)] = float(embed(gen) @ embed(ref))
        if i % 200 == 0 or i == len(pairs):
            print(f"  sim {i}/{len(pairs)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="row set that defines the comparison (e.g. s30_higgs.jsonl)")
    ap.add_argument("--system", action="append", required=True, metavar="NAME=ASR_JSONL",
                    help="repeatable; the ASR output for one system")
    ap.add_argument("--out", help="write the full per-row and per-dialect scores here")
    ap.add_argument("--sim-device", default="cuda:0")
    ap.add_argument("--no-sim", action="store_true", help="skip speaker similarity")
    args = ap.parse_args()

    manifest = [json.loads(line) for line in open(args.manifest, encoding="utf-8")]
    # A row is identified by its ground-truth audio: every system synthesises
    # the same sentence from the same reference, so that path is the join key.
    wanted = {r["audio"] for r in manifest}
    truth = {r["audio"]: r for r in manifest}

    systems: dict[str, dict[str, dict]] = {}
    for spec in args.system:
        if "=" not in spec:
            raise SystemExit(f"--system wants NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        rows = {}
        for r in load_asr(path):
            key = r.get("gt_audio", r["audio"])  # generated rows carry the original
            if key in wanted:
                rows[key] = r
        missing = len(wanted) - len(rows)
        print(f"[score] {name}: {len(rows)}/{len(wanted)} rows" +
              (f" ({missing} missing)" if missing else ""), flush=True)
        systems[name] = rows

    sims: dict[str, dict[str, float]] = {n: {} for n in systems}
    if not args.no_sim:
        pairs, owners = [], []
        for name, rows in systems.items():
            for key, r in rows.items():
                gen, ref = r["audio"], truth[key]["ref_audio"]
                if os.path.exists(gen) and os.path.exists(ref):
                    pairs.append((gen, ref))
                    owners.append((name, key))
        print(f"[score] speaker similarity over {len(pairs)} pairs", flush=True)
        scored = speaker_similarity(pairs, args.sim_device)
        for (name, key), pair in zip(owners, pairs):
            sims[name][key] = scored[pair]

    per_row: list[dict] = []
    agg: dict[str, dict[str, dict[str, float]]] = {
        n: defaultdict(lambda: {"we": 0, "wn": 0, "ce": 0, "cn": 0, "sim": 0.0, "sn": 0})
        for n in systems
    }
    for name, rows in systems.items():
        for key, r in rows.items():
            ref_text = normalize(truth[key]["ortho"])
            hyp_text = normalize(r.get("asr", ""))
            we = edit_distance(ref_text.split(), hyp_text.split())
            ce = edit_distance(list(ref_text.replace(" ", "")), list(hyp_text.replace(" ", "")))
            wn, cn = len(ref_text.split()), len(ref_text.replace(" ", ""))
            sim = sims[name].get(key)
            for bucket in (agg[name][truth[key]["lang_code"]], agg[name]["__all__"]):
                bucket["we"] += we; bucket["wn"] += wn
                bucket["ce"] += ce; bucket["cn"] += cn
                if sim is not None:
                    bucket["sim"] += sim; bucket["sn"] += 1
            per_row.append({
                "system": name, "audio": key, "lang_code": truth[key]["lang_code"],
                "ref": ref_text, "hyp": hyp_text,
                "wer": rate(we, wn), "cer": rate(ce, cn), "sim": sim,
            })

    names = list(systems)
    print()
    header = f"{'dialect':<16}{'n':>5}" + "".join(f"{n[:10]:>22}" for n in names)
    print(header)
    print(f"{'':<16}{'':>5}" + "".join(f"{'WER%':>8}{'CER%':>7}{'SIM':>7}" for _ in names))
    print("-" * len(header))
    dialects = sorted({d for n in names for d in agg[n] if d != "__all__"})
    for d in dialects + ["__all__"]:
        n_rows = sum(1 for r in per_row if r["system"] == names[0] and (d == "__all__" or r["lang_code"] == d))
        line = f"{'OVERALL' if d == '__all__' else d:<16}{n_rows:>5}"
        for name in names:
            b = agg[name].get(d)
            if not b:
                line += f"{'-':>22}"
                continue
            sim = b["sim"] / b["sn"] if b["sn"] else float("nan")
            line += f"{rate(b['we'], b['wn']):>8.2f}{rate(b['ce'], b['cn']):>7.2f}{sim:>7.3f}"
        print(line)

    if args.out:
        summary = {
            name: {
                d: {"wer": rate(b["we"], b["wn"]), "cer": rate(b["ce"], b["cn"]),
                    "sim": (b["sim"] / b["sn"]) if b["sn"] else None}
                for d, b in agg[name].items()
            }
            for name in names
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "rows": per_row}, fh, ensure_ascii=False, indent=1)
        print(f"\n[score] wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# coding=utf-8
"""Convert a Higgs Audio v3 finetuned checkpoint to the **native** format that
``sglang-omni`` and ``vllm-omni`` serve (i.e. the layout of
``bosonai/higgs-audio-v3-tts-4b``).

Why this is needed
------------------
This finetuning toolkit trains on the model class
``HiggsMultimodalQwen3ForConditionalGeneration`` (``model/modeling.py``), which
loads the upstream *native* checkpoint through a key remapping
(``_HIGGS_KEY_MAPPING``)::

    tied.embedding.text_embedding.            -> model.embed_tokens.
    tied.embedding.modality_embeddings.0.embedding. -> audio_embedding.
    body.                                     -> model.

It also drops the (frozen) audio codec weights
(``tied.embedding.modality_embeddings.0.model.*``), loading that codec separately
from ``audio_tokenizer_id``. As a result a checkpoint produced by ``sft.py`` has
the **transformers-internal** key layout (``model.*``, ``audio_embedding.*``,
``audio_head.*``) and is **missing** the codec weights.

``sgl-omni serve`` / ``vllm-omni serve`` instead consume the **native** layout
(``body.*``, ``tied.embedding.*``, with the codec bundled). See sglang-omni's
``weight_loader.py`` (``DiscreteWeightMapper``) — it maps the very same prefixes.

This script reverses the remap, copies the codec weights from a base native
checkpoint, and rewrites the native ``config.json`` / tokenizer so the output
directory is directly servable::

    sgl-omni serve /path/to/output --port 8000
    vllm-omni serve /path/to/output --host 0.0.0.0 --port 8095 --trust-remote-code --omni

Supports both **full-SFT** checkpoints and **LoRA** adapter directories
(auto-merged into the base before key remapping).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --------------------------------------------------------------------------- #
# Key mapping — exact inverse of ``model/modeling.py::_HIGGS_KEY_MAPPING``.
# Evaluated in ORDER: the first matching prefix wins, so the more specific
# ``model.embed_tokens.`` must be tried before the catch-all ``model.``.
# (Forward map, native -> transformers-internal, is the mirror image of this.)
# --------------------------------------------------------------------------- #
# transformers-internal key (finetune checkpoint)  ->  native key (omni serve)
REVERSE_KEY_MAPPING: list[tuple[str, str]] = [
    ("model.embed_tokens.", "tied.embedding.text_embedding."),
    ("audio_embedding.", "tied.embedding.modality_embeddings.0.embedding."),
    ("model.", "body."),
]
# Keys present in a finetune checkpoint that have no native counterpart.
# ``audio_head`` is tied with ``audio_embedding`` (``tie_weights`` in modeling.py)
# and is *not* kept as a separate native tensor.
DROP_KEYS = ("audio_head.",)

# The native ``bosonai`` checkpoint additionally carries two *tied* head tensors
# (``tied.head.text_head`` ≡ ``tied.embedding.text_embedding`` and
# ``tied.head.modality_heads.0`` ≡ the audio embedding) that are not listed in its
# ``model.safetensors.index.json`` and are dropped on load by the finetuning model
# (``_keys_to_ignore_on_load_unexpected``). sglang-omni/vllm-omni serve the model
# fine without them, but reproducing them makes the output a byte-faithful
# drop-in for the upstream checkpoint. ``--with-tied-heads`` turns this on.
TIED_HEADS = (
    # (native head key, native embedding key it is tied to)
    ("tied.head.text_head.weight", "tied.embedding.text_embedding.weight"),
    ("tied.head.modality_heads.0.weight",
     "tied.embedding.modality_embeddings.0.embedding.weight"),
)

# Native files copied verbatim from the base native checkpoint into the output.
NATIVE_AUX_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)

# Codec weights bundled in the native checkpoint (frozen, ignored at load time
# by the finetuning model, hence absent from finetune checkpoints).
CODEC_KEY_PREFIX = "tied.embedding.modality_embeddings.0.model."


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _load_safetensors_dir(directory: Path) -> Dict[str, torch.Tensor]:
    """Load every ``*.safetensors`` file in ``directory`` into one state dict.

    Keys are kept as-is (no remapping). Tensors are loaded onto CPU, share-free.
    """
    shard_files = sorted(directory.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files found in {directory}")
    state: Dict[str, torch.Tensor] = {}
    for shard in shard_files:
        from safetensors import safe_open

        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key)
    return state


def _reverse_map_keys(
    state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Remap transformers-internal keys -> native keys (inverse of load mapping).

    Drops tied ``audio_head`` and any stray codec keys.
    """
    out: Dict[str, torch.Tensor] = {}
    dropped = 0
    for key, tensor in state.items():
        if any(key.startswith(d) for d in DROP_KEYS):
            dropped += 1
            continue
        # Codec weights should never be present in a finetune checkpoint; skip
        # defensively in case a user points us at a native dir by mistake.
        if key.startswith(CODEC_KEY_PREFIX):
            dropped += 1
            continue
        mapped: Optional[str] = None
        for src_prefix, dst_prefix in REVERSE_KEY_MAPPING:
            if key.startswith(src_prefix):
                mapped = dst_prefix + key[len(src_prefix):]
                break
        if mapped is None:
            # Unknown key — keep verbatim so we surface it rather than silently
            # dropping finetuned weights. (Verify step will flag the mismatch.)
            mapped = key
        if mapped in out:
            raise ValueError(f"Duplicate native key after remap: {mapped!r}")
        out[mapped] = tensor
    if dropped:
        print(f"  dropped {dropped} tied/codec tensor(s) (audio_head / codec)")
    return out


def _collect_codec_weights(base_native: Path) -> Dict[str, torch.Tensor]:
    """Pull the frozen codec weights (``...model.*``) from a base native dir."""
    full = _load_safetensors_dir(base_native)
    codec = {k: v for k, v in full.items() if k.startswith(CODEC_KEY_PREFIX)}
    if not codec:
        raise ValueError(
            f"No codec weights ({CODEC_KEY_PREFIX}*) found in base native dir "
            f"{base_native!s}. Is it the bosonai native checkpoint?"
        )
    return codec


def _ensure_dtype(
    state: Dict[str, torch.Tensor], dtype: str
) -> Dict[str, torch.Tensor]:
    """Cast every tensor to ``dtype`` (bf16 by default, matching the base)."""
    if dtype == "keep":
        return state
    torch_dtype = getattr(torch, dtype)
    return {k: (v if v.dtype == torch_dtype else v.to(torch_dtype)) for k, v in state.items()}


def _shard_and_save(
    state: Dict[str, torch.Tensor], out_dir: Path, max_shard_bytes: int
) -> None:
    """Write ``state`` as sharded ``model-XXXXX-of-YYYYY.safetensors`` + index.

    Greedy first-fit bin packing by byte size, mirroring HF's layout. The index
    records per-key shard filenames and the global metadata.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # If everything fits in one shard, HF names it ``model.safetensors``.
    total_bytes = sum(t.nbytes for t in state.values())
    keys_sorted = sorted(state.keys())

    if total_bytes <= max_shard_bytes:
        shard_name = "model.safetensors"
        save_file({k: state[k] for k in keys_sorted}, str(out_dir / shard_name),
                  metadata={"format": "pt"})
        weight_map = {k: shard_name for k in keys_sorted}
        index = {
            "metadata": {"total_size": total_bytes},
            "weight_map": weight_map,
        }
        (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
        print(f"  wrote {shard_name} ({total_bytes / 1e9:.2f} GB, single shard)")
        return

    # Multi-shard: greedy first-fit.
    shards: list[Dict[str, torch.Tensor]] = []
    shard_sizes: list[int] = []
    assignment: Dict[str, int] = {}
    for key in keys_sorted:
        t = state[key]
        placed = False
        for i, size in enumerate(shard_sizes):
            if size + t.nbytes <= max_shard_bytes:
                shards[i][key] = t
                shard_sizes[i] += t.nbytes
                assignment[key] = i
                placed = True
                break
        if not placed:
            shards.append({key: t})
            shard_sizes.append(t.nbytes)
            assignment[key] = len(shards) - 1

    n = len(shards)
    weight_map: Dict[str, str] = {}
    for i, shard in enumerate(shards):
        shard_name = f"model-{i + 1:05d}-of-{n:05d}.safetensors"
        save_file(shard, str(out_dir / shard_name), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = shard_name
        print(f"  wrote {shard_name} ({shard_sizes[i] / 1e9:.2f} GB)")
    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))


def _copy_aux_files(base_native: Path, out_dir: Path) -> list[str]:
    """Copy config + tokenizer from the base native dir; report what was copied."""
    copied: list[str] = []
    for name in NATIVE_AUX_FILES:
        src = base_native / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
            copied.append(name)
        else:
            print(f"  WARNING: base native dir missing {name!r}; not copied")
    return copied


# --------------------------------------------------------------------------- #
# Checkpoint type detection + state-dict acquisition
# --------------------------------------------------------------------------- #
def is_lora_checkpoint(checkpoint: Path) -> bool:
    return (checkpoint / "adapter_config.json").exists()


def _load_lora_base_repo_id(adapter_config_path: Path) -> str:
    cfg = json.loads(adapter_config_path.read_text())
    base = cfg.get("base_model_name_or_path")
    if not base:
        raise ValueError(
            f"{adapter_config_path!s} has no 'base_model_name_or_path'; pass "
            f"--lora-base explicitly."
        )
    return base


def get_transformers_state_dict(
    checkpoint: Path, lora_base: Optional[str]
) -> Dict[str, torch.Tensor]:
    """Return the finetuned weights as a *transformers-internal-key* state dict.

    * full-SFT checkpoint: load its ``.safetensors`` directly (no model build).
    * LoRA adapter: build the base + adapter on CPU, merge, then return the
      merged state dict. Mirrors ``scripts/infer.py``'s load/merge flow.
    """
    if is_lora_checkpoint(checkpoint):
        return _merge_lora_to_state_dict(checkpoint, lora_base)
    return _load_safetensors_dir(checkpoint)


def _merge_lora_to_state_dict(
    checkpoint: Path, lora_base: Optional[str]
) -> Dict[str, torch.Tensor]:
    """Load a LoRA adapter on its base, merge it, and return the merged state dict.

    Runs entirely on CPU to avoid GPU OOM (a 4B model fits in ~9 GB bf16 RAM).

    We merge rather than export the adapter because neither sglang-omni nor
    vllm-omni currently supports serving a LoRA on Higgs TTS v3: vllm-omni's
    ``HiggsAudioV3TalkerForConditionalGeneration`` does not implement
    ``SupportsLoRA`` (and its LoRA path is gated on ``stage_type == "diffusion"``),
    and sglang-omni's Higgs engine builder passes no LoRA args. See the README
    "Convert" section for details. This mirrors ``scripts/infer.py``'s load/merge.
    """
    import os

    # The custom model class lazily loads the codec only on generation; merging
    # never triggers it, so we can keep CUDA off entirely.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    base_repo = lora_base or _load_lora_base_repo_id(checkpoint / "adapter_config.json")
    print(f"  LoRA detected; merging adapter into base {base_repo!r} (CPU, bf16) ...")

    from model.modeling import HiggsMultimodalQwen3ForConditionalGeneration  # noqa: WPS433

    model = HiggsMultimodalQwen3ForConditionalGeneration.from_pretrained(
        base_repo,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(checkpoint))
    model = model.merge_and_unload()

    # ``state_dict`` here still carries any PEFT bookkeeping prefixes; the merged
    # base module is the unwrapped Higgs model, so keys are transformers-internal.
    return {k: v.cpu() for k, v in model.state_dict().items()}


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify_against_native(out_dir: Path, base_native: Path) -> bool:
    """Compare the converted output's key set + spot-checksums vs base native.

    Reads the actual tensor keys from the base native safetensors files rather
    than from ``model.safetensors.index.json``: the bosonai index omits the two
    tied ``tied.head.*`` tensors that the checkpoint does contain, so the on-disk
    keys are the source of truth.
    """
    out_state = _load_safetensors_dir(out_dir)
    native_state = _load_safetensors_dir(base_native)
    native_keys = set(native_state.keys())
    out_keys = set(out_state.keys())

    print("\n=== verify ===")
    print(f"  converted keys : {len(out_keys)}")
    print(f"  native keys    : {len(native_keys)}")

    missing = native_keys - out_keys
    extra = out_keys - native_keys
    if missing:
        print(f"  MISSING ({len(missing)}): {sorted(missing)[:5]} ...")
    if extra:
        print(f"  EXTRA ({len(extra)}): {sorted(extra)[:5]} ...")
    ok_keys = not missing and not extra
    print(f"  key set match : {'OK' if ok_keys else 'FAIL'}")

    # Spot-check a few critical tensors against the base native weights.
    # (For a true finetune these will of course differ; this just confirms the
    # remap targets the right tensors. For a round-trip identity test they match.)
    spot_keys = [
        "tied.embedding.text_embedding.weight",
        "tied.embedding.modality_embeddings.0.embedding.weight",
        "body.layers.0.self_attn.q_proj.weight",
        # A codec tensor — confirms codec weights were copied intact.
        "tied.embedding.modality_embeddings.0.model.quantizer.vq.layers.0.codebook",
    ]
    print("  spot-check (output vs base native):")
    all_match = ok_keys
    for k in spot_keys:
        if k not in out_state or k not in native_state:
            continue
        same = torch.equal(out_state[k], native_state[k])
        flag = "match" if same else "differ (expected for a real finetune)"
        print(f"    {k[:60]:60s} {flag}")
    return ok_keys


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert a Higgs Audio v3 finetuned checkpoint (transformers-internal "
            "key layout) to the native layout served by sglang-omni / vllm-omni."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Full-SFT checkpoint\n"
            "  python scripts/convert_to_omni.py \\\n"
            "      --checkpoint output/higgs_sft_full/checkpoint-epoch-0 \\\n"
            "      --output higgs-native-omni\n\n"
            "  # LoRA adapter (auto-merged into the base)\n"
            "  python scripts/convert_to_omni.py \\\n"
            "      --checkpoint output/higgs_sft_lora/checkpoint-epoch-0 \\\n"
            "      --output higgs-native-omni\n\n"
            "  # Then serve:\n"
            "  sgl-omni serve /abs/path/to/higgs-native-omni --port 8000\n"
            "  vllm-omni serve /abs/path/to/higgs-native-omni --trust-remote-code --omni"
        ),
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Finetune checkpoint dir: full-SFT model dir OR LoRA adapter dir.")
    p.add_argument("--output", type=Path, required=True,
                   help="Output dir (native layout, directly servable).")
    p.add_argument("--base", type=str, default="bosonai/higgs-audio-v3-tts-4b",
                   help="Base NATIVE checkpoint (repo id or local path) to copy codec "
                        "weights + config/tokenizer from. Default: bosonai native.")
    p.add_argument("--lora-base", type=str, default=None,
                   help="Base (transformers-format) to merge a LoRA adapter into. "
                        "Default: read from adapter_config.json -> base_model_name_or_path.")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32", "keep"],
                   help="Cast weights to this dtype. 'keep' preserves the checkpoint dtype.")
    p.add_argument("--max-shard-size", type=str, default="5GB",
                   help="Max safetensors shard size, e.g. 5GB / 2GB. Single shard if it fits.")
    p.add_argument("--with-tied-heads", action="store_true",
                   help="Also write the two tied 'tied.head.*' tensors present in the upstream "
                        "bosonai native checkpoint (cloned from their tied embedding), so the "
                        "output is a byte-faithful drop-in. Off by default; serving works "
                        "either way.")
    p.add_argument("--dry-run", action="store_true",
                   help="Plan only: report detected type, key counts, and what would happen.")
    p.add_argument("--verify", action="store_true",
                   help="After writing, compare the output key set against the base native.")
    return p.parse_args()


def _resolve_base_local(base: str) -> Path:
    """Return a local directory for ``base`` (download from the Hub if needed)."""
    base_path = Path(base)
    if base_path.exists() and base_path.is_dir():
        return base_path
    print(f"  base {base!r} not local; downloading from the Hub ...")
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        base,
        allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.txt"],
    )
    return Path(local)


def _parse_bytes(spec: str) -> int:
    spec = spec.strip()
    mult = 1
    if spec.endswith("GB"):
        mult, spec = 10**9, spec[:-2]
    elif spec.endswith("MB"):
        mult, spec = 10**6, spec[:-2]
    return int(float(spec) * mult)


def main() -> int:
    args = parse_args()
    checkpoint: Path = args.checkpoint.resolve()
    output: Path = args.output.resolve()

    if not checkpoint.exists():
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2

    is_lora = is_lora_checkpoint(checkpoint)
    print("=" * 70)
    print(f"checkpoint : {checkpoint}")
    print(f"type       : {'LoRA adapter' if is_lora else 'full-SFT'}")
    print(f"output     : {output}")
    print(f"base native: {args.base}")
    if is_lora:
        print(f"lora base  : {args.lora_base or '(from adapter_config.json)'}")
    print("=" * 70)

    base_native = _resolve_base_local(args.base)

    if args.dry_run:
        print("\n[dry-run] would:")
        if is_lora:
            print(f"  1. merge LoRA from {checkpoint} into base "
                  f"({args.lora_base or 'adapter_config base'})")
        else:
            print(f"  1. load full-SFT safetensors from {checkpoint}")
        print("  2. reverse key map (model.* -> body.*, embed_tokens -> tied..., "
              f"drop audio_head){', reproduce tied heads' if args.with_tied_heads else ''}")
        print(f"  3. copy codec weights from {base_native}")
        print(f"  4. cast to {args.dtype}, shard at {args.max_shard_size}")
        print(f"  5. write native config/tokenizer + safetensors to {output}")
        return 0

    # 1. Acquire the finetuned weights in transformers-internal key layout.
    print("\n[1/5] loading finetuned weights (transformers-internal keys) ...")
    ft_state = get_transformers_state_dict(checkpoint, args.lora_base)
    print(f"  loaded {len(ft_state)} tensors")

    # 2. Reverse the key mapping -> native layout.
    print("\n[2/5] reversing key map to native layout ...")
    native_from_ft = _reverse_map_keys(ft_state)
    print(f"  {len(native_from_ft)} native tensors from finetune")

    # 3. Copy frozen codec weights from the base native checkpoint.
    print(f"\n[3/5] copying codec weights from {base_native} ...")
    codec = _collect_codec_weights(base_native)
    print(f"  {len(codec)} codec tensors")
    merged: Dict[str, torch.Tensor] = {**native_from_ft, **codec}
    # Guard against accidental overlap.
    overlap = set(native_from_ft) & set(codec)
    if overlap:
        raise ValueError(f"Codec/finetune key overlap (should not happen): {sorted(overlap)[:3]}")

    # Optionally reproduce the two tied ``tied.head.*`` tensors present in the
    # upstream native checkpoint (cloned from their tied embedding so safetensors
    # serializes them as independent copies, matching the bosonai layout).
    if args.with_tied_heads:
        added = 0
        for head_key, emb_key in TIED_HEADS:
            if emb_key in merged:
                merged[head_key] = merged[emb_key].clone()
                added += 1
        print(f"  reproduced {added} tied head tensor(s) (--with-tied-heads)")

    # 4. Cast dtype + shard + write.
    print(f"\n[4/5] casting to {args.dtype} and writing shards ...")
    merged = _ensure_dtype(merged, args.dtype)
    output.mkdir(parents=True, exist_ok=True)
    _shard_and_save(merged, output, _parse_bytes(args.max_shard_size))

    # 5. Copy native config + tokenizer.
    print(f"\n[5/5] copying native config/tokenizer from {base_native} ...")
    copied = _copy_aux_files(base_native, output)
    print(f"  copied: {', '.join(copied)}")

    print(f"\nDone. Servable native checkpoint at:\n  {output}")
    print("\nServe with:")
    print(f"  sgl-omni serve {output} --port 8000")
    print(f"  vllm-omni serve {output} --host 0.0.0.0 --port 8095 "
          f"--trust-remote-code --omni")

    if args.verify:
        ok = verify_against_native(output, base_native)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

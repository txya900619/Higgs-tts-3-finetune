# Higgs Audio v3 TTS — Fine-tuning Toolkit

Fine-tune the [Higgs Audio v3](https://huggingface.co/multimodalart/higgs-audio-v3-tts-4b-transformers) Text-to-Speech model on your own data with full-parameter SFT or LoRA. Supports multi-GPU training via Accelerate / FSDP.

## Install

```bash
uv sync
```

Optional extras — both are imported lazily, so you only need them if you use the feature:
```bash
uv sync --extra lora        # LoRA training (peft)
uv sync --extra deepspeed   # DeepSpeed ZeRO
```

> PyTorch with CUDA 13.0 is pinned in `pyproject.toml`. If your CUDA version differs, change the `pytorch-cu130` index URL and the `torch` / `torchaudio` / `torchcodec` pins together — the three must come from the same CUDA build.

> Audio file I/O goes through **torchcodec**, not torchaudio. torchaudio remains a dependency only because transformers' `higgs_audio_v2_tokenizer` codec requires it internally.

## Prepare Data

Create a JSONL file where each line has the following format:

```jsonc
{
  "audio": "/path/to/target_audio.wav",       // required
  "text": "The transcript of the audio.",      // required
  "ref_audio": "/path/to/reference.wav",       // required for voice cloning
  "ref_text": "Transcript of reference audio",       // optional for voice cloning
}
```

Then extract audio codes with the Higgs codec:

```bash
python scripts/prepare_data.py \
    --codec-path bosonai/higgs-audio-v2-tokenizer \
    --input-jsonl data/train.jsonl \
    --output-jsonl data/train_with_codes.jsonl \
    --device cuda
```

Multi-GPU data preparation is also supported:
```bash
accelerate launch scripts/prepare_data.py \
    --input-jsonl data/train.jsonl \
    --output-jsonl data/train_with_codes.jsonl
```

## Train

### LoRA (recommended)

```bash
accelerate launch sft.py \
    --model-path multimodalart/higgs-audio-v3-tts-4b-transformers \
    --train-jsonl data/train_with_codes.jsonl \
    --output-dir output/higgs_sft_lora \
    --use-lora \
    --lora-rank 16 \
    --lora-alpha 32 \
    --per-device-batch-size 4 \
    --gradient-accumulation-steps 8 \
    --learning-rate 2e-5 \
    --num-epochs 3 \
    --mixed-precision bf16 \
    --gradient-checkpointing
```

### Full fine-tuning

```bash
accelerate launch sft.py \
    --model-path multimodalart/higgs-audio-v3-tts-4b-transformers \
    --train-jsonl data/train_with_codes.jsonl \
    --output-dir output/higgs_sft_full \
    --per-device-batch-size 1 \
    --gradient-accumulation-steps 16 \
    --learning-rate 1e-5 \
    --num-epochs 3 \
    --mixed-precision bf16 \
    --gradient-checkpointing
```

For the full list of arguments, run `python sft.py --help`.

## Inference

Base model:
```bash
python scripts/infer.py \
    --model-path multimodalart/higgs-audio-v3-tts-4b-transformers \
    --text "Hello, this is a test." \
    --output output.wav
```

Voice cloning LoRA adapter (pass a reference audio):
```bash
python scripts/infer.py \
    --model-path multimodalart/higgs-audio-v3-tts-4b-transformers \
    --lora-path output/higgs_sft_lora/checkpoint-epoch-0 \
    --ref-audio /path/to/reference.wav \
    --text "Hello, this is a test." \
    --output clone.wav
```

For all options, run `python scripts/infer.py --help`.

## Convert to sglang-omni / vllm-omni

Checkpoints produced by this toolkit use the **transformers-internal** weight-key
layout (`model.*`, `audio_embedding.*`, ...) inherited from the custom model class
in `model/modeling.py`. Production servers — **[sglang-omni](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html)**
and **[vllm-omni](https://github.com/vllm-project/vllm-omni)** — instead serve the
**native** layout of [`bosonai/higgs-audio-v3-tts-4b`](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b)
(`body.*`, `tied.embedding.*`, with the frozen audio codec bundled in). The two
differ by a weight-key remap plus the codec weights, which finetuning drops on load.

`scripts/convert_to_omni.py` bridges that gap: it reverses the key remap, copies
the codec weights from a base native checkpoint, and rewrites the native
config/tokenizer — producing a directory that is directly servable. It supports
both **full-SFT** checkpoints and **LoRA** adapters (auto-merged into the base).

> **Why merge LoRA instead of serving it directly?** Core vLLM and SGLang both
> support efficient per-request / multi-LoRA serving, but only for model classes
> that declare the LoRA interface. As of writing, neither **vllm-omni** nor
> **sglang-omni** exposes LoRA serving for Higgs TTS v3:
> - In **vllm-omni**, `HiggsAudioV3TalkerForConditionalGeneration`
>   (`vllm_omni/model_executor/models/higgs_audio_v3/higgs_audio_v3_talker.py`)
>   does **not** implement `SupportsLoRA` (only the Qwen2.5/Qwen3 omni *thinkers*
>   and `HunyuanImage3` do), and the engine's LoRA-injection path is gated on
>   `stage_type == "diffusion"`, which Higgs's AR `tts_engine` stage is not.
> - In **sglang-omni**, there is no LoRA handling at all in the Higgs pipeline
>   (`HiggsTtsEngineBuilder` builds the sglang engine without passing any LoRA args).
>
> So for Higgs v3 today, **merge-then-serve is the only path** — which is exactly
> what this converter does for LoRA checkpoints (`merge_and_unload()` into the base
> before the key remap). If upstream adds `SupportsLoRA` to the Higgs talker, a
> "serve LoRA directly" mode could be added; track the upstream `higgs_audio_v3`
> model code for that change.

```bash
# Full-SFT checkpoint
python scripts/convert_to_omni.py \
    --checkpoint output/higgs_sft_full/checkpoint-epoch-0 \
    --output higgs-native-omni

# LoRA adapter (auto-merged into the base)
python scripts/convert_to_omni.py \
    --checkpoint output/higgs_sft_lora/checkpoint-epoch-0 \
    --output higgs-native-omni
```

Then serve with either engine:

```bash
# sglang-omni
sgl-omni serve /abs/path/to/higgs-native-omni --port 8000

# vllm-omni
vllm-omni serve /abs/path/to/higgs-native-omni \
    --host 0.0.0.0 --port 8095 --trust-remote-code --omni
```

Useful options:
- `--base` — base native checkpoint (repo id or local path) to copy codec weights
  + config/tokenizer from. Defaults to `bosonai/higgs-audio-v3-tts-4b` (downloaded
  automatically if not local).
- `--lora-base` — base (transformers-format) to merge a LoRA adapter into; defaults
  to `base_model_name_or_path` in `adapter_config.json`.
- `--dtype bfloat16` / `--max-shard-size 5GB` — control output precision and sharding.
- `--with-tied-heads` — also reproduce the two tied `tied.head.*` tensors present in
  the upstream bosonai checkpoint, for a byte-faithful drop-in (serving works either way).
- `--verify` — after writing, compare the output key set against the base native.
- `--dry-run` — report the detected checkpoint type and planned steps without writing.

> **Note on serving hardware:** the 4B model + KV cache + codec needs ~10–12 GB+ of
> VRAM. The conversion itself runs fine on CPU (a LoRA merge needs ~9 GB RAM).

For all options, run `python scripts/convert_to_omni.py --help`.

## Data Format

After running `prepare_data.py`, each JSONL record looks like:

```jsonc
{
  "audio_codes": [[...], ...],       // [T, 8] target audio codes
  "text": "...",                      // target text
  "ref_audio_codes": [[...], ...],   // [T_ref, 8] reference codes (optional)
}
```

## Requirements

- Python 3.10+
- PyTorch 2.9+ with CUDA
- transformers 5.0+
- accelerate >= 1.10.1
- peft (LoRA only)

## License

Please refer to the Higgs Audio v3 model license for usage terms.

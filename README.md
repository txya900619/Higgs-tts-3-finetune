# Higgs Audio v3 TTS — Fine-tuning Toolkit

Fine-tune the [Higgs Audio v3](https://huggingface.co/multimodalart/higgs-audio-v3-tts-4b-transformers) Text-to-Speech model on your own data with full-parameter SFT or LoRA. Supports multi-GPU training via Accelerate / FSDP.

## Install

```bash
pip install -r requirements.txt
```

For LoRA training, also install:
```bash
pip install peft
```

> PyTorch with CUDA 12.8 is pinned in `requirements.txt`. Adjust `torch` / `torchaudio` versions if your CUDA version differs.

## Prepare Data

Create a JSONL file where each line has the following format:

```jsonc
{
  "audio": "/path/to/target_audio.wav",       // required
  "text": "The transcript of the audio.",      // required
  "ref_audio": "/path/to/reference.wav",       // required for voice cloning
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

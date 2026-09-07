"""Inference script for Higgs Audio v3 TTS (base model or LoRA fine-tuned)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.modeling import HiggsMultimodalQwen3ForConditionalGeneration
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Higgs Audio v3 TTS inference.")
    parser.add_argument("--model-path", type=str, required=True,
                        help="HuggingFace repo ID or local path to the base model.")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Path to a LoRA adapter checkpoint. If not set, uses the base model only.")
    parser.add_argument("--text", type=str, required=True,
                        help="Text to synthesize.")
    parser.add_argument("--ref-audio", type=str, default=None,
                        help="Path to reference audio for voice cloning.")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Output WAV file path.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"Loading base model from {args.model_path} ...")
    model = HiggsMultimodalQwen3ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if args.lora_path:
        from peft import PeftModel

        print(f"Loading LoRA adapter from {args.lora_path} ...")
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()

    model = model.to("cuda").eval()

    ref_audio = None
    ref_sr = None
    if args.ref_audio:
        _samples = AudioDecoder(args.ref_audio).get_all_samples()
        ref_audio, ref_sr = _samples.data, _samples.sample_rate

    print(f"Generating speech ...")
    wav = model.generate_speech(
        args.text,
        tokenizer,
        reference_audio=ref_audio,
        reference_sample_rate=ref_sr,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )

    AudioEncoder(wav.unsqueeze(0), sample_rate=model.config.sample_rate).to_file(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

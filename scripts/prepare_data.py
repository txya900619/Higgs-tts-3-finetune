from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common import (
    dump_jsonl,
    load_jsonl,
    normalize_audio_path_list,
    resolve_shard_spec,
    select_rank_shard,
    shard_output_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Higgs Audio v3 finetuning JSONL by extracting audio codes."
    )
    parser.add_argument(
        "--codec-path",
        type=str,
        default="bosonai/higgs-audio-v2-tokenizer",
        help="HuggingFace repo ID or local path to the Higgs audio codec.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument(
        "--num-shards", type=int, default=None,
        help="Advanced override. Normally inferred from Accelerate world size.",
    )
    parser.add_argument(
        "--shard-rank", type=int, default=None,
        help="Advanced override. Normally inferred from Accelerate rank.",
    )
    parser.add_argument(
        "--skip-reference-audio-codes",
        dest="encode_reference_audio",
        action="store_false",
    )
    parser.set_defaults(encode_reference_audio=True)
    return parser.parse_args()


def load_codec(codec_path: str, device: str):
    """Load the Higgs audio codec."""
    from transformers import AutoModel

    codec = AutoModel.from_pretrained(
        codec_path, dtype=torch.float32, trust_remote_code=True
    )
    codec = codec.to(device).eval()
    for p in codec.parameters():
        p.requires_grad_(False)
    return codec


def encode_audio_file(
    codec,
    wav_path: str,
    sample_rate: int,
    device: str,
) -> torch.Tensor:
    """Encode a single audio file to ``[T, N]`` int64 codes."""
    import torch.nn.functional as F

    # Decode, downmix to mono and resample to `sample_rate` in one torchcodec
    # step (replaces torchaudio.load + mean + functional.resample).
    wav = AudioDecoder(wav_path, sample_rate=sample_rate, num_channels=1).get_all_samples().data.float()
    # Ensure minimum length of 1 second
    if wav.shape[-1] < sample_rate:
        wav = F.pad(wav, (0, sample_rate - wav.shape[-1]))

    wav = wav.unsqueeze(0).to(device, dtype=torch.float32)  # [1, C, L]
    with torch.no_grad():
        codes_BNT = codec.encode(wav).audio_codes  # [1, N, T]
    return codes_BNT.squeeze(0).transpose(0, 1).to(torch.long).cpu()  # [T, N]


def batch_encode_paths(
    codec,
    paths: List[str],
    sample_rate: int,
    device: str,
    batch_size: int,
    desc: str,
) -> List[torch.Tensor]:
    """Encode a list of audio paths, one at a time (batch_size for progress)."""
    all_codes: List[torch.Tensor] = []
    for i in tqdm(range(len(paths)), desc=desc):
        codes = encode_audio_file(codec, paths[i], sample_rate, device)
        all_codes.append(codes)
    return all_codes


def main() -> None:
    args = parse_args()

    try:
        from accelerate import Accelerator
        accelerator = Accelerator()
        default_device = str(accelerator.device)
        default_world_size = accelerator.num_processes
        default_rank = accelerator.process_index
    except ImportError:
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        default_world_size = 1
        default_rank = 0

    device = default_device if args.device == "auto" else args.device

    all_records = load_jsonl(args.input_jsonl)
    world_size, rank = resolve_shard_spec(
        args.num_shards,
        args.shard_rank,
        default_num_shards=default_world_size,
        default_shard_rank=default_rank,
    )
    records = select_rank_shard(all_records, world_size, rank)
    if not records:
        raise ValueError(
            f"No records for shard rank={rank} / world_size={world_size} in {args.input_jsonl}."
        )

    print(f"[prepare_data] Loading codec from {args.codec_path} ...")
    codec = load_codec(args.codec_path, device)

    # Encode target audio
    target_paths = []
    for idx, record in enumerate(records):
        audio_path = record.get("audio")
        if not isinstance(audio_path, str) or not audio_path:
            raise ValueError(f"Record {idx} is missing a valid `audio` field.")
        target_paths.append(audio_path)

    target_codes = batch_encode_paths(
        codec, target_paths, args.sample_rate, device,
        args.batch_size, desc="Encoding target audio",
    )
    # Reference audio is drawn from the same manifest, so nearly every ref file
    # is also some other row's target -- 171,599 of 171,614 unique refs in the
    # Formosan train set. Keeping the target codes keyed by path lets the
    # reference pass skip re-encoding them: 1.63x fewer encodes unsharded, 1.28x
    # at two shards (a shard only holds a fraction of the files its refs point
    # at). Encoding is deterministic for a given input, so reuse is exact, not
    # an approximation.
    encoded_by_path: Dict[str, Any] = {}
    for record, path, codes in zip(records, target_paths, target_codes):
        listed = codes.tolist()
        record["audio_codes"] = listed
        encoded_by_path[path] = listed

    # Optionally encode reference audio
    if args.encode_reference_audio:
        ref_paths_to_encode: Dict[str, None] = {}
        for record in records:
            ref_audio = record.get("ref_audio")
            if isinstance(ref_audio, str) and ref_audio:
                ref_paths_to_encode[ref_audio] = None

        if ref_paths_to_encode:
            path_to_codes = {
                path: encoded_by_path[path]
                for path in ref_paths_to_encode
                if path in encoded_by_path
            }
            unique_paths = [p for p in ref_paths_to_encode if p not in encoded_by_path]
            print(
                f"[prepare_data] reference audio: {len(ref_paths_to_encode)} unique, "
                f"{len(path_to_codes)} reused from target encodes, "
                f"{len(unique_paths)} still to encode"
            )
            ref_codes = batch_encode_paths(
                codec, unique_paths, args.sample_rate, device,
                args.batch_size, desc="Encoding reference audio",
            ) if unique_paths else []
            path_to_codes.update({
                path: codes.tolist()
                for path, codes in zip(unique_paths, ref_codes)
            })
            for record in records:
                ref_audio = record.get("ref_audio")
                if isinstance(ref_audio, str) and ref_audio and ref_audio in path_to_codes:
                    record["ref_audio_codes"] = path_to_codes[ref_audio]

    # Write output
    output_path = args.output_jsonl
    if world_size > 1:
        output_path = str(shard_output_path(args.output_jsonl, rank, world_size))
    dump_jsonl(records, output_path)
    print(
        f"[prepare_data] rank={rank}/{world_size} "
        f"input_records={len(all_records)} local_records={len(records)} "
        f"device={device} output={output_path}"
    )


if __name__ == "__main__":
    main()

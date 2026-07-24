# coding=utf-8
"""Higgs Audio v3 SFT dataset.

Loads JSONL records with ``audio_codes`` (precomputed by ``prepare_data.py``)
and optional reference audio fields, then packs them into teacher-forcing
training samples via ``HiggsAudioProcessor``.

Expected JSONL format:
    {"audio_codes": [[...], ...], "text": "...", "ref_audio_codes": [[...], ...], "ref_text": "..."}

Where:
    - ``audio_codes``: required — ``[T, N]`` list-of-lists, target audio codes
    - ``text``: required — target text
    - ``ref_audio_codes``: optional — ``[T_ref, N]`` reference audio codes
    - ``ref_text``: optional — reference audio transcript
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .processor import HiggsAudioProcessor, BOC_ID


def normalize_audio_codes(value: Any, field_name: str) -> torch.Tensor:
    """Convert list-of-lists or tensor to a ``[T, N]`` int64 tensor."""
    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.ndim != 2:
        raise ValueError(f"`{field_name}` must have shape (T, N), got {tuple(tensor.shape)}.")
    return tensor.cpu().contiguous()


class HiggsAudioSFTDataset(Dataset):
    """SFT dataset for Higgs Audio v3 training.

    Args:
        records: List of dicts loaded from JSONL.
        processor: ``HiggsAudioProcessor`` instance.
        num_codebooks: Number of codebooks (for validation).
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        processor: HiggsAudioProcessor,
        num_codebooks: Optional[int] = None,
    ) -> None:
        self.records = list(records)
        self.processor = processor
        self.num_codebooks = num_codebooks

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self._pack_record(self.records[index])

    def _pack_record(self, record: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert a single JSONL record to a training sample."""
        if "audio_codes" not in record:
            raise ValueError("Each record must contain `audio_codes`. Run prepare_data.py first.")
        if "text" not in record:
            raise ValueError("Each record must contain `text`.")

        target_codes = normalize_audio_codes(record["audio_codes"], "audio_codes")
        target_n = int(target_codes.shape[1])
        if self.num_codebooks is not None and target_n != self.num_codebooks:
            raise ValueError(
                f"Expected num_codebooks={self.num_codebooks}, got {target_n}."
            )

        # Optional reference audio
        ref_codes = None
        if record.get("ref_audio_codes") is not None:
            ref_codes = normalize_audio_codes(record["ref_audio_codes"], "ref_audio_codes")
            if ref_codes.shape[1] != target_n:
                raise ValueError(
                    f"ref_audio_codes num_codebooks={ref_codes.shape[1]} != target {target_n}."
                )

        ref_text = record.get("ref_text")

        return self.processor.build_training_sample(
            text=record["text"],
            target_audio_codes=target_codes,
            reference_codes=ref_codes,
            reference_text=ref_text,
        )

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader."""
        return self.processor.pad_and_collate(batch)

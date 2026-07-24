from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizerBase

# Import delay pattern and constants from the model module
BOC_ID = 1024
EOC_ID = 1025
AUDIO_PLACEHOLDER_ID = -100

_REQUIRED_SPECIALS = ("<|tts|>", "<|ref_audio|>", "<|text|>", "<|audio|>")
_OPTIONAL_SPECIALS = ("<|ref_text|>",)


def apply_delay_pattern(codes_TN: torch.Tensor) -> torch.Tensor:
    """``[T, N]`` raw codes -> ``[T + N - 1, N]`` delayed, BOC/EOC padded."""
    T, N = codes_TN.shape
    out = torch.full(
        (T + N - 1, N), EOC_ID, device=codes_TN.device, dtype=codes_TN.dtype
    )
    t_idx = torch.arange(T + N - 1, device=codes_TN.device)
    for c in range(N):
        out[t_idx < c, c] = BOC_ID
        out[c : c + T, c] = codes_TN[:, c]
    return out


class HiggsAudioProcessor:
    """Builds training sequences for Higgs Audio v3 model.

    This processor assembles mixed text/audio sequences following the Higgs
    prompt format. It handles:
    - Reference audio embedding (optional, for voice cloning)
    - Delay pattern application to target audio codes
    - Teacher-forcing sequence construction with proper masking

    Args:
        tokenizer: HuggingFace tokenizer (from the Higgs model).
        num_codebooks: Number of audio codebooks (N).
        codebook_vocab_size: Vocabulary size per codebook (V), including BOC/EOC.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_codebooks: int = 8,
        codebook_vocab_size: int = 1026,
    ):
        self.tokenizer = tokenizer
        self.num_codebooks = num_codebooks
        self.codebook_vocab_size = codebook_vocab_size

        # Resolve special token IDs
        vocab = dict(tokenizer.get_added_vocab())
        missing = [t for t in _REQUIRED_SPECIALS if t not in vocab]
        if missing:
            raise ValueError(f"Tokenizer is missing Higgs TTS specials: {missing}")

        self.special_ids = {t: vocab[t] for t in _REQUIRED_SPECIALS}
        self.special_ids["<|ref_text|>"] = vocab.get("<|ref_text|>")

    def build_training_sample(
        self,
        text: str,
        target_audio_codes: torch.Tensor,
        *,
        reference_codes: Optional[torch.Tensor] = None,
        reference_text: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build a single training sample.

        Args:
            text: Target text to synthesize.
            target_audio_codes: ``[T, N]`` raw (un-delayed) audio codes for the target.
            reference_codes: ``[T_ref, N]`` raw (un-delayed) codes for voice cloning.
            reference_text: Transcript of the reference audio.

        Returns:
            Dict with keys: ``input_ids``, ``audio_codes``, ``audio_mask``, ``loss_mask``.
        """
        N = self.num_codebooks
        sp = self.special_ids

        # --- Apply delay pattern ---
        delayed_target = apply_delay_pattern(target_audio_codes.to(torch.long))  # [T+N-1, N]

        delayed_ref = None
        if reference_codes is not None:
            delayed_ref = apply_delay_pattern(reference_codes.to(torch.long))  # [T_ref+N-1, N]

        # --- Build text token sequence (prompt portion) ---
        prompt_token_ids: List[int] = [sp["<|tts|>"]]

        # Optional: reference text
        if reference_text and delayed_ref is not None and sp["<|ref_text|>"] is not None:
            prompt_token_ids.append(sp["<|ref_text|>"])
            prompt_token_ids.extend(
                self.tokenizer.encode(reference_text, add_special_tokens=False)
            )

        # Reference audio markers + placeholders
        num_ref_tokens = 0
        if delayed_ref is not None:
            num_ref_tokens = delayed_ref.shape[0]
            prompt_token_ids.append(sp["<|ref_audio|>"])
            prompt_token_ids.extend([0] * num_ref_tokens)  # placeholder IDs

        # Target text
        prompt_token_ids.append(sp["<|text|>"])
        prompt_token_ids.extend(
            self.tokenizer.encode(text, add_special_tokens=False)
        )

        # Audio marker (start of generation)
        prompt_token_ids.append(sp["<|audio|>"])

        prompt_len = len(prompt_token_ids)
        audio_gen_len = delayed_target.shape[0]
        total_len = prompt_len + audio_gen_len

        # --- Build full input_ids [S] ---
        input_ids = torch.zeros(total_len, dtype=torch.long)
        input_ids[:prompt_len] = torch.tensor(prompt_token_ids, dtype=torch.long)
        # Audio generation portion: fill with 0 (will be overridden by audio_embedding via audio_mask)
        # input_ids[prompt_len:] remains 0

        # --- Build audio_codes [S, N] ---
        audio_codes = torch.full((total_len, N), BOC_ID, dtype=torch.long)  # PAD with BOC

        # Fill reference audio positions
        if delayed_ref is not None:
            ref_audio_start = prompt_len - audio_gen_len - num_ref_tokens  # position after <|ref_audio|>
            # Find the <|ref_audio|> position
            ref_audio_marker_pos = None
            for i, tid in enumerate(prompt_token_ids):
                if tid == sp["<|ref_audio|>"]:
                    ref_audio_marker_pos = i
                    break
            if ref_audio_marker_pos is not None:
                ref_start = ref_audio_marker_pos + 1
                ref_end = ref_start + num_ref_tokens
                audio_codes[ref_start:ref_end] = delayed_ref

        # Fill target audio generation positions
        audio_codes[prompt_len:] = delayed_target

        # --- Build audio_mask [S] ---
        audio_mask = torch.zeros(total_len, dtype=torch.bool)
        # Reference audio positions
        if delayed_ref is not None and ref_audio_marker_pos is not None:
            ref_start = ref_audio_marker_pos + 1
            ref_end = ref_start + num_ref_tokens
            audio_mask[ref_start:ref_end] = True
        # Target audio generation positions
        audio_mask[prompt_len:] = True

        # --- Build loss_mask [S-1] ---
        # Loss is computed only on the audio generation portion (teacher-forcing)
        # For position i, the label is the token at position i+1
        # So loss_mask[i] = True means we compute loss for predicting position i+1
        loss_mask = torch.zeros(total_len - 1, dtype=torch.bool)
        # We want loss for predicting positions prompt_len...(total_len-1)
        # That means loss_mask[prompt_len-1...(total_len-2)] = True
        loss_mask[prompt_len - 1 :] = True

        return {
            "input_ids": input_ids,
            "audio_codes": audio_codes,
            "audio_mask": audio_mask,
            "loss_mask": loss_mask,
        }

    def pad_and_collate(
        self,
        samples: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Pad and collate a batch of training samples.

        Args:
            samples: List of dicts from ``build_training_sample()``.

        Returns:
            Batched dict with keys: ``input_ids``, ``audio_codes``, ``audio_mask``,
            ``attention_mask``, ``labels_audio``. All tensors are left-padded.
        """
        N = self.num_codebooks
        batch_size = len(samples)
        lengths = [s["input_ids"].shape[0] for s in samples]
        max_len = max(lengths)

        # Initialize with padding
        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        audio_codes = torch.full((batch_size, max_len, N), BOC_ID, dtype=torch.long)
        audio_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

        # Loss masks (for max_len - 1 positions)
        loss_masks = torch.zeros(batch_size, max_len - 1, dtype=torch.bool)

        for i, sample in enumerate(samples):
            seq_len = sample["input_ids"].shape[0]
            pad_len = max_len - seq_len  # left padding

            input_ids[i, pad_len:] = sample["input_ids"]
            audio_codes[i, pad_len:] = sample["audio_codes"]
            audio_mask[i, pad_len:] = sample["audio_mask"]
            attention_mask[i, pad_len:] = True

            # Loss mask: align to the right
            lm_len = sample["loss_mask"].shape[0]  # seq_len - 1
            loss_masks[i, pad_len:pad_len + lm_len] = sample["loss_mask"]

        # --- Construct labels_audio [B, S-1, N] ---
        # Labels are the shifted audio_codes: labels[t] = audio_codes[t+1]
        labels_audio = audio_codes[:, 1:, :].clone()  # [B, S-1, N]

        # Mask out non-loss positions
        labels_audio[~loss_masks.unsqueeze(-1).expand_as(labels_audio)] = -100

        # Mask out padding positions in attention
        labels_audio[~attention_mask[:, 1:].unsqueeze(-1).expand_as(labels_audio)] = -100

        # Mask out BOC_ID positions in labels (structural delay padding, not trainable)
        labels_audio = labels_audio.masked_fill(labels_audio == BOC_ID, -100)

        return {
            "input_ids": input_ids[:, :-1].contiguous(),
            "audio_codes": audio_codes[:, :-1].contiguous(),
            "audio_mask": audio_mask[:, :-1].contiguous(),
            "attention_mask": attention_mask[:, :-1].contiguous(),
            "labels_audio": labels_audio.contiguous(),
        }

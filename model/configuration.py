# coding=utf-8
"""Config for HiggsMultimodalQwen3 — the Higgs Audio v3 TTS model.
A standard Qwen3 text backbone plus a fused multi-codebook audio embedding /
head. Audio is encoded/decoded by the separately-loaded
``bosonai/higgs-audio-v2-tokenizer`` (``higgs_audio_v2_tokenizer``), which is
native to transformers >= 5.5.
"""

from __future__ import annotations

from typing import Any

from transformers import CONFIG_MAPPING, PretrainedConfig

# Higgs Qwen3 sub-configs ship ``rope_theta=null``; transformers' default of
# 10000 is wrong for Qwen3 (trained at 1e6). Patch before instantiation.
_QWEN3_ROPE_THETA = 1_000_000


def _build_text_config(raw: Any) -> PretrainedConfig:
    """Realise a text-backbone sub-config into a concrete ``PretrainedConfig``."""
    if isinstance(raw, PretrainedConfig):
        return raw
    cfg = dict(raw or {})
    model_type = cfg.get("model_type", "qwen3")
    if model_type == "qwen3":
        rope = cfg.get("rope_parameters") or {}
        if cfg.get("rope_theta") is None and rope.get("rope_theta") is None:
            cfg["rope_theta"] = _QWEN3_ROPE_THETA
    cfg_cls = CONFIG_MAPPING[model_type]
    return cfg_cls(**cfg)


_DEFAULT_AUDIO_ENCODER_CONFIG: dict[str, Any] = {
    "encoder_type": "discrete",
    "num_codebooks": 8,
    "vocab_size": 1026,
    "out_dim": 2560,
    "tie_word_embeddings": True,
    "use_delay_pattern": True,
    "model_type": "higgs_audio_encoder",
}


class HiggsMultimodalQwen3Config(PretrainedConfig):
    """Config for ``HiggsMultimodalQwen3ForConditionalGeneration``.
    Args:
        audio_encoder_config: discrete-codec descriptor. ``num_codebooks`` /
            ``vocab_size`` (incl. BOC/EOC specials) / ``out_dim`` /
            ``tie_word_embeddings`` drive the fused embedding + head.
        text_config: Qwen3 backbone config, eagerly realised so
            ``config.text_config.num_attention_heads`` works directly.
        audio_token_id: placeholder id (``-100``) marking reference-audio slots
            in ``input_ids`` that the fused audio embedding fills.
        audio_tokenizer_id: repo id of the codec used to encode reference audio
            and decode generated codes back to a waveform.
        sample_rate: codec sample rate, Hz.
    """

    model_type = "higgs_multimodal_qwen3"
    is_composition = True

    def __init__(
        self,
        audio_encoder_config: dict[str, Any] | None = None,
        text_config: dict[str, Any] | PretrainedConfig | None = None,
        audio_token_id: int = -100,
        mel_per_sample: int = 8,
        audio_tokenizer_id: str = "bosonai/higgs-audio-v2-tokenizer",
        sample_rate: int = 24_000,
        **kwargs,
    ):
        self.audio_token_id = audio_token_id
        self.mel_per_sample = mel_per_sample
        self.audio_tokenizer_id = audio_tokenizer_id
        self.sample_rate = sample_rate
        self.audio_encoder_config = audio_encoder_config or dict(
            _DEFAULT_AUDIO_ENCODER_CONFIG
        )
        self.text_config = _build_text_config(text_config)
        super().__init__(**kwargs)

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:
        del decoder
        return self.text_config


__all__ = ["HiggsMultimodalQwen3Config"]

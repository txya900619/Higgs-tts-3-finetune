# coding=utf-8
"""HiggsMultimodalQwen3 — Higgs Audio v3 TTS, ported to plain transformers.
Architecture:
  * a standard Qwen3 backbone (``Qwen3Model``) for the autoregressive LM;
  * one fused multi-codebook embedding ``[N*V, D]`` whose per-codebook lookups
    are summed — it both embeds reference-audio codes into the prompt and
    re-embeds each generated row during decoding;
  * a tied fused head ``[L, D] -> [L, N, V]`` producing per-codebook logits.
Audio I/O (waveform <-> discrete codes) is handled by the transformers-native
``HiggsAudioV2TokenizerModel`` (``bosonai/higgs-audio-v2-tokenizer``), loaded
lazily on first use.
Generation follows Higgs' delay pattern: codebook ``c`` is shifted by ``c``
steps, padded with BOC (1024) before and EOC (1025) after its data span. A
small per-request state machine drives the delay ramp-up, EOC detection and
wind-down; the produced rows are de-delayed and decoded to a 24 kHz waveform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

from .configuration import HiggsMultimodalQwen3Config

# Codec-vocab specials, inside the per-codebook [0, V) space (NOT the text vocab).
BOC_ID = 1024
EOC_ID = 1025
# Placeholder id marking reference-audio slots in ``input_ids``.
AUDIO_PLACEHOLDER_ID = -100

_REQUIRED_SPECIALS = ("<|tts|>", "<|ref_audio|>", "<|text|>", "<|audio|>")


# --------------------------------------------------------------------------- #
# Training output
# --------------------------------------------------------------------------- #
@dataclass
class HiggsAudioOutputWithPast(ModelOutput):
    """Output type for Higgs Audio training forward pass.

    Args:
        loss: Weighted scalar loss (when ``labels_audio`` is provided).
        all_sum_losses: Per-sample, per-codebook summed losses ``[B, N]``.
        all_token_nums: Valid (non-masked) token count per sample per codebook ``[B, N]``.
        channel_losses: Mean loss per codebook ``[N]``.
        logits: Per-codebook logits ``[B, S, N, V]``.
        past_key_values: Cached KV states for incremental decoding.
        hidden_states: Intermediate hidden states (when requested).
        attentions: Attention weights (when requested).
    """
    loss: Optional[torch.FloatTensor] = None
    all_sum_losses: Optional[torch.FloatTensor] = None
    all_token_nums: Optional[torch.LongTensor] = None
    channel_losses: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# --------------------------------------------------------------------------- #
# Delay pattern
# --------------------------------------------------------------------------- #
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


def reverse_delay_pattern(delayed_LN: torch.Tensor) -> torch.Tensor:
    """``[L, N]`` delayed (L >= N) -> ``[L - (N - 1), N]`` raw codes."""
    L, N = delayed_LN.shape
    T = L - (N - 1)
    if T <= 0:
        raise ValueError(f"delayed rows L={L} < num_codebooks N={N}")
    out = torch.empty((T, N), device=delayed_LN.device, dtype=delayed_LN.dtype)
    for c in range(N):
        out[:, c] = delayed_LN[c : c + T, c]
    return out


# --------------------------------------------------------------------------- #
# Fused multi-codebook modules
# --------------------------------------------------------------------------- #
class HiggsFusedMultiTextEmbedding(nn.Module):
    """Fused multi-codebook embedding: one ``[N*V, D]`` weight + offset lookup.
    ``codes_LN[..., N]`` -> ``[..., D]`` summed across the codebook axis.
    """

    def __init__(self, num_codebooks: int, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_codebooks * vocab_size, hidden_size))
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size

    def forward(self, codes_LN: torch.Tensor) -> torch.Tensor:
        offsets = (
            torch.arange(self.num_codebooks, device=codes_LN.device, dtype=codes_LN.dtype)
            * self.vocab_size
        )
        return F.embedding(codes_LN + offsets, self.weight).sum(dim=-2)


class HiggsFusedMultiTextHead(nn.Module):
    """Fused multi-codebook head: ``[L, D]`` -> ``[L, N, V]`` via one linear."""

    def __init__(self, num_codebooks: int, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_codebooks * vocab_size, hidden_size))
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size

    def forward(self, hidden_LD: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden_LD, self.weight)
        return logits.reshape(hidden_LD.shape[0], self.num_codebooks, self.vocab_size)


# --------------------------------------------------------------------------- #
# Per-request delay/EOC sampler state machine (reference oracle, pure torch)
# --------------------------------------------------------------------------- #
@dataclass
class _SamplerState:
    num_codebooks: int
    delay_count: int = 0
    eoc_countdown: int | None = None
    generation_done: bool = False
    last_codes: torch.Tensor | None = None


def _sample(logits_NV: torch.Tensor, temperature: float, top_p: float | None,
            top_k: int | None) -> torch.Tensor:
    if temperature <= 1e-5:
        return logits_NV.argmax(dim=-1)
    logits = logits_NV / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = logits.topk(k, dim=-1).values[:, -1:]
        logits = torch.where(logits < kth, float("-inf"), logits)
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        scatter = torch.zeros_like(remove)
        scatter.scatter_(-1, sorted_idx, remove)
        logits = torch.where(scatter, float("-inf"), logits)
    return logits.softmax(dim=-1).multinomial(num_samples=1).squeeze(-1)


def _sampler_step(logits_NV: torch.Tensor, state: _SamplerState, *,
                  temperature: float, top_p: float | None,
                  top_k: int | None) -> torch.Tensor:
    """One AR step of the multi-codebook delay sampler. Mutates ``state``."""
    N = state.num_codebooks
    codes_N = _sample(logits_NV, temperature, top_p, top_k).to(torch.long)

    if state.delay_count < N:
        next_cb = state.delay_count + 1
        if next_cb < N:
            codes_N[next_cb:] = BOC_ID
        state.delay_count += 1
    elif state.eoc_countdown is not None:
        state.eoc_countdown -= 1
        if state.eoc_countdown <= 0:
            state.generation_done = True
    elif int(codes_N[0].item()) == EOC_ID:
        if N <= 2:
            state.generation_done = True
        else:
            state.eoc_countdown = N - 2

    if not state.generation_done:
        state.last_codes = codes_N.clone()
    return codes_N


class HiggsMultimodalQwen3PreTrainedModel(PreTrainedModel):
    config_class = HiggsMultimodalQwen3Config
    base_model_prefix = "model"
    _supports_cache_class = True
    _supports_sdpa = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _supports_flash_attn = True
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        """Initialize weights for newly created modules.
        Relies on ``super()._init_weights`` for standard Linear / Embedding /
        LayerNorm handling; only adds custom init for the fused audio modules.
        """
        super()._init_weights(module)
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, (HiggsFusedMultiTextEmbedding, HiggsFusedMultiTextHead)):
            nn.init.normal_(module.weight, mean=0.0, std=std)


class HiggsMultimodalQwen3ForConditionalGeneration(HiggsMultimodalQwen3PreTrainedModel):
    """Higgs Audio v3 TTS model.
    Loadable via ``AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)``.
    Use :meth:`generate_speech` for end-to-end text -> waveform synthesis.
    """

    # Higgs ckpt names -> this module's parameter tree. transformers 5.x applies
    # this via the ``key_mapping`` weight-conversion path (see ``from_pretrained``).
    _HIGGS_KEY_MAPPING = {
        r"tied\.embedding\.text_embedding\.": "model.embed_tokens.",
        r"tied\.embedding\.modality_embeddings\.0\.embedding\.": "audio_embedding.",
        r"body\.": "model.",
    }
    # Codec weights (bundled in the ckpt) and the tied text head are not part of
    # the AR graph; the codec is loaded separately from ``audio_tokenizer_id``.
    _keys_to_ignore_on_load_unexpected = [
        r"tied\.embedding\.modality_embeddings\.0\.model\.",
        r"tied\.head\.",
    ]
    _tied_weights_keys = ["audio_head.weight"]

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        kwargs.setdefault("key_mapping", dict(cls._HIGGS_KEY_MAPPING))
        return super().from_pretrained(*args, **kwargs)

    def __init__(self, config: HiggsMultimodalQwen3Config):
        super().__init__(config)
        text_config = config.get_text_config()
        self.model = Qwen3Model(text_config)

        enc = config.audio_encoder_config or {}
        self.num_codebooks = int(enc["num_codebooks"])
        self.codebook_vocab_size = int(enc["vocab_size"])
        hidden = int(enc.get("out_dim", text_config.hidden_size))
        self._tie_audio_head = bool(enc.get("tie_word_embeddings", True))

        self.audio_embedding = HiggsFusedMultiTextEmbedding(
            self.num_codebooks, self.codebook_vocab_size, hidden
        )
        self.audio_head = HiggsFusedMultiTextHead(
            self.num_codebooks, self.codebook_vocab_size, hidden
        )
        self._audio_codec = None  # lazily loaded
        self.post_init()

    def tie_weights(self, *args, **kwargs):
        super().tie_weights(*args, **kwargs)
        if self._tie_audio_head:
            self.audio_head.weight = self.audio_embedding.weight

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # ----- training forward ----------------------------------------------- #
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        audio_codes: Optional[torch.LongTensor] = None,
        audio_mask: Optional[torch.BoolTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels_audio: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        channelwise_loss_weight: Optional[List[float]] = None,
        **kwargs,
    ) -> Union[Tuple, HiggsAudioOutputWithPast]:
        """Training-capable forward pass.

        Args:
            input_ids: ``[B, S]`` text token IDs. Audio placeholder positions
                (marked by ``audio_mask``) can contain any value and will be
                overridden by the fused audio embedding.
            audio_codes: ``[B, S, N]`` per-codebook codes (delay-patterned).
                Only positions where ``audio_mask`` is True are used.
            audio_mask: ``[B, S]`` boolean mask. True = audio position (use
                ``audio_codes`` for embedding), False = text position (use
                ``input_ids`` for embedding).
            attention_mask: ``[B, S]`` standard causal attention mask.
            labels_audio: ``[B, S, N]`` teacher-forcing targets for the audio
                head. Use ``-100`` to ignore a position/codebook.
            channelwise_loss_weight: Optional ``[N]`` per-codebook loss weights.
        """
        # 1. Compute input embeddings
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either `input_ids` or `inputs_embeds` must be provided.")
            # Text embedding for all positions
            safe_ids = input_ids.clone()
            if audio_mask is not None:
                safe_ids[audio_mask] = 0  # placeholder for audio positions
            inputs_embeds = self.model.embed_tokens(safe_ids)  # [B, S, D]

            # Overlay audio embedding at audio positions
            if audio_codes is not None and audio_mask is not None and audio_mask.any():
                audio_emb = self.audio_embedding(audio_codes)  # [B, S, D]
                inputs_embeds = inputs_embeds.clone()
                inputs_embeds[audio_mask] = audio_emb[audio_mask].to(inputs_embeds.dtype)

        # 2. Backbone forward
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state  # [B, S, D]

        # 3. Audio head: [B, S, D] -> [B, S, N, V]
        # Reshape for the fused head which expects [L, D]
        B, S, D = hidden_states.shape
        logits_flat = self.audio_head(hidden_states.reshape(-1, D))  # [B*S, N, V]
        logits = logits_flat.reshape(B, S, self.num_codebooks, self.codebook_vocab_size)

        # 4. Loss computation
        loss = None
        all_sum_losses = None
        all_token_nums = None
        channel_losses = None

        if labels_audio is not None:
            N = self.num_codebooks
            V = self.codebook_vocab_size
            loss_fct = CrossEntropyLoss(reduction="none")

            # labels_audio: [B, S, N]
            all_sum_losses_list = []
            all_token_nums_list = []

            for c in range(N):
                logits_c = logits[:, :, c, :]          # [B, S, V]
                labels_c = labels_audio[:, :, c]        # [B, S]

                per_token_loss = loss_fct(
                    logits_c.reshape(-1, V),
                    labels_c.reshape(-1),
                )  # [B*S]
                per_token_loss = per_token_loss.reshape(B, S)
                per_sample_loss = per_token_loss.sum(dim=-1)  # [B]
                valid_tokens = (labels_c != -100).sum(dim=-1)  # [B]

                all_sum_losses_list.append(per_sample_loss)
                all_token_nums_list.append(valid_tokens)

            all_sum_losses = torch.stack(all_sum_losses_list, dim=1)  # [B, N]
            all_token_nums = torch.stack(all_token_nums_list, dim=1)  # [B, N]

            # Aggregate
            if channelwise_loss_weight is not None:
                if len(channelwise_loss_weight) != N:
                    raise ValueError(
                        f"channelwise_loss_weight length {len(channelwise_loss_weight)} != {N}"
                    )
                w = torch.tensor(
                    channelwise_loss_weight,
                    device=all_sum_losses.device,
                    dtype=all_sum_losses.dtype,
                )
                total_loss_per_channel = all_sum_losses.sum(dim=0)  # [N]
                total_tokens_per_channel = all_token_nums.sum(dim=0).float().clamp(min=1.0)
                channel_losses = total_loss_per_channel / total_tokens_per_channel
                loss = (channel_losses * w).sum() / w.sum()
            else:
                total_tokens = all_token_nums.sum().float().clamp(min=1.0)
                loss = all_sum_losses.sum() / total_tokens
                total_tokens_per_channel = all_token_nums.sum(dim=0).float().clamp(min=1.0)
                channel_losses = all_sum_losses.sum(dim=0) / total_tokens_per_channel

        return HiggsAudioOutputWithPast(
            loss=loss,
            all_sum_losses=all_sum_losses,
            all_token_nums=all_token_nums,
            channel_losses=channel_losses,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # ----- audio codec (lazy) --------------------------------------------- #
    def get_audio_codec(self):
        """Load + cache the ``higgs_audio_v2_tokenizer`` codec (fp32, eval)."""
        if self._audio_codec is None:
            codec = AutoModel.from_pretrained(
                self.config.audio_tokenizer_id, dtype=torch.float32,
                trust_remote_code=True,
            )
            codec = codec.to(self.device).eval()
            for p in codec.parameters():
                p.requires_grad_(False)
            self._audio_codec = codec
        return self._audio_codec

    @torch.no_grad()
    def _encode_reference(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Reference waveform -> ``[T, N]`` int64 codes (on model device)."""
        import torchaudio

        codec = self.get_audio_codec()
        wav = waveform.float()
        while wav.ndim < 3:
            wav = wav.unsqueeze(0)
        if sample_rate != self.config.sample_rate:
            wav = torchaudio.functional.resample(wav, sample_rate, self.config.sample_rate)
        if wav.shape[-1] < self.config.sample_rate:
            wav = F.pad(wav, (0, self.config.sample_rate - wav.shape[-1]))
        wav = wav.to(self.device, dtype=torch.float32)
        codes_BNT = codec.encode(wav).audio_codes
        return codes_BNT.squeeze(0).transpose(0, 1).to(torch.long)

    @torch.no_grad()
    def _decode_codes(self, codes_TN: torch.Tensor) -> torch.Tensor:
        """``[T, N]`` raw codes -> mono waveform ``[L]`` (CPU float32)."""
        codec = self.get_audio_codec()
        codec_vocab = self.codebook_vocab_size - 2  # drop BOC/EOC
        codes_TN = torch.where(codes_TN >= codec_vocab, torch.zeros_like(codes_TN), codes_TN)
        codes_BNT = codes_TN.transpose(0, 1).unsqueeze(0).to(self.device, torch.long)
        audio = codec.decode(codes_BNT).audio_values
        return audio.squeeze(0).squeeze(0).detach().cpu().float()

    # ----- prompt assembly ------------------------------------------------ #
    @staticmethod
    def _special_ids(tokenizer) -> dict[str, int | None]:
        vocab = dict(tokenizer.get_added_vocab())
        missing = [t for t in _REQUIRED_SPECIALS if t not in vocab]
        if missing:
            raise ValueError(f"Tokenizer is missing Higgs TTS specials: {missing}")
        ids = {t: vocab[t] for t in _REQUIRED_SPECIALS}
        ids["<|ref_text|>"] = vocab.get("<|ref_text|>")
        return ids

    def _build_prompt_ids(self, tokenizer, text: str, *, num_ref_tokens: int,
                          reference_text: str | None) -> list[int]:
        sp = self._special_ids(tokenizer)
        ids: list[int] = [sp["<|tts|>"]]
        if reference_text and num_ref_tokens > 0 and sp["<|ref_text|>"] is not None:
            ids.append(sp["<|ref_text|>"])
            ids.extend(tokenizer.encode(reference_text, add_special_tokens=False))
        if num_ref_tokens > 0:
            ids.append(sp["<|ref_audio|>"])
            ids.extend([AUDIO_PLACEHOLDER_ID] * num_ref_tokens)
        ids.append(sp["<|text|>"])
        ids.extend(tokenizer.encode(text, add_special_tokens=False))
        ids.append(sp["<|audio|>"])
        return ids

    def _prefill_embeds(self, prompt_ids: list[int],
                        delayed_ref: torch.Tensor | None) -> torch.Tensor:
        """Embed the prompt; overlay fused audio embedding at ``-100`` slots."""
        ids = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
        mask = ids == AUDIO_PLACEHOLDER_ID
        safe = torch.where(mask, torch.zeros_like(ids), ids)
        embeds = self.model.embed_tokens(safe)
        if delayed_ref is not None and mask.any():
            n = int(mask.sum().item())
            audio = self.audio_embedding(delayed_ref[:n].to(self.device))
            embeds[mask] = audio.to(embeds.dtype)
        return embeds.unsqueeze(0)  # [1, S, D]

    # ----- generation ----------------------------------------------------- #
    @torch.no_grad()
    def generate_speech(
        self,
        text: str,
        tokenizer,
        *,
        reference_audio: torch.Tensor | None = None,
        reference_sample_rate: int | None = None,
        reference_codes: torch.Tensor | None = None,
        reference_text: str | None = None,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Synthesize ``text`` into a mono 24 kHz waveform (CPU float32 ``[L]``).
        Voice cloning: pass ``reference_audio`` (``[L]`` / ``[C, L]`` tensor with
        ``reference_sample_rate``) or pre-encoded ``reference_codes`` (``[T, N]``).
        ``reference_text`` (the transcript of the reference) improves cloning.
        """
        N = self.num_codebooks

        delayed_ref = None
        if reference_codes is not None:
            delayed_ref = apply_delay_pattern(reference_codes.to(torch.long))
        elif reference_audio is not None:
            sr = reference_sample_rate or self.config.sample_rate
            codes_TN = self._encode_reference(reference_audio, sr)
            delayed_ref = apply_delay_pattern(codes_TN.cpu())

        prompt_ids = self._build_prompt_ids(
            tokenizer, text,
            num_ref_tokens=0 if delayed_ref is None else delayed_ref.shape[0],
            reference_text=reference_text,
        )

        inputs_embeds = self._prefill_embeds(prompt_ids, delayed_ref)
        out = self.model(inputs_embeds=inputs_embeds, use_cache=True)
        past = out.past_key_values
        hidden_last = out.last_hidden_state[:, -1, :]
        position = inputs_embeds.shape[1]

        state = _SamplerState(num_codebooks=N)
        rows: list[torch.Tensor] = []
        for _ in range(max_new_tokens):
            logits_NV = self.audio_head(hidden_last).to(torch.float32)[0]  # [N, V]
            codes_N = _sampler_step(
                logits_NV, state,
                temperature=temperature, top_p=top_p, top_k=top_k,
            )
            if state.generation_done:
                break
            rows.append(codes_N.cpu())

            step_embed = self.audio_embedding(codes_N.unsqueeze(0)).unsqueeze(1)
            cache_pos = torch.tensor([position], device=self.device)
            out = self.model(
                inputs_embeds=step_embed.to(inputs_embeds.dtype),
                past_key_values=past,
                use_cache=True,
                cache_position=cache_pos,
            )
            past = out.past_key_values
            hidden_last = out.last_hidden_state[:, -1, :]
            position += 1

        if len(rows) < N:
            return torch.zeros(0, dtype=torch.float32)
        delayed_LN = torch.stack(rows, dim=0)
        codes_TN = reverse_delay_pattern(delayed_LN)
        return self._decode_codes(codes_TN)


__all__ = [
    "HiggsMultimodalQwen3ForConditionalGeneration",
    "HiggsMultimodalQwen3PreTrainedModel",
]

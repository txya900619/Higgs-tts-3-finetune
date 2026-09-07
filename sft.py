from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import shutil
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType, enable_fsdp_ram_efficient_loading, set_seed
from accelerate.utils.dataclasses import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, get_scheduler

from src.common import load_jsonl, resolve_jsonl_paths
from src.dataset import HiggsAudioSFTDataset
from src.processor import HiggsAudioProcessor


SCHEDULER_CHOICES = (
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
    "inverse_sqrt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised finetuning for Higgs Audio v3."
    )
    parser.add_argument("--model-path", type=str, required=True,
                        help="HuggingFace repo ID or local path to pretrained Higgs Audio v3 model.")
    parser.add_argument(
        "--train-jsonl", type=str, required=True,
        help="Single JSONL, directory, glob, or comma-separated list of JSONL files.",
    )
    parser.add_argument(
        "--eval-jsonl", type=str, default=None,
        help="Optional eval JSONL (same schema as --train-jsonl, already prepared "
             "with prepare_data.py). Enables validation loss and best-checkpoint "
             "selection; without it neither is computed.",
    )
    parser.add_argument(
        "--eval-steps", type=int, default=None,
        help="Also evaluate every N optimizer steps. Default: only at the end of "
             "each epoch.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size", type=int, default=None,
        help="Defaults to --per-device-batch-size. Evaluation has no backward "
             "pass, so this can usually be larger.",
    )
    parser.add_argument(
        "--no-save-best", dest="save_best", action="store_false",
        help="Do not keep a checkpoint-best/ copy of the lowest-eval-loss step.",
    )
    parser.set_defaults(save_best=True)
    parser.add_argument("--output-dir", type=str, default="output/higgs_sft")
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", type=str, default="linear", choices=SCHEDULER_CHOICES)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--attn-implementation", type=str, default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--channelwise-loss-weight", type=str, default=None,
        help="Comma-separated per-codebook loss weights. Default: uniform.",
    )
    parser.add_argument("--seed", type=int, default=42)

    # LoRA configuration
    parser.add_argument("--use-lora", action="store_true",
                        help="Enable LoRA training instead of full fine-tuning.")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank (r). Higher = more capacity, more memory.")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha (scaling factor). Typically 2*rank.")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="Dropout probability for LoRA layers.")
    parser.add_argument("--lora-target-modules", type=str, default=None,
                        help="Comma-separated list of module names to apply LoRA. "
                             "Default: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--lora-modules-to-save", type=str, default=None,
                        help="Comma-separated list of modules to save fully (not LoRA). "
                             "Default: audio_embedding,audio_head")

    return parser.parse_args()


def configure_torch_backends() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)


def resolve_torch_dtype(mixed_precision: str) -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def resolve_attn_implementation(requested: str, dtype: torch.dtype) -> str:
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "eager"
    if (
        importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    return "sdpa"


def format_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=max(0, int(seconds))))


def resolve_warmup_steps(args: argparse.Namespace, num_training_steps: int) -> int:
    if args.warmup_steps > 0:
        return args.warmup_steps
    if args.warmup_ratio > 0:
        return math.ceil(num_training_steps * args.warmup_ratio)
    return 0


def parse_channelwise_loss_weight(spec: Optional[str], n_codebooks: int) -> Optional[List[float]]:
    """Parse channelwise loss weight specification.

    Supports:
        - N values: one per codebook
        - None: uniform weighting (default)
    """
    if spec is None:
        return None
    values = [float(item.strip()) for item in spec.split(",") if item.strip()]
    if not values:
        return None
    if len(values) == n_codebooks:
        return values
    raise ValueError(
        f"`channelwise_loss_weight` expects {n_codebooks} values, got {len(values)}."
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.per_device_batch_size <= 0:
        raise ValueError("`per_device_batch_size` must be > 0.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("`gradient_accumulation_steps` must be > 0.")
    if args.learning_rate <= 0:
        raise ValueError("`learning_rate` must be > 0.")
    if args.weight_decay < 0:
        raise ValueError("`weight_decay` must be >= 0.")
    if args.num_epochs <= 0:
        raise ValueError("`num_epochs` must be > 0.")


def shard_paths_for_rank(paths: List[Path], world_size: int, rank: int) -> tuple[List[Path], bool]:
    if world_size <= 1:
        return paths, False

    shard_pattern = re.compile(r"\.rank(\d+)-of-(\d+)\.jsonl$")
    parsed: List[tuple[Path, int, int]] = []
    for path in paths:
        match = shard_pattern.search(path.name)
        if match is None:
            return paths, False
        shard_rank = int(match.group(1))
        shard_world_size = int(match.group(2))
        parsed.append((path, shard_rank, shard_world_size))

    shard_world_sizes = {item[2] for item in parsed}
    if len(shard_world_sizes) != 1:
        return paths, False

    selected = [path for path, shard_rank, _ in parsed if shard_rank % world_size == rank]
    if not selected:
        raise ValueError(
            f"No shard assigned for rank={rank} world_size={world_size}."
        )
    return selected, True


def load_jsonl_for_rank(
    spec: str, world_size: int, rank: int,
) -> tuple[List[Path], List[Dict[str, Any]], List[Path], bool]:
    all_paths = resolve_jsonl_paths(spec)
    rank_paths, using_pre_sharded = shard_paths_for_rank(
        all_paths, world_size=world_size, rank=rank,
    )
    records: List[Dict[str, Any]] = []
    for path in rank_paths:
        records.extend(load_jsonl(path))
    return all_paths, records, rank_paths, using_pre_sharded


@contextmanager
def processor_init_context(accelerator: Accelerator):
    if accelerator.distributed_type != DistributedType.DEEPSPEED:
        yield
        return
    plugin = accelerator.state.deepspeed_plugin
    if plugin is None or not plugin.is_zero3_init_enabled():
        yield
        return
    import deepspeed
    with plugin.zero3_init_context_manager(enable=False):
        deepspeed.zero.partition_parameters.shutdown_init_context()
        try:
            yield
        finally:
            deepspeed.zero.partition_parameters.restore_init_context()


def model_init_context(accelerator: Accelerator):
    if accelerator.distributed_type == DistributedType.FSDP:
        enable_fsdp_ram_efficient_loading()
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        plugin = accelerator.state.deepspeed_plugin
        if plugin is not None and plugin.is_zero3_init_enabled():
            return plugin.zero3_init_context_manager(enable=True)
    return nullcontext()


def save_checkpoint(
    accelerator: Accelerator,
    model,
    output_dir: Path,
    train_args: Dict[str, Any],
    is_lora: bool = False,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    if is_lora:
        # Save only LoRA adapter weights + modules_to_save
        unwrapped = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            unwrapped.save_pretrained(
                output_dir, safe_serialization=True,
            )
            with open(output_dir / "finetune_args.json", "w", encoding="utf-8") as f:
                json.dump(train_args, f, indent=2, ensure_ascii=False)
    else:
        state_dict = accelerator.get_state_dict(model)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True,
        )
        if accelerator.is_main_process:
            with open(output_dir / "finetune_args.json", "w", encoding="utf-8") as f:
                json.dump(train_args, f, indent=2, ensure_ascii=False)
    accelerator.wait_for_everyone()


def main() -> None:
    args = parse_args()
    validate_args(args)
    configure_torch_backends()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )
    set_seed(args.seed, device_specific=True)

    global_micro_batch_size = args.per_device_batch_size * accelerator.num_processes
    global_batch_size = global_micro_batch_size * args.gradient_accumulation_steps
    accelerator.print(
        f"[{format_timestamp()}] [sft] global_batch_size="
        f"{args.per_device_batch_size} x {accelerator.num_processes} x "
        f"{args.gradient_accumulation_steps} = {global_batch_size}"
    )

    # Load data
    train_paths, records, local_paths, using_pre_sharded = load_jsonl_for_rank(
        args.train_jsonl,
        world_size=accelerator.num_processes,
        rank=accelerator.process_index,
    )
    if not records:
        raise ValueError(f"No records found in {args.train_jsonl}.")
    accelerator.print(
        f"[{format_timestamp()}] [sft] dist_type={accelerator.distributed_type} "
        f"num_processes={accelerator.num_processes} "
        f"using_pre_sharded={using_pre_sharded} "
        f"total_files={len(train_paths)} local_files={len(local_paths)} "
        f"local_records={len(records)}"
    )

    # Build processor
    with processor_init_context(accelerator):
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True,
        )
        config = AutoConfig.from_pretrained(
            args.model_path, trust_remote_code=True,
        )
        enc = config.audio_encoder_config or {}
        num_codebooks = int(enc["num_codebooks"])
        codebook_vocab_size = int(enc["vocab_size"])

        processor = HiggsAudioProcessor(
            tokenizer=tokenizer,
            num_codebooks=num_codebooks,
            codebook_vocab_size=codebook_vocab_size,
        )

    # Build dataset
    dataset = HiggsAudioSFTDataset(
        records=records,
        processor=processor,
        num_codebooks=num_codebooks,
    )

    # Build eval dataset (optional). Sharding mirrors train: a pre-sharded eval
    # set is read per-rank and left unprepared, anything else is prepared so
    # Accelerate splits it.
    eval_dataset = None
    eval_pre_sharded = False
    if args.eval_jsonl:
        eval_paths, eval_records, eval_local_paths, eval_pre_sharded = load_jsonl_for_rank(
            args.eval_jsonl,
            world_size=accelerator.num_processes,
            rank=accelerator.process_index,
        )
        if not eval_records:
            raise ValueError(f"No records found in {args.eval_jsonl}.")
        accelerator.print(
            f"[{format_timestamp()}] [sft] eval: using_pre_sharded={eval_pre_sharded} "
            f"total_files={len(eval_paths)} local_files={len(eval_local_paths)} "
            f"local_records={len(eval_records)}"
        )
        eval_dataset = HiggsAudioSFTDataset(
            records=eval_records,
            processor=processor,
            num_codebooks=num_codebooks,
        )

    # Load model
    model_dtype = resolve_torch_dtype(args.mixed_precision)
    attn_impl = resolve_attn_implementation(args.attn_implementation, model_dtype)

    # Import the model class
    from model.modeling import HiggsMultimodalQwen3ForConditionalGeneration

    with model_init_context(accelerator):
        model = HiggsMultimodalQwen3ForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=model_dtype,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )

    # Parse channelwise loss weight
    resolved_channelwise = parse_channelwise_loss_weight(
        args.channelwise_loss_weight, num_codebooks,
    )
    accelerator.print(
        f"[{format_timestamp()}] [sft] num_codebooks={num_codebooks} "
        f"codebook_vocab_size={codebook_vocab_size} "
        f"channelwise_loss_weight={resolved_channelwise}"
    )

    # LoRA wrapping (must happen before gradient checkpointing & optimizer)
    is_lora = getattr(args, "use_lora", False)
    if is_lora:
        from peft import LoraConfig, get_peft_model

        lora_target_modules = (
            [m.strip() for m in args.lora_target_modules.split(",")]
            if args.lora_target_modules
            else ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
        )
        lora_modules_to_save = (
            [m.strip() for m in args.lora_modules_to_save.split(",")]
            if args.lora_modules_to_save
            else ["audio_embedding", "audio_head"]
        )

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=lora_target_modules,
            modules_to_save=lora_modules_to_save,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        accelerator.print(
            f"[{format_timestamp()}] [sft] LoRA enabled: rank={args.lora_rank} "
            f"alpha={args.lora_alpha} dropout={args.lora_dropout} "
            f"target_modules={lora_target_modules} "
            f"modules_to_save={lora_modules_to_save}"
        )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        drop_last=using_pre_sharded,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
    )

    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size or args.per_device_batch_size,
            shuffle=False,
            drop_last=False,  # every eval sample must count
            num_workers=args.num_workers,
            collate_fn=eval_dataset.collate_fn,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    accelerator.print(
        f"[{format_timestamp()}] [sft] trainable_params={sum(p.numel() for p in trainable_params):,}"
    )
    optimizer = AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )

    if using_pre_sharded:
        micro_batches_per_epoch = math.ceil(len(records) / args.per_device_batch_size)
    else:
        micro_batches_per_epoch = math.ceil(len(records) / global_micro_batch_size)
    update_steps_per_epoch = math.ceil(micro_batches_per_epoch / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps or (args.num_epochs * update_steps_per_epoch)
    warmup_steps = resolve_warmup_steps(args, max_train_steps)

    accelerator.print(
        f"[{format_timestamp()}] [sft] scheduler={args.lr_scheduler_type} "
        f"warmup_steps={warmup_steps} "
        f"micro_batches_per_epoch={micro_batches_per_epoch} "
        f"optimizer_steps_per_epoch={update_steps_per_epoch} "
        f"max_train_steps={max_train_steps}"
    )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    if using_pre_sharded:
        model, optimizer, lr_scheduler = accelerator.prepare(
            model, optimizer, lr_scheduler,
        )
    else:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler,
        )
    if eval_dataloader is not None and not eval_pre_sharded:
        eval_dataloader = accelerator.prepare(eval_dataloader)

    output_root = Path(args.output_dir)
    if accelerator.is_main_process:
        output_root.mkdir(parents=True, exist_ok=True)

    train_args_to_save = vars(args).copy()
    train_args_to_save["global_batch_size"] = global_batch_size
    train_args_to_save["resolved_channelwise_loss_weight"] = resolved_channelwise

    # W&B
    wandb_module = None
    if args.wandb_project and accelerator.is_main_process:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("wandb not installed. pip install wandb") from exc
        init_kwargs: Dict[str, Any] = {
            "project": args.wandb_project,
            "config": train_args_to_save,
        }
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        if args.wandb_run_name:
            init_kwargs["name"] = args.wandb_run_name
        if args.wandb_tags:
            init_kwargs["tags"] = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wandb.init(**init_kwargs)
        wandb_module = wandb

    try:
        _training_loop(
            accelerator, args, model, train_dataloader, optimizer,
            lr_scheduler, resolved_channelwise, max_train_steps,
            global_batch_size, output_root, train_args_to_save, wandb_module,
            eval_dataloader,
        )
    finally:
        if wandb_module is not None:
            wandb_module.finish()


@torch.no_grad()
def evaluate(
    accelerator: Accelerator,
    model,
    eval_dataloader: DataLoader,
    resolved_channelwise: Optional[List[float]],
) -> tuple[Optional[float], Optional[List[float]]]:
    """Corpus-level eval loss, aggregated exactly the way the model defines it.

    The training loss is normalised by *token* count, not by sample count
    (`all_sum_losses.sum() / total_tokens` in modeling.py), so averaging the
    per-batch losses would silently weight short utterances more heavily. This
    instead accumulates the per-codebook loss sums and token counts across the
    whole eval set, then applies the model's own aggregation -- including the
    channelwise-weighted branch -- so the number is directly comparable to the
    training loss.

    `gather_for_metrics` is used rather than a plain gather because Accelerate
    pads the final batch of a prepared dataloader by repeating samples to keep
    ranks in step; gathering per-sample rows lets it drop those duplicates.
    """
    was_training = model.training
    model.eval()

    loss_sums: Optional[torch.Tensor] = None   # [N] summed loss per codebook
    token_nums: Optional[torch.Tensor] = None  # [N] valid tokens per codebook

    for batch in eval_dataloader:
        outputs = model(
            input_ids=batch["input_ids"],
            audio_codes=batch["audio_codes"],
            audio_mask=batch["audio_mask"],
            attention_mask=batch["attention_mask"],
            labels_audio=batch["labels_audio"],
            channelwise_loss_weight=resolved_channelwise,
        )
        sums = accelerator.gather_for_metrics(outputs.all_sum_losses.detach().float())
        nums = accelerator.gather_for_metrics(outputs.all_token_nums.detach().float())
        sums = sums.sum(dim=0)
        nums = nums.sum(dim=0)
        loss_sums = sums if loss_sums is None else loss_sums + sums
        token_nums = nums if token_nums is None else token_nums + nums

    if was_training:
        model.train()
    if loss_sums is None:
        return None, None

    channel_losses = loss_sums / token_nums.clamp(min=1.0)
    if resolved_channelwise is not None:
        weights = torch.tensor(
            resolved_channelwise, device=channel_losses.device, dtype=channel_losses.dtype,
        )
        loss = (channel_losses * weights).sum() / weights.sum()
    else:
        loss = loss_sums.sum() / token_nums.sum().clamp(min=1.0)
    return loss.item(), channel_losses.tolist()


def _training_loop(
    accelerator: Accelerator,
    args: argparse.Namespace,
    model,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    resolved_channelwise: Optional[List[float]],
    max_train_steps: int,
    global_batch_size: int,
    output_root: Path,
    train_args_to_save: Dict[str, Any],
    wandb_module: Optional[Any],
    eval_dataloader: Optional[DataLoader] = None,
) -> None:
    global_step = 0
    completed_epochs = 0
    last_log_time = time.perf_counter()
    last_logged_step = 0
    best_eval_loss: Optional[float] = None
    last_eval_step = -1

    def run_eval(step: int, epoch: int) -> None:
        """Evaluate, log, and keep the best checkpoint. No-op without --eval-jsonl.

        Idempotent per step: an epoch boundary that lands exactly on an
        --eval-steps multiple would otherwise evaluate the same weights twice.
        """
        nonlocal best_eval_loss, last_log_time, last_eval_step
        if eval_dataloader is None or step == last_eval_step:
            return
        last_eval_step = step
        eval_start = time.perf_counter()
        eval_loss, channel_losses = evaluate(
            accelerator, model, eval_dataloader, resolved_channelwise,
        )
        if eval_loss is None:
            return
        improved = best_eval_loss is None or eval_loss < best_eval_loss
        if improved:
            best_eval_loss = eval_loss
        accelerator.print(
            f"[{format_timestamp()}] [eval] epoch={epoch} step={step} "
            f"eval_loss={eval_loss:.4f} best={best_eval_loss:.4f}"
            f"{' *' if improved else ''} "
            f"took={format_duration(time.perf_counter() - eval_start)}"
        )
        if wandb_module is not None:
            payload = {"eval/loss": eval_loss, "eval/best_loss": best_eval_loss}
            for i, cl in enumerate(channel_losses or []):
                payload[f"eval/channel_{i}_loss"] = cl
            wandb_module.log(payload, step=step)

        if improved and args.save_best:
            save_checkpoint(
                accelerator=accelerator,
                model=model,
                output_dir=output_root / "checkpoint-best",
                train_args={**train_args_to_save, "best_eval_loss": eval_loss,
                            "best_eval_step": step, "best_eval_epoch": epoch},
                is_lora=getattr(args, "use_lora", False),
            )
        # Evaluation and checkpointing are not training time; don't let them
        # contaminate the next step-rate/ETA measurement.
        last_log_time = time.perf_counter()

    for epoch in range(args.num_epochs):
        model.train()
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    audio_codes=batch["audio_codes"],
                    audio_mask=batch["audio_mask"],
                    attention_mask=batch["attention_mask"],
                    labels_audio=batch["labels_audio"],
                    channelwise_loss_weight=resolved_channelwise,
                )
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.logging_steps == 0:
                    now = time.perf_counter()
                    steps_since = max(global_step - last_logged_step, 1)
                    elapsed = max(now - last_log_time, 1e-12)
                    last_log_time = now
                    last_logged_step = global_step
                    step_time = elapsed / steps_since
                    steps_per_sec = steps_since / elapsed
                    samples_per_sec = (global_batch_size * steps_since) / elapsed
                    eta_seconds = max(max_train_steps - global_step, 0) / steps_per_sec
                    logged_loss = accelerator.gather(
                        loss.detach().float().reshape(1)
                    ).mean().item()
                    lr_val = lr_scheduler.get_last_lr()[0]
                    accelerator.print(
                        f"[{format_timestamp()}] "
                        f"epoch={epoch} step={global_step}/{max_train_steps} "
                        f"loss={logged_loss:.4f} "
                        f"lr={lr_val:.2e} "
                        f"step_time={step_time:.2f}s "
                        f"steps/s={steps_per_sec:.3f} "
                        f"samples/s={samples_per_sec:.2f} "
                        f"eta={format_duration(eta_seconds)}"
                    )
                    if wandb_module is not None:
                        wandb_module.log(
                            {
                                "train/loss": logged_loss,
                                "train/lr": lr_val,
                                "train/step_time": step_time,
                                "train/steps_per_sec": steps_per_sec,
                                "train/samples_per_sec": samples_per_sec,
                                "train/epoch": epoch,
                            },
                            step=global_step,
                        )

                if args.eval_steps and global_step % args.eval_steps == 0:
                    run_eval(global_step, epoch)

                if global_step >= max_train_steps:
                    break

        run_eval(global_step, epoch)

        checkpoint_dir = output_root / f"checkpoint-epoch-{epoch}"
        save_checkpoint(
            accelerator=accelerator,
            model=model,
            output_dir=checkpoint_dir,
            train_args=train_args_to_save,
            is_lora=getattr(args, "use_lora", False),
        )
        completed_epochs = epoch + 1

        if global_step >= max_train_steps:
            break

    best = f", best_eval_loss={best_eval_loss:.4f}" if best_eval_loss is not None else ""
    accelerator.print(
        f"[{format_timestamp()}] Finished training: "
        f"global_step={global_step}, saved_epochs={completed_epochs}{best}, "
        f"output_dir={output_root}"
    )


if __name__ == "__main__":
    main()

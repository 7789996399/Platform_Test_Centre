"""
LoRA fine-tuning entry point for the Layer 2 meta-classifier.

GPU mode: loads Llama 3 8B with 4-bit quantization, applies LoRA on q/v/k/o_proj.
CPU test mode: loads sshleifer/tiny-gpt2 (~500KB), LoRA on c_attn, 2 steps, no
quantization, no fp16.

Saves adapter weights only (not the full base model).

CLI usage:
    python -m layer2.training.train_lora [--test-mode] [--config path/to/config.yaml]
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def train(
    config_path: Optional[str] = None,
    test_mode: bool = False,
    output_dir: Optional[str] = None,
) -> str:
    """
    Run LoRA fine-tuning and return the path to saved adapter weights.

    Parameters
    ----------
    config_path : str, optional
        Path to training_config.yaml. Uses default location if None.
    test_mode : bool
        If True, uses tiny-gpt2 on CPU with mock data and minimal steps.
    output_dir : str, optional
        Directory to save adapter weights. Defaults to ``layer2_output/``.

    Returns
    -------
    str
        Path to the directory containing the saved LoRA adapter.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, TaskType

    from layer2.config import load_config
    from layer2.data.prepare_training_data import prepare_mock_dataset, prepare_dataset

    config = load_config(config_path)

    if output_dir is None:
        output_dir = "layer2_output"

    adapter_dir = os.path.join(output_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)

    # ── Model selection ──────────────────────────────────────────────────
    if test_mode:
        model_name = config.model.test_name
        target_modules = config.lora.test_target_modules
        logger.info("Test mode: using %s on CPU", model_name)
    else:
        model_name = config.model.name
        target_modules = config.lora.target_modules
        logger.info("GPU mode: using %s", model_name)

    # ── Tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Base model ───────────────────────────────────────────────────────
    model_kwargs = {}
    if not test_mode and config.quantization.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=config.quantization.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=torch.float16,
            )
            logger.info("4-bit quantization enabled")
        except ImportError:
            logger.warning("bitsandbytes not available, loading without quantization")

    if not test_mode:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # ── LoRA ─────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ──────────────────────────────────────────────────────────
    if test_mode:
        dataset = prepare_mock_dataset(
            num_examples=config.data.mock_num_examples,
            seed=42,
        )
    else:
        dataset = prepare_dataset(config.data.train_file)

    max_length = config.model.max_length

    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # ── Training arguments ───────────────────────────────────────────────
    if test_mode:
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=config.training.test_batch_size,
            max_steps=config.training.test_max_steps,
            logging_steps=1,
            save_strategy="no",
            fp16=False,
            no_cuda=True,
            report_to="none",
            remove_unused_columns=False,
        )
    else:
        report_to = "wandb" if config.wandb.enabled else "none"
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=config.training.batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            num_train_epochs=config.training.num_epochs,
            learning_rate=config.training.learning_rate,
            lr_scheduler_type=config.training.lr_scheduler_type,
            warmup_ratio=config.training.warmup_ratio,
            weight_decay=config.training.weight_decay,
            max_grad_norm=config.training.max_grad_norm,
            fp16=config.training.fp16,
            logging_steps=config.training.logging_steps,
            save_steps=config.training.save_steps,
            eval_strategy="steps" if config.training.eval_steps else "no",
            eval_steps=config.training.eval_steps,
            report_to=report_to,
            remove_unused_columns=False,
        )

    # ── W&B setup ────────────────────────────────────────────────────────
    if config.wandb.enabled and not test_mode:
        try:
            import wandb
            wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                name=config.wandb.run_name,
            )
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")

    # ── Train ────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete")

    # ── Save adapter only ────────────────────────────────────────────────
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("Saved adapter to %s", adapter_dir)

    return adapter_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Layer 2 meta-classifier with LoRA"
    )
    parser.add_argument("--test-mode", action="store_true",
                        help="Use tiny model on CPU with mock data")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to training_config.yaml")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for adapter weights")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    adapter_path = train(
        config_path=args.config,
        test_mode=args.test_mode,
        output_dir=args.output_dir,
    )
    print(f"Adapter saved to: {adapter_path}")


if __name__ == "__main__":
    main()

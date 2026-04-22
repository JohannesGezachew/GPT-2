"""Fine-tune GPT-2 on CNN/DailyMail with masked-LM loss over the summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from transformers import Trainer, TrainingArguments, set_seed

from data_engine.dataset import (
    TokenizerConfig,
    build_datasets,
    load_tokenizer,
)
from models.build import load_pretrained_model


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class PrecomputedLabelsCollator:
    """Pads-free collator: sequences are already padded to `max_length`.

    We still stack into tensors here. `labels` already carries -100 on both
    the article region and pad positions.
    """

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch = {}
        for key in ("input_ids", "attention_mask", "labels"):
            batch[key] = torch.tensor(
                [f[key] for f in features],
                dtype=torch.long,
            )
        return batch


def _build_training_args(cfg: dict) -> TrainingArguments:
    t = cfg["train"]
    return TrainingArguments(
        output_dir=t["output_dir"],
        overwrite_output_dir=True,
        seed=t["seed"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        weight_decay=t["weight_decay"],
        adam_beta1=t["adam_beta1"],
        adam_beta2=t["adam_beta2"],
        adam_epsilon=t["adam_epsilon"],
        max_grad_norm=t["max_grad_norm"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_steps=t["warmup_steps"],
        fp16=t["fp16"] and torch.cuda.is_available(),
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_steps=t["eval_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=t["save_total_limit"],
        report_to=[],
        load_best_model_at_end=False,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    set_seed(cfg["train"]["seed"])

    tokenizer = load_tokenizer(cfg["model"]["name_or_path"])
    tok_cfg = TokenizerConfig(
        max_length=cfg["model"]["max_length"],
        max_article_tokens=cfg["data"]["max_article_tokens"],
        max_summary_tokens=cfg["data"]["max_summary_tokens"],
    )

    datasets = build_datasets(
        dataset_name=cfg["data"]["dataset_name"],
        dataset_config=cfg["data"]["dataset_config"],
        tokenizer=tokenizer,
        tok_cfg=tok_cfg,
        min_summary_ratio=cfg["data"]["min_summary_ratio"],
        num_proc=cfg["data"]["num_proc"],
        cache_dir=cfg["data"]["cache_dir"],
    )

    eval_max = cfg["train"].get("eval_max_samples")
    eval_dataset = datasets["validation"]
    if eval_max and eval_max < len(eval_dataset):
        eval_dataset = eval_dataset.select(range(eval_max))

    model = load_pretrained_model(cfg["model"]["name_or_path"], tokenizer)

    training_args = _build_training_args(cfg)
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=PrecomputedLabelsCollator(),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()

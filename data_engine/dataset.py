"""Dataset loading + tokenization for GPT-2 summarization fine-tuning.

Produces sequences of the form

    {article}\\n\\nTL;DR:\\n{summary}<|endoftext|>

truncated so that article + prompt + summary fit in `max_length`. The
`labels` tensor masks everything up to and including the TL;DR separator
with -100 so cross-entropy only scores the summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import DatasetDict, load_dataset
from transformers import GPT2TokenizerFast

from .preprocess import (
    PROMPT_SEPARATOR,
    build_prompt,
    clean_article,
    clean_summary,
    should_keep,
)

IGNORE_INDEX = -100


def load_tokenizer(name_or_path: str) -> GPT2TokenizerFast:
    tokenizer = GPT2TokenizerFast.from_pretrained(name_or_path)
    # GPT-2 has no pad token; reuse EOS for padding. Attention mask prevents
    # the model from attending to pad positions.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_raw_dataset(dataset_name: str, config: str, cache_dir: str | None) -> DatasetDict:
    return load_dataset(dataset_name, config, cache_dir=cache_dir)


def _clean_example(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "article": clean_article(example["article"]),
        "summary": clean_summary(example["highlights"]),
    }


def _filter_example(example: dict[str, Any], min_summary_ratio: float) -> bool:
    return should_keep(example["article"], example["summary"], min_summary_ratio)


@dataclass
class TokenizerConfig:
    max_length: int = 1024
    max_article_tokens: int = 800
    max_summary_tokens: int = 160


def _tokenize_batch(
    batch: dict[str, list[str]],
    tokenizer: GPT2TokenizerFast,
    cfg: TokenizerConfig,
    separator_ids: list[int],
) -> dict[str, list[list[int]]]:
    eos_id = tokenizer.eos_token_id

    input_ids_out: list[list[int]] = []
    attention_mask_out: list[list[int]] = []
    labels_out: list[list[int]] = []

    for article, summary in zip(batch["article"], batch["summary"]):
        art_ids = tokenizer(
            article,
            add_special_tokens=False,
            truncation=True,
            max_length=cfg.max_article_tokens,
        )["input_ids"]
        sum_ids = tokenizer(
            summary,
            add_special_tokens=False,
            truncation=True,
            max_length=cfg.max_summary_tokens,
        )["input_ids"]

        prompt_ids = art_ids + separator_ids
        target_ids = sum_ids + [eos_id]

        # Trim article further if the total overflows max_length.
        overflow = len(prompt_ids) + len(target_ids) - cfg.max_length
        if overflow > 0:
            prompt_ids = separator_ids if overflow >= len(art_ids) else art_ids[:-overflow] + separator_ids

        input_ids = prompt_ids + target_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids
        attention_mask = [1] * len(input_ids)

        # Pad up to max_length for static-shape batching.
        pad_len = cfg.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [eos_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [IGNORE_INDEX] * pad_len

        input_ids_out.append(input_ids)
        attention_mask_out.append(attention_mask)
        labels_out.append(labels)

    return {
        "input_ids": input_ids_out,
        "attention_mask": attention_mask_out,
        "labels": labels_out,
    }


def build_datasets(
    dataset_name: str,
    dataset_config: str,
    tokenizer: GPT2TokenizerFast,
    tok_cfg: TokenizerConfig,
    min_summary_ratio: float,
    num_proc: int = 4,
    cache_dir: str | None = None,
) -> DatasetDict:
    raw = load_raw_dataset(dataset_name, dataset_config, cache_dir)

    cleaned = raw.map(
        _clean_example,
        num_proc=num_proc,
        remove_columns=raw["train"].column_names,
        desc="cleaning",
    )
    cleaned = cleaned.filter(
        _filter_example,
        fn_kwargs={"min_summary_ratio": min_summary_ratio},
        num_proc=num_proc,
        desc="filtering",
    )

    separator_ids = tokenizer(PROMPT_SEPARATOR, add_special_tokens=False)["input_ids"]

    tokenized = cleaned.map(
        _tokenize_batch,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=["article", "summary"],
        fn_kwargs={
            "tokenizer": tokenizer,
            "cfg": tok_cfg,
            "separator_ids": separator_ids,
        },
        desc="tokenizing",
    )

    return tokenized


def format_for_inference(tokenizer: GPT2TokenizerFast, article: str, max_article_tokens: int) -> dict[str, Any]:
    """Build a prompt tensor for generation."""
    article = clean_article(article)
    art_ids = tokenizer(
        article,
        add_special_tokens=False,
        truncation=True,
        max_length=max_article_tokens,
    )["input_ids"]
    sep_ids = tokenizer(PROMPT_SEPARATOR, add_special_tokens=False)["input_ids"]
    input_ids = art_ids + sep_ids
    return {
        "input_ids": input_ids,
        "prompt_text": build_prompt(article),
    }

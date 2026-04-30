"""Generate a summary from a single article using a fine-tuned GPT-2."""

from __future__ import annotations

import argparse
import sys

import torch
import yaml
from transformers import GPT2LMHeadModel

from data_engine.dataset import format_for_inference, load_tokenizer
from data_engine.preprocess import PROMPT_SEPARATOR


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def summarize(
    model: GPT2LMHeadModel,
    tokenizer,
    article: str,
    max_article_tokens: int,
    gen_cfg: dict,
    device: torch.device,
) -> str:
    prepared = format_for_inference(tokenizer, article, max_article_tokens)
    input_ids = torch.tensor([prepared["input_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=gen_cfg["max_new_tokens"],
        do_sample=gen_cfg["do_sample"],
        top_p=gen_cfg["top_p"],
        top_k=gen_cfg["top_k"],
        temperature=gen_cfg["temperature"],
        repetition_penalty=gen_cfg["repetition_penalty"],
        no_repeat_ngram_size=gen_cfg["no_repeat_ngram_size"],
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated = output[0, input_ids.shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    # Trim anything after a second TL;DR: if the model emits one.
    marker = PROMPT_SEPARATOR.strip()
    if marker in text:
        text = text.split(marker)[0].strip()
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--config", required=True, help="YAML config (uses generate + data sections)")
    parser.add_argument("--article", help="Article text. If omitted, reads from stdin.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    tokenizer = load_tokenizer(args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(args.ckpt).to(device)
    model.eval()

    article = args.article if args.article is not None else sys.stdin.read()
    if not article.strip():
        raise SystemExit("No article provided (use --article or pipe text on stdin).")

    summary = summarize(
        model,
        tokenizer,
        article,
        cfg["data"]["max_article_tokens"],
        cfg["generate"],
        device,
    )
    print(summary)


if __name__ == "__main__":
    main()

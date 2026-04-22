"""Run ROUGE-1/2/L evaluation on the CNN/DailyMail test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate
import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2LMHeadModel

from data_engine.dataset import load_tokenizer
from data_engine.preprocess import clean_article, clean_summary
from eval.generate import summarize


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output", default=None, help="Optional JSONL of (reference, prediction) pairs")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    tokenizer = load_tokenizer(args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(args.ckpt).to(device)
    model.eval()

    dataset = load_dataset(
        cfg["data"]["dataset_name"],
        cfg["data"]["dataset_config"],
        cache_dir=cfg["data"]["cache_dir"],
        split=args.split,
    )
    if args.num_samples and args.num_samples < len(dataset):
        dataset = dataset.select(range(args.num_samples))

    rouge = evaluate.load("rouge")

    predictions: list[str] = []
    references: list[str] = []
    records = []

    for example in tqdm(dataset, desc="generating"):
        article = clean_article(example["article"])
        reference = clean_summary(example["highlights"])
        with torch.no_grad():
            prediction = summarize(
                model,
                tokenizer,
                article,
                cfg["data"]["max_article_tokens"],
                cfg["generate"],
                device,
            )
        predictions.append(prediction)
        references.append(reference)
        records.append({"reference": reference, "prediction": prediction})

    scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    print(json.dumps(scores, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()

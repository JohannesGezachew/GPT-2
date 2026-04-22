"""Model loader for pretrained GPT-2 summarizer."""

from __future__ import annotations

from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def load_pretrained_model(name_or_path: str, tokenizer: GPT2TokenizerFast) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained(name_or_path)
    # Ensure the LM head matches the tokenizer (no-op for stock gpt2, but
    # safe if anyone later adds a custom special token).
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    return model

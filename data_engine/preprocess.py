"""Cleaning + prompt formatting for CNN/DailyMail."""

from __future__ import annotations

import re
import unicodedata

PROMPT_SEPARATOR = "\n\nTL;DR:\n"

_CNN_PREFIX = re.compile(r"^\s*\(CNN\)\s*-+\s*", re.IGNORECASE)
_BYLINE = re.compile(
    r"^\s*By\s+[^\n]{0,80}?(Reporter|Editor|Correspondent|Daily Mail|Associated Press)[^\n]*\n",
    re.IGNORECASE,
)
_PUBLISHED = re.compile(
    r"PUBLISHED:[^\n]*\n(?:UPDATED:[^\n]*\n)?", re.IGNORECASE
)
_UPDATED = re.compile(r"UPDATED:[^\n]*\n", re.IGNORECASE)
_HIGHLIGHT_NEW = re.compile(r"^\s*NEW:\s*", re.MULTILINE)
_WS = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")


def _normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "…": "...", "\xa0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def clean_article(text: str) -> str:
    text = _normalize_unicode(text)
    text = _CNN_PREFIX.sub("", text)
    text = _BYLINE.sub("", text)
    text = _PUBLISHED.sub("", text)
    text = _UPDATED.sub("", text)
    text = _WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def clean_summary(text: str) -> str:
    text = _normalize_unicode(text)
    text = _HIGHLIGHT_NEW.sub("", text)
    text = _WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def should_keep(article: str, summary: str, min_summary_ratio: float) -> bool:
    if not article or not summary:
        return False
    if len(summary) < min_summary_ratio * len(article):
        return False
    return True


def build_prompt(article: str) -> str:
    return f"{article}{PROMPT_SEPARATOR}"


def build_training_text(article: str, summary: str, eos: str) -> str:
    return f"{article}{PROMPT_SEPARATOR}{summary}{eos}"

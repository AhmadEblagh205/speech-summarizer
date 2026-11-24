"""Abstractive summarisation of long transcripts.

The production path uses a Hugging Face summarisation pipeline, but the
long-document orchestration is deliberately dependency-light and can be tested
without downloading a model.  The central rule is simple: input chunks are
bounded in *model tokens*, not guessed from a word count.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Iterable, Protocol


class TokenizerLike(Protocol):
    model_max_length: int

    def encode(self, text: str, add_special_tokens: bool = False): ...

    def num_special_tokens_to_add(self, pair: bool = False) -> int: ...


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence segmentation used only to preserve boundaries.

    This is intentionally not presented as a linguistic sentence tokenizer; it
    is a deterministic boundary heuristic for chunk construction.
    """
    clean = " ".join(text.split())
    if not clean:
        return []
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(clean) if s.strip()]


def _effective_model_limit(tokenizer: TokenizerLike, fallback: int = 1024) -> int:
    raw = int(getattr(tokenizer, "model_max_length", fallback) or fallback)
    # Hugging Face uses very large sentinels when a tokenizer has no declared
    # finite limit.  In that case a conservative explicit fallback is safer.
    if raw <= 0 or raw >= 100_000:
        return fallback
    return raw


def _token_count(tokenizer: TokenizerLike, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _word_chunks_for_oversized_unit(
    text: str,
    tokenizer: TokenizerLike,
    budget: int,
) -> list[str]:
    """Greedily split one oversized sentence while preserving word order."""
    words = text.split()
    if not words:
        return []
    out: list[str] = []
    cur: list[str] = []
    for word in words:
        trial = " ".join(cur + [word])
        if cur and _token_count(tokenizer, trial) > budget:
            out.append(" ".join(cur))
            cur = [word]
        else:
            cur.append(word)
        # One pathological word may itself exceed the budget.  I cannot split
        # subword tokens without decoding support, so fail loudly instead of
        # silently truncating the model input.
        if len(cur) == 1 and _token_count(tokenizer, cur[0]) > budget:
            raise ValueError(
                "a single whitespace-delimited token exceeds the model input budget"
            )
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_for_token_budget(
    text: str,
    tokenizer: TokenizerLike,
    *,
    max_input_tokens: int | None = None,
    safety_margin: int = 8,
) -> list[str]:
    """Split text into sentence-aware chunks that fit a model token window.

    ``max_input_tokens`` caps the *full* model input.  Special tokens and a
    small safety margin are reserved before ordinary text is packed.  Oversized
    individual sentences are greedily split by words.
    """
    if safety_margin < 0:
        raise ValueError("safety_margin must be non-negative")
    model_limit = _effective_model_limit(tokenizer)
    full_limit = model_limit if max_input_tokens is None else min(model_limit, int(max_input_tokens))
    try:
        special = int(tokenizer.num_special_tokens_to_add(pair=False))
    except Exception:
        special = 2
    budget = full_limit - special - safety_margin
    if budget <= 0:
        raise ValueError("model input budget is too small after reserving special tokens")

    sentences = split_sentences(text)
    if not sentences:
        return []

    units: list[str] = []
    for sentence in sentences:
        if _token_count(tokenizer, sentence) <= budget:
            units.append(sentence)
        else:
            units.extend(_word_chunks_for_oversized_unit(sentence, tokenizer, budget))

    chunks: list[str] = []
    current = ""
    for unit in units:
        trial = unit if not current else f"{current} {unit}"
        if current and _token_count(tokenizer, trial) > budget:
            chunks.append(current)
            current = unit
        else:
            current = trial
    if current:
        chunks.append(current)

    # This is a core safety invariant; do not let model truncation hide a bug.
    if any(_token_count(tokenizer, c) > budget for c in chunks):
        raise AssertionError("constructed chunk exceeds token budget")
    return chunks


def chunk_words(text: str, max_words: int = 700):
    """Legacy word-count chunker retained for backwards compatibility only."""
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])


@dataclass(frozen=True)
class HierarchyResult:
    text: str
    passes: int
    leaf_chunks: int


def hierarchical_reduce(
    text: str,
    *,
    tokenizer: TokenizerLike,
    summarize_chunk: Callable[[str], str],
    max_input_tokens: int | None = None,
    max_passes: int = 6,
) -> HierarchyResult:
    """Map-reduce a long transcript until one model-safe summary remains.

    ``summarize_chunk`` is injected so the hierarchy can be tested without a
    transformer model.  Each pass re-chunks the joined summaries.  The routine
    fails rather than looping forever if a summarizer does not actually reduce
    the document enough.
    """
    if max_passes < 1:
        raise ValueError("max_passes must be at least 1")
    chunks = chunk_for_token_budget(
        text, tokenizer, max_input_tokens=max_input_tokens
    )
    if not chunks:
        return HierarchyResult("", 0, 0)
    leaf_chunks = len(chunks)

    current_chunks = chunks
    previous_size = sum(_token_count(tokenizer, c) for c in current_chunks)
    for pass_no in range(1, max_passes + 1):
        summaries = [summarize_chunk(c).strip() for c in current_chunks]
        combined = " ".join(s for s in summaries if s).strip()
        next_chunks = chunk_for_token_budget(
            combined, tokenizer, max_input_tokens=max_input_tokens
        )
        if len(next_chunks) <= 1:
            return HierarchyResult(combined, pass_no, leaf_chunks)

        next_size = sum(_token_count(tokenizer, c) for c in next_chunks)
        if next_size >= previous_size:
            raise RuntimeError(
                "hierarchical summarizer did not reduce token count; aborting to avoid an infinite loop"
            )
        previous_size = next_size
        current_chunks = next_chunks

    raise RuntimeError("hierarchical summarization exceeded max_passes")


def summarize(
    text: str,
    model_name: str = "sshleifer/distilbart-cnn-12-6",
    *,
    max_input_tokens: int | None = None,
) -> str:
    """Summarise ``text`` with a token-budgeted hierarchical pipeline."""
    if not text.strip():
        return ""
    from transformers import pipeline  # lazy import - heavy dependency

    summarizer = pipeline("summarization", model=model_name)
    tokenizer = summarizer.tokenizer

    def run(chunk: str) -> str:
        n_tokens = max(1, _token_count(tokenizer, chunk))
        # Output limits are generation limits, not input-window limits.  Keep the
        # target moderate and ensure min < max even for very short chunks.
        max_new = max(24, min(160, int(round(n_tokens * 0.35))))
        min_new = max(5, min(40, max_new - 8))
        out = summarizer(
            chunk,
            max_new_tokens=max_new,
            min_new_tokens=min_new,
            do_sample=False,
            truncation=False,
        )
        return out[0]["summary_text"].strip()

    return hierarchical_reduce(
        text,
        tokenizer=tokenizer,
        summarize_chunk=run,
        max_input_tokens=max_input_tokens,
    ).text

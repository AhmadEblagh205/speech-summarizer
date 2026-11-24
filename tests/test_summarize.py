import math
import pytest

from summarize import (
    chunk_for_token_budget,
    chunk_words,
    hierarchical_reduce,
    split_sentences,
)


class CharTokenizer:
    """One pseudo-token per four characters of each whitespace word."""
    model_max_length = 32

    def encode(self, text, add_special_tokens=False):
        out = []
        for word in text.split():
            out.extend([1] * max(1, math.ceil(len(word) / 4)))
        return out

    def num_special_tokens_to_add(self, pair=False):
        return 2


def test_legacy_chunk_words_preserves_all_words_in_order():
    text = " ".join(f"w{i}" for i in range(2500))
    chunks = list(chunk_words(text, max_words=700))
    assert len(chunks) == 4
    assert " ".join(chunks).split() == text.split()


def test_legacy_chunk_words_rejects_nonpositive_limit():
    with pytest.raises(ValueError):
        list(chunk_words("a b", 0))


def test_sentence_split_is_deterministic():
    assert split_sentences("One sentence. Two? Three!") == [
        "One sentence.", "Two?", "Three!"
    ]


def test_token_chunking_respects_budget_and_preserves_word_order():
    tok = CharTokenizer()
    text = "Short sentence. " + "abcdefghijk " * 30 + "Final point."
    chunks = chunk_for_token_budget(text, tok, safety_margin=4)
    budget = tok.model_max_length - 2 - 4
    assert chunks
    assert all(len(tok.encode(c)) <= budget for c in chunks)
    assert " ".join(chunks).split() == " ".join(text.split()).split()


def test_token_chunker_splits_oversized_sentence_by_words():
    tok = CharTokenizer()
    text = "abcdefgh abcdefgh abcdefgh abcdefgh abcdefgh abcdefgh abcdefgh."
    chunks = chunk_for_token_budget(text, tok, max_input_tokens=16, safety_margin=2)
    assert len(chunks) > 1
    assert " ".join(chunks).split() == text.split()


def test_pathological_single_word_fails_instead_of_truncating():
    tok = CharTokenizer()
    with pytest.raises(ValueError):
        chunk_for_token_budget("x" * 500, tok, max_input_tokens=16, safety_margin=2)


def test_hierarchical_reduce_compresses_until_one_safe_chunk():
    tok = CharTokenizer()
    text = " ".join(f"word{i}." for i in range(300))

    def compress(chunk):
        return " ".join(chunk.split()[::4])

    result = hierarchical_reduce(text, tokenizer=tok, summarize_chunk=compress)
    assert result.leaf_chunks > 1
    assert result.passes >= 2
    # Returned result must itself fit one chunk under the same policy.
    assert len(chunk_for_token_budget(result.text, tok)) == 1


def test_hierarchical_reduce_detects_noncompressing_summarizer():
    tok = CharTokenizer()
    text = " ".join(f"word{i}." for i in range(200))
    with pytest.raises(RuntimeError, match="did not reduce"):
        hierarchical_reduce(text, tokenizer=tok, summarize_chunk=lambda x: x)

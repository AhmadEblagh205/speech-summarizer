"""Pure-Python audit experiments for metric and chunking behaviour."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from evaluate import rouge_scores, wer_counts
from summarize import chunk_for_token_budget, hierarchical_reduce

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


class ToyTokenizer:
    """Deterministic subword surrogate for orchestration stress tests.

    Each whitespace-delimited word consumes ceil(len(word)/4) pseudo-tokens.
    It is deliberately *not* a language-model tokenizer and is used only to
    stress the token-budget logic without downloading a transformer model.
    """
    model_max_length = 128

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False):
        ids: list[int] = []
        for word in text.split():
            ids.extend([1] * max(1, math.ceil(len(word) / 4)))
        return ids

    @staticmethod
    def num_special_tokens_to_add(pair: bool = False) -> int:
        return 2


def _metric_demo() -> dict:
    ref = "alpha beta gamma delta"
    reversed_hyp = "delta gamma beta alpha"
    rouge = rouge_scores(ref, reversed_hyp)
    wer = wer_counts("one two three four", "one too three four five")
    return {
        "reversed_tokens_rouge1_f": rouge["rouge1_f"],
        "reversed_tokens_rougeL_f": rouge["rougeL_f"],
        "wer_example": {
            "wer": wer.wer,
            "substitutions": wer.substitutions,
            "deletions": wer.deletions,
            "insertions": wer.insertions,
        },
    }


def _chunk_stress(seed: int = 18, worlds: int = 500) -> dict:
    rng = random.Random(seed)
    tok = ToyTokenizer()
    violations = 0
    max_chunks = 0
    for _ in range(worlds):
        n = rng.randint(200, 4000)
        words = []
        for i in range(n):
            # Variable word lengths make pseudo-token/word ratios non-constant.
            k = rng.randint(2, 15)
            word = chr(97 + i % 26) * k
            punct = "." if i % rng.randint(8, 25) == 0 else ""
            words.append(word + punct)
        text = " ".join(words)
        chunks = chunk_for_token_budget(text, tok, safety_margin=8)
        budget = tok.model_max_length - tok.num_special_tokens_to_add(False) - 8
        counts = [len(tok.encode(c)) for c in chunks]
        max_chunks = max(max_chunks, len(chunks))
        violations += sum(c > budget for c in counts)
    return {
        "worlds": worlds,
        "token_budget_violations": violations,
        "max_chunks_observed": max_chunks,
    }


def _hierarchy_demo() -> dict:
    tok = ToyTokenizer()
    text = " ".join(f"concept{i}." for i in range(2500))

    def compress(chunk: str) -> str:
        words = chunk.split()
        # Deterministic 4:1 map step, preserving order.
        return " ".join(words[::4])

    result = hierarchical_reduce(text, tokenizer=tok, summarize_chunk=compress)
    return {
        "source_words": len(text.split()),
        "leaf_chunks": result.leaf_chunks,
        "passes": result.passes,
        "final_words": len(result.text.split()),
    }


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    payload = {
        "note": "Pure algorithm/metric audit; no Whisper or summarisation model was run.",
        "metrics": _metric_demo(),
        "chunk_stress": _chunk_stress(),
        "hierarchy": _hierarchy_demo(),
    }
    (REPORTS / "audit_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

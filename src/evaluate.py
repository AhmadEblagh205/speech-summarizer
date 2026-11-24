"""Dependency-light evaluation for transcription and summarisation.

The implementation is intentionally transparent.  Metrics use a documented
normalisation: Unicode alphanumerics are lower-cased; punctuation becomes
whitespace; no stemming or synonym matching is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def normalised_words(text: str) -> list[str]:
    return "".join(
        c.lower() if c.isalnum() or c.isspace() else " " for c in text
    ).split()


@dataclass(frozen=True)
class WERCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        if self.reference_words == 0:
            return 0.0 if self.errors == 0 else 1.0
        return self.errors / self.reference_words


def wer_counts(reference: str, hypothesis: str) -> WERCounts:
    """Return substitution/deletion/insertion counts from word Levenshtein."""
    r, h = normalised_words(reference), normalised_words(hypothesis)
    nr, nh = len(r), len(h)
    # dp cells store (edit_count, substitutions, deletions, insertions).
    dp: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(nh + 1)] for _ in range(nr + 1)
    ]
    for i in range(1, nr + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, nh + 1):
        dp[0][j] = (j, 0, 0, j)

    for i in range(1, nr + 1):
        for j in range(1, nh + 1):
            if r[i - 1] == h[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            e, s, d, ins = dp[i - 1][j - 1]
            sub = (e + 1, s + 1, d, ins)
            e, s, d, ins = dp[i - 1][j]
            dele = (e + 1, s, d + 1, ins)
            e, s, d, ins = dp[i][j - 1]
            insert = (e + 1, s, d, ins + 1)
            # Deterministic tie-break order: substitution, deletion, insertion.
            dp[i][j] = min((sub, dele, insert), key=lambda x: x[0])

    _, s, d, ins = dp[nr][nh]
    return WERCounts(s, d, ins, nr)


def word_error_rate(reference: str, hypothesis: str) -> float:
    return wer_counts(reference, hypothesis).wer


def corpus_word_error_rate(pairs: Iterable[tuple[str, str]]) -> float:
    """Corpus WER = total edits / total reference words, not mean utterance WER."""
    errors = 0
    refs = 0
    empty_hyp_errors = 0
    for reference, hypothesis in pairs:
        counts = wer_counts(reference, hypothesis)
        errors += counts.errors
        refs += counts.reference_words
        if counts.reference_words == 0:
            empty_hyp_errors += counts.errors
    if refs == 0:
        return 0.0 if errors == 0 else 1.0
    return errors / refs


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _prf(overlap: int, ref_n: int, hyp_n: int) -> tuple[float, float, float]:
    if ref_n == 0 or hyp_n == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / hyp_n
    recall = overlap / ref_n
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def rouge_scores(reference: str, hypothesis: str) -> dict[str, float]:
    """Lightweight ROUGE-1 and ROUGE-L precision/recall/F1.

    This is not a drop-in replacement for every ROUGE package: it uses the
    repository's explicit word normalisation and no stemming.
    """
    r, h = normalised_words(reference), normalised_words(hypothesis)
    if not r or not h:
        return {
            "rouge1_p": 0.0,
            "rouge1_r": 0.0,
            "rouge1_f": 0.0,
            "rougeL_p": 0.0,
            "rougeL_r": 0.0,
            "rougeL_f": 0.0,
        }
    unigram_overlap = sum((Counter(r) & Counter(h)).values())
    r1p, r1r, r1f = _prf(unigram_overlap, len(r), len(h))
    lcs = _lcs(r, h)
    rlp, rlr, rlf = _prf(lcs, len(r), len(h))
    return {
        "rouge1_p": r1p,
        "rouge1_r": r1r,
        "rouge1_f": r1f,
        "rougeL_p": rlp,
        "rougeL_r": rlr,
        "rougeL_f": rlf,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--wer", nargs=2, metavar=("REF", "HYP"))
    group.add_argument("--rouge", nargs=2, metavar=("REF", "HYP"))
    args = ap.parse_args()
    if args.wer:
        ref, hyp = (Path(f).read_text(encoding="utf-8") for f in args.wer)
        c = wer_counts(ref, hyp)
        print(f"WER: {c.wer:.3f} (S={c.substitutions}, D={c.deletions}, I={c.insertions})")
    else:
        ref, hyp = (Path(f).read_text(encoding="utf-8") for f in args.rouge)
        for k, v in rouge_scores(ref, hyp).items():
            print(f"{k}: {v:.3f}")


if __name__ == "__main__":
    main()

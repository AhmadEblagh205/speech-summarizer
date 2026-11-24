import math

from evaluate import (
    corpus_word_error_rate,
    normalised_words,
    rouge_scores,
    wer_counts,
    word_error_rate,
)


def test_normalisation_is_explicit_and_case_punctuation_insensitive():
    assert normalised_words("Hello, WORLD!") == ["hello", "world"]


def test_wer_identity_and_known_insertion():
    assert word_error_rate("the cat sat", "the cat sat") == 0.0
    assert math.isclose(word_error_rate("the cat sat", "the cat sat down"), 1 / 3)


def test_wer_breakdown_known_substitution_and_insertion():
    c = wer_counts("one two three four", "one too three four five")
    assert c.substitutions == 1
    assert c.deletions == 0
    assert c.insertions == 1
    assert c.reference_words == 4
    assert c.wer == 0.5


def test_empty_reference_convention():
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "hello") == 1.0


def test_corpus_wer_is_reference_word_weighted_not_mean_of_utterances():
    pairs = [
        ("a", "x"),          # 1 / 1
        ("a b c d e f g h i", "a b c d e f g h x"),  # 1 / 9
    ]
    assert math.isclose(corpus_word_error_rate(pairs), 2 / 10)


def test_rouge_identity_and_empty():
    same = rouge_scores("results improved after tuning", "results improved after tuning")
    assert same["rouge1_f"] == 1.0
    assert same["rougeL_f"] == 1.0
    empty = rouge_scores("reference", "")
    assert all(v == 0.0 for v in empty.values())


def test_rouge1_ignores_order_but_rougeL_does_not():
    score = rouge_scores("alpha beta gamma delta", "delta gamma beta alpha")
    assert score["rouge1_f"] == 1.0
    assert score["rougeL_f"] == 0.25


def test_rouge_duplicate_tokens_use_multiset_overlap():
    score = rouge_scores("a a b", "a b b")
    assert math.isclose(score["rouge1_f"], 2 / 3)

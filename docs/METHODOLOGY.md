# Methodology and evidence notes

## 1. Scope

This repository is an orchestration and evaluation study around two pretrained model families:

- Whisper-compatible speech-to-text;
- sequence-to-sequence abstractive summarisation.

It does not train either foundation model from scratch and currently does not claim application-specific model performance.

## 2. Transcription backend

`backend=auto` selects the first installed backend in this order:

1. `faster-whisper`;
2. `openai-whisper`.

The selection only handles package availability. Runtime/model errors are allowed to surface rather than silently changing backend after a failed run, because otherwise two experiments with the same command could be produced by different models without an obvious provenance signal.

`TranscriptionResult` records backend, model size and detected/requested language metadata alongside the transcript.

## 3. Long-document chunking

The old project used 700 words as an approximate transformer input size. The audited implementation instead obtains the tokenizer's finite model limit, subtracts special-token requirements and a safety margin, and greedily packs sentence-like units until the remaining token budget would be exceeded.

If one sentence alone is too long, it is split greedily by words. A single whitespace-delimited unit that itself exceeds the remaining model budget causes a failure; the code does not silently truncate it.

Sentence splitting is intentionally lightweight and deterministic. It should not be interpreted as a linguistically complete sentence-segmentation algorithm.

## 4. Hierarchical summarisation

Let `C_0` be the set of model-safe transcript chunks. At pass `k`:

1. summarise every chunk in `C_k`;
2. concatenate those summaries;
3. re-tokenise/re-chunk the concatenation;
4. stop if it fits one model-safe chunk;
5. otherwise continue.

A non-compressing summariser causes an explicit error instead of an unbounded loop.

This is a practical map-reduce architecture. It does not eliminate information loss caused by summarising chunks independently, nor does it solve cross-chunk coreference or factuality.

## 5. WER

For a reference with `N` normalised words:

`WER = (substitutions + deletions + insertions) / N`.

Normalisation is lower-case alphanumeric word matching with punctuation replaced by whitespace. Different WER toolkits can return different numbers if their text normalisation differs.

Corpus WER pools edit counts and reference-word counts across items. It does not average item-level WERs.

## 6. ROUGE

The local implementation reports ROUGE-1 and ROUGE-L precision, recall and F1. It performs no stemming.

ROUGE-1 measures unigram overlap and is insensitive to ordering once the same multiset of words is present. ROUGE-L uses longest common subsequence and therefore carries some sequence information.

Neither metric directly proves factual correctness, coverage of all key ideas, or usefulness as revision notes.

## 7. Dependency-free audit experiments

`src/audit_experiments.py` intentionally avoids heavy model dependencies. It runs three checks:

- controlled WER / ROUGE metric examples;
- randomised token-budget stress using a documented pseudo-subword tokenizer;
- hierarchical control-flow stress using a deterministic compressor.

These are implementation checks only. They must never be presented as Whisper or DistilBART performance.

## 8. Model-performance release condition

Before a WER/ROUGE headline can be published, the repository should contain or document a fixed evaluation set with human references, fixed model versions/configurations, hardware/runtime metadata and a reproducible command that regenerates the result.

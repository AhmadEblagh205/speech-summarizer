# Speech Summariser — long-transcript orchestration & evaluation lab

[![CI](https://github.com/AhmadEblagh205/speech-summarizer/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmadEblagh205/speech-summarizer/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A local pipeline for turning an audio recording into a transcript and then an abstractive summary. Whisper provides speech-to-text; a Hugging Face summarisation model provides the text reduction; the audited part of the project is the orchestration around them: **token-safe long-document chunking, hierarchical reduction, run provenance and transparent WER/ROUGE evaluation**.

The original version was built to condense recorded lectures into revision notes. The current version is deliberately more careful about what has and has not been demonstrated: **this repository does not publish a model-level WER or ROUGE performance claim until labelled audio/transcript/summary references are actually run through the heavy models.**

## What changed in the audit

The biggest problem in the old design was the statement that roughly `700 words` was a safe chunk for a transformer summariser. Model context windows are measured in **tokens, not words**. Word count is only a loose proxy, and a 700-word transcript chunk can map to very different numbers of subword tokens depending on language and vocabulary.

V0.3 therefore:

- chunks against the **actual summariser tokenizer and declared model input window**;
- preserves sentence boundaries when possible and falls back to word-level splitting only for an oversized sentence;
- fails rather than silently truncating an input that cannot fit;
- recursively performs map-reduce summarisation until the joined summary fits a single model-safe chunk;
- aborts if the summariser is not reducing the document enough to converge;
- records each run in its own directory instead of overwriting `transcript.txt` and `summary.txt`;
- records the transcription backend/model and summary-model configuration;
- makes backend choice explicit rather than silently changing models after a runtime failure;
- expands WER into substitution/deletion/insertion counts and adds corpus-level WER;
- reports ROUGE-1 and ROUGE-L precision, recall and F1 with the repository's normalisation stated explicitly.

## Architecture

```text
Audio
  |
  v
Whisper backend
(faster-whisper preferred; openai-whisper optional)
  |
  v
Transcript + transcription metadata
  |
  v
Tokenizer-aware sentence-preserving chunker
  |
  v
Chunk summaries
  |
  +---- if joined summaries still exceed model window ----+
  |                                                       |
  +---------------- hierarchical reduce <-----------------+
  |
  v
Final summary + run metadata
```

`--text` skips transcription and runs the same summarisation path on an existing transcript.

## Why token-aware chunking matters

The summariser now asks the loaded tokenizer how many tokens each candidate chunk consumes. It reserves the model's special tokens plus a small safety margin and ensures every call stays inside the input budget with `truncation=False`.

That distinction matters because silent truncation would create a deceptively successful pipeline while dropping transcript content before summarisation.

A dependency-free stress test uses a deliberately variable pseudo-subword tokenizer to exercise the orchestration without pretending it is the DistilBART tokenizer. Across **500 random long-document stress cases**, the audited chunker produced **zero token-budget violations**. That is a software invariant test, not a statement about summary quality.

## Hierarchical reduction

For a long transcript, each safe chunk is summarised independently. The chunk summaries are joined, re-tokenised, and if still too long, re-chunked and summarised again. This repeats until the text fits one model-safe chunk.

A deterministic dependency-free stress case begins with 2,500 words, creates 65 first-pass chunks under its toy tokenizer, and reaches a single safe result in four reduction passes. The fake compressor exists only to test control flow; it is **not** a language-model benchmark.

## Evaluation

### Word error rate

WER is word-level Levenshtein error divided by reference-word count:

\[
\mathrm{WER}=\frac{S+D+I}{N_{ref}}.
\]

The implementation exposes substitutions, deletions and insertions separately. Corpus WER is computed as **total edits / total reference words**, rather than averaging utterance-level WERs and accidentally giving short recordings equal weight to long ones.

Normalisation in this repository is explicit: Unicode alphanumeric characters are lower-cased, punctuation becomes whitespace, and there is no stemming or synonym matching. WER values from another toolkit are only directly comparable when its normalisation matches.

### ROUGE

The repository implements lightweight word-normalised:

- ROUGE-1 precision / recall / F1;
- ROUGE-L precision / recall / F1.

These are transparent educational implementations, not a claim of bit-for-bit compatibility with every ROUGE package or stemming configuration.

A useful audit example shows why both metrics are kept. For:

```text
reference: alpha beta gamma delta
summary:   delta gamma beta alpha
```

ROUGE-1 F1 is **1.00** because every unigram is present, while ROUGE-L F1 is **0.25** because word order has collapsed. High unigram overlap is therefore not equivalent to preserving sequence or meaning.

ROUGE also does **not** directly measure factual consistency. A summary can overlap strongly with a reference and still contain unsupported statements.

## Usage

```bash
pip install -r requirements.txt

# Audio -> transcript -> summary
python src/cli.py lecture.mp3

# Explicit transcription backend/model
python src/cli.py lecture.mp3 --backend faster-whisper --model-size small

# Existing transcript -> summary only
python src/cli.py --text transcript.txt
```

Each invocation writes a unique directory below `reports/runs/` containing:

```text
transcript.txt
summary.txt
run.json
```

so one experiment cannot silently overwrite another.

### Evaluate labelled examples

```bash
python src/evaluate.py --wer reference_transcript.txt model_transcript.txt
python src/evaluate.py --rouge reference_summary.txt model_summary.txt
```

Example WER output includes the error decomposition:

```text
WER: 0.125 (S=3, D=1, I=2)
```

## Current evidence boundary

The CI environment intentionally does **not** download Whisper, DistilBART or audio corpora. The current audited evidence is therefore about the pipeline logic and evaluation implementation:

- **21 lightweight tests pass** without heavy model dependencies;
- **500 random token-budget stress cases produce zero violations** under the documented toy-tokenizer audit;
- WER edit counts and known values are regression-tested;
- ROUGE order sensitivity is regression-tested;
- hierarchical reduction is tested for both convergence and fail-closed behaviour;
- backend selection and missing-input handling are tested;
- all heavy imports remain lazy.

What is **not** yet claimed:

- a benchmark WER for Whisper on this application;
- a benchmark ROUGE score for DistilBART on lecture summaries;
- factuality or pedagogical-quality gains;
- real-time throughput;
- superiority of faster-whisper over openai-whisper in this repository's own hardware experiment.

Those require a labelled evaluation set and an actual model run.

## Reproducible audit

```bash
pytest -q
PYTHONPATH=src python src/audit_experiments.py
```

The second command writes `reports/audit_summary.json`. Its pseudo-tokenizer and deterministic compressor are **software stress fixtures only** and are labelled as such in the report.

## Next scientific step

A defensible model comparison would use a small held-out corpus of lecture/audio clips with:

1. human reference transcripts;
2. human-written reference summaries;
3. fixed model versions and decoding settings;
4. WER plus error decomposition for speech-to-text;
5. ROUGE plus factuality/human usefulness checks for summarisation;
6. repeated runtime measurements on named hardware.

Only then should the portfolio surface a headline such as `WER = ...` or `ROUGE-L = ...`.

## Licence

MIT, see [LICENSE](LICENSE).

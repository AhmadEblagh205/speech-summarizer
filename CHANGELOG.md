# Changelog

## 0.3.0 — defensibility audit

- replaced word-count-only summarisation chunks with tokenizer-aware input budgets;
- added sentence-preserving chunk construction with fail-closed oversize handling;
- added recursive map-reduce summarisation with convergence protection;
- changed CLI outputs to unique run directories with provenance metadata;
- made Whisper backend selection explicit and recorded backend/model metadata;
- expanded WER to substitution/deletion/insertion counts and corpus WER;
- expanded ROUGE-1 / ROUGE-L to precision, recall and F1 with explicit normalisation;
- added dependency-free metric, chunk-budget and hierarchy audit experiments;
- added `docs/METHODOLOGY.md` and a clear model-performance evidence boundary;
- expanded the lightweight test suite from 3 to 21 passing tests.

## 0.2.0 — update round 4

Added a from-scratch evaluation module: word error rate by word-level edit distance and ROUGE-1 and ROUGE-L F scores, runnable and tested with no models installed.

## 0.1.0

Initial release.

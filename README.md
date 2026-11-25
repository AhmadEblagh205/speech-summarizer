# Speech Summariser — long-transcript orchestration & evaluation lab

[![CI](https://github.com/AhmadEblagh205/speech-summarizer/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmadEblagh205/speech-summarizer/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

I built this to turn recorded lectures into revision notes. Whisper converts the audio to a transcript and a Hugging Face summarisation model reduces it. My work is the code around those two models: chunking that respects the model's token limit, repeated summarisation of long transcripts, run records, and WER and ROUGE evaluation. I have not run labelled audio through the models yet, so I do not quote a WER or ROUGE score for them.

## What it does

- transcribes audio with faster-whisper or openai-whisper, chosen explicitly;
- splits a transcript into chunks using the summariser's own tokenizer, keeping sentences together and refusing to truncate;
- summarises each chunk and repeats the process on the joined summaries until the result fits one chunk;
- writes every run to its own directory under `reports/runs/` with the transcript, the summary and `run.json`;
- computes WER (with substitution, deletion and insertion counts) and ROUGE-1 and ROUGE-L precision, recall and F1.

## Results

The tests and checks use no Whisper or summarisation model, so they show that the code works and say nothing about summary quality. 21 lightweight tests pass. In a stress test of 500 random long documents with a toy tokenizer, the chunker produced zero token-budget violations. One hierarchical case starts at 2,500 words, makes 65 first-pass chunks and reaches a single result in four passes, using a fake compressor.

For the reference "alpha beta gamma delta" and the summary "delta gamma beta alpha", ROUGE-1 F1 is 1.00 and ROUGE-L F1 is 0.25, because the word order is lost.

I have not measured Whisper WER, DistilBART ROUGE, factual consistency or speed on real lecture data. That needs reference transcripts and summaries.

## How to run

```bash
pip install -r requirements.txt

# Audio -> transcript -> summary
python src/cli.py lecture.mp3

# Explicit transcription backend/model
python src/cli.py lecture.mp3 --backend faster-whisper --model-size small

# Existing transcript -> summary only
python src/cli.py --text transcript.txt

# Evaluate labelled examples
python src/evaluate.py --wer reference_transcript.txt model_transcript.txt
python src/evaluate.py --rouge reference_summary.txt model_summary.txt

pytest -q
PYTHONPATH=src python src/audit_experiments.py   # writes reports/audit_summary.json
```

## Notes

CI does not download Whisper, DistilBART or any audio, and the model imports are lazy. The stress test uses a pseudo-tokenizer and a fake compressor, which are fixtures only. The WER and ROUGE code is my own lightweight implementation: text is lower-cased, punctuation becomes whitespace, and there is no stemming, so scores from other packages may differ. Corpus WER is total edits over total reference words. What changed from my first version, and the design of the chunker and the metrics, is in docs/METHODOLOGY.md.

## Licence

MIT, see [LICENSE](LICENSE).

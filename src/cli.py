"""Turn a speech recording or transcript into a summary with run metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "runs"


def _safe_stem(path: str | Path) -> str:
    stem = Path(path).stem or "input"
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?", help="audio file (mp3/wav/m4a...)")
    ap.add_argument("--text", help="summarise an existing transcript instead")
    ap.add_argument("--model-size", default="base", help="Whisper size")
    ap.add_argument("--backend", default="auto", choices=["auto", "faster-whisper", "openai-whisper"])
    ap.add_argument("--language", default=None)
    ap.add_argument("--summary-model", default="sshleifer/distilbart-cnn-12-6")
    ap.add_argument("--max-input-tokens", type=int, default=None)
    args = ap.parse_args()
    if bool(args.audio) == bool(args.text):
        ap.error("provide exactly one of an audio file or --text transcript")

    input_path = args.text or args.audio
    path = Path(input_path)
    if not path.is_file():
        ap.error(f"input file not found: {path}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + _safe_stem(path)
    run_dir = OUT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata: dict[str, object] = {
        "run_id": run_id,
        "input_path": str(path),
        "mode": "text" if args.text else "audio",
        "summary_model": args.summary_model,
        "max_input_tokens": args.max_input_tokens,
    }

    if args.text:
        transcript = path.read_text(encoding="utf-8")
    else:
        from transcribe import transcribe_with_metadata

        print(f"Transcribing {path} ({args.backend}, Whisper {args.model_size})...")
        result = transcribe_with_metadata(
            path,
            model_size=args.model_size,
            language=args.language,
            backend=args.backend,
        )
        transcript = result.text
        metadata["transcription"] = result.metadata()

    (run_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
    metadata["transcript_words"] = len(transcript.split())

    from summarize import summarize

    print("Summarising...")
    summary = summarize(
        transcript,
        model_name=args.summary_model,
        max_input_tokens=args.max_input_tokens,
    )
    (run_dir / "summary.txt").write_text(summary, encoding="utf-8")
    metadata["summary_words"] = len(summary.split())
    metadata["compression_ratio_words"] = (
        len(summary.split()) / len(transcript.split()) if transcript.split() else 0.0
    )
    (run_dir / "run.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"\n--- Summary ---\n{summary}\n")
    print(f"Saved run -> {run_dir}")


if __name__ == "__main__":
    main()

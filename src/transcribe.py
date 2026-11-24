"""Speech-to-text backends with explicit provenance metadata."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import importlib.util
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    backend: str
    model_size: str
    language: str | None
    language_probability: float | None = None

    def metadata(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("text", None)
        return d


def available_backends() -> list[str]:
    found: list[str] = []
    if importlib.util.find_spec("faster_whisper") is not None:
        found.append("faster-whisper")
    if importlib.util.find_spec("whisper") is not None:
        found.append("openai-whisper")
    return found


def _choose_backend(backend: str) -> str:
    if backend not in {"auto", "faster-whisper", "openai-whisper"}:
        raise ValueError("backend must be auto, faster-whisper or openai-whisper")
    found = available_backends()
    if backend == "auto":
        if found:
            return found[0]
        raise RuntimeError(
            "no Whisper backend installed; install faster-whisper (preferred) or openai-whisper"
        )
    if backend not in found:
        raise RuntimeError(f"requested backend {backend!r} is not installed")
    return backend


def transcribe_with_metadata(
    audio_path: str | Path,
    model_size: str = "base",
    language: str | None = None,
    *,
    backend: str = "auto",
) -> TranscriptionResult:
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    chosen = _choose_backend(backend)

    if chosen == "faster-whisper":
        from faster_whisper import WhisperModel

        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, info = model.transcribe(str(path), language=language)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return TranscriptionResult(
            text=text,
            backend=chosen,
            model_size=model_size,
            language=getattr(info, "language", language),
            language_probability=getattr(info, "language_probability", None),
        )

    import whisper

    model = whisper.load_model(model_size)
    result = model.transcribe(str(path), language=language)
    return TranscriptionResult(
        text=result["text"].strip(),
        backend=chosen,
        model_size=model_size,
        language=result.get("language", language),
        language_probability=None,
    )


def transcribe(
    audio_path: str | Path,
    model_size: str = "base",
    language: str | None = None,
    *,
    backend: str = "auto",
) -> str:
    """Backwards-compatible text-only wrapper."""
    return transcribe_with_metadata(
        audio_path, model_size=model_size, language=language, backend=backend
    ).text

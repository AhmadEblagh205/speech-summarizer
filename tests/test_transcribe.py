from pathlib import Path
import pytest

import transcribe as tr


def test_backend_name_validation():
    with pytest.raises(ValueError):
        tr._choose_backend("mystery")


def test_auto_backend_fails_clearly_when_none_installed(monkeypatch):
    monkeypatch.setattr(tr, "available_backends", lambda: [])
    with pytest.raises(RuntimeError, match="no Whisper backend installed"):
        tr._choose_backend("auto")


def test_auto_backend_prefers_first_available(monkeypatch):
    monkeypatch.setattr(tr, "available_backends", lambda: ["faster-whisper", "openai-whisper"])
    assert tr._choose_backend("auto") == "faster-whisper"


def test_transcribe_checks_path_before_loading_models(tmp_path):
    with pytest.raises(FileNotFoundError):
        tr.transcribe_with_metadata(tmp_path / "missing.wav")

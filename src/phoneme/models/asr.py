"""ASR backend factory — pick Whisper / MLX-Whisper / Parakeet from Settings."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from ..config import ASRBackend, Settings
from ..runtime import has_mlx, is_apple_silicon

log = logging.getLogger(__name__)


class ASR(Protocol):
    def load(self) -> None: ...
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000, language: str | None = "en") -> list: ...


def build_asr(settings: Settings) -> ASR:
    """Return the ASR backend the user asked for, falling back gracefully.

    If the user selected an Apple-Silicon-only backend on non-Mac hardware,
    we log a warning and fall through to faster-whisper so the app still
    functions.
    """
    backend = settings.asr_backend

    if backend is ASRBackend.MLX_WHISPER:
        if has_mlx():
            from .asr_mlx import MLXWhisperASR

            return MLXWhisperASR(model_size=settings.whisper_model_size)
        log.warning("MLX Whisper requested but MLX unavailable; falling back to faster-whisper")
        backend = ASRBackend.FASTER_WHISPER

    if backend is ASRBackend.PARAKEET_MLX:
        if is_apple_silicon():
            try:
                from .asr_parakeet import ParakeetMLXASR

                return ParakeetMLXASR(model_id=settings.parakeet_model)
            except Exception as e:
                log.warning("Parakeet-MLX init failed (%s); falling back to faster-whisper", e)
        else:
            log.warning(
                "Parakeet-MLX requested on non-Apple-Silicon host; falling back to faster-whisper"
            )
        backend = ASRBackend.FASTER_WHISPER

    from .asr_whisper import FasterWhisperASR

    return FasterWhisperASR(model_size=settings.whisper_model_size)

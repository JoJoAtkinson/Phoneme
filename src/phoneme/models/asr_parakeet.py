"""Parakeet-TDT via parakeet-mlx (Apple Silicon only).

Heavier and slower than Whisper-small on CPU but produces noticeably cleaner
transcripts in practice, especially on short utterances. Only imported on Mac
arm64 hardware; other platforms should pick the Whisper backend.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class WordToken:
    text: str
    start_s: float
    end_s: float
    confidence: float


class ParakeetMLXASR:
    def __init__(self, model_id: str = "mlx-community/parakeet-tdt-0.6b-v2"):
        self.model_id = model_id
        self._model = None

    def available(self) -> bool:
        return sys.platform == "darwin" and (
            "arm64" in (__import__("platform").machine().lower())
        )

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.available():
            raise RuntimeError(
                "Parakeet-MLX only runs on Apple Silicon. Use faster-whisper instead."
            )
        from parakeet_mlx import from_pretrained  # type: ignore

        log.info("loading parakeet-mlx: %s", self.model_id)
        self._model = from_pretrained(self.model_id)

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000, language: str | None = "en"
    ) -> list[WordToken]:
        self.load()
        # parakeet-mlx returns a Result with .sentences[].tokens[] or similar.
        result = self._model.transcribe(audio, sampling_rate=sample_rate)
        out: list[WordToken] = []
        # Normalize across possible result shapes
        for sentence in getattr(result, "sentences", [result]):
            tokens = getattr(sentence, "tokens", None) or getattr(sentence, "words", None)
            if tokens is None:
                text = getattr(sentence, "text", "") or str(sentence)
                if text:
                    out.append(WordToken(text.strip(), 0.0, 0.0, 1.0))
                continue
            for w in tokens:
                out.append(
                    WordToken(
                        text=getattr(w, "text", str(w)).strip(),
                        start_s=float(getattr(w, "start", 0.0) or 0.0),
                        end_s=float(getattr(w, "end", 0.0) or 0.0),
                        confidence=float(getattr(w, "probability", 1.0) or 1.0),
                    )
                )
        return out

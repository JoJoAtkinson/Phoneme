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
        assert self._model is not None

        # parakeet-mlx.transcribe() takes a file path, not a raw array. Write
        # the utterance to a short-lived temp WAV and hand it the path.
        import tempfile

        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, audio, sample_rate, subtype="FLOAT")
            result = self._model.transcribe(tmp.name)

        # AlignedResult → AlignedSentence → AlignedToken; word timings live
        # on the tokens. NVIDIA Parakeet checkpoints are English-only, so no
        # language routing is needed here.
        out: list[WordToken] = []
        for sentence in result.sentences:
            for w in sentence.tokens:
                text = w.text.strip()
                if not text:
                    continue
                out.append(
                    WordToken(
                        text=text,
                        start_s=float(w.start),
                        end_s=float(w.end),
                        confidence=float(w.confidence),
                    )
                )
        return out

"""faster-whisper ASR backend. Returns words + timestamps."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..paths import models_dir

log = logging.getLogger(__name__)


@dataclass
class WordToken:
    text: str
    start_s: float
    end_s: float
    confidence: float


class FasterWhisperASR:
    def __init__(self, model_size: str = "small", compute_type: str | None = None):
        self.model_size = model_size
        self.compute_type = compute_type
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        # On Apple Silicon int8 is fastest for CPU; on Linux/x86 also fine.
        compute = self.compute_type or "int8"
        cache = models_dir() / "faster-whisper"
        cache.mkdir(parents=True, exist_ok=True)
        log.info("loading faster-whisper (%s, %s)", self.model_size, compute)
        self._model = WhisperModel(
            self.model_size,
            device="auto",
            compute_type=compute,
            download_root=str(cache),
        )

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000, language: str | None = "en"
    ) -> list[WordToken]:
        self.load()
        assert self._model is not None
        # faster-whisper accepts numpy directly
        segments, _ = self._model.transcribe(
            audio.astype(np.float32, copy=False),
            language=language,
            beam_size=1,
            word_timestamps=True,
            vad_filter=False,
        )
        out: list[WordToken] = []
        for seg in segments:
            if seg.words is None:
                continue
            for w in seg.words:
                out.append(
                    WordToken(
                        text=w.word.strip(),
                        start_s=float(w.start or 0.0),
                        end_s=float(w.end or 0.0),
                        confidence=float(getattr(w, "probability", 1.0) or 1.0),
                    )
                )
        return out


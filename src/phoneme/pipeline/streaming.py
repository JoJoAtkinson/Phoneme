"""Philosophy A — live streaming phonemes during speech.

Rolling 480 ms window fed to Wav2Vec2 every 160 ms. Only phonemes whose end
is sufficiently old within the window are emitted, and duplicates against the
previous emit are filtered. VAD still runs in parallel to detect utterance
boundaries; on utterance end, we clear the phoneme buffer.
"""

from __future__ import annotations

import logging

import numpy as np

from ..models.phoneme import Wav2Vec2PhonemeRecognizer
from .base import Pipeline
from .events import PartialPhonemes, UtteranceEnded

log = logging.getLogger(__name__)


class StreamingPipeline(Pipeline):
    def __init__(self, settings, emit):
        super().__init__(settings, emit)
        self._phonemes = Wav2Vec2PhonemeRecognizer(
            model_key="phoneme/wav2vec2-espeak",
            device="mps" if settings.prefer_mps else None,
        )

    def _wants_streaming(self) -> bool:
        return True

    def warmup(self) -> None:
        self._phonemes.load()

    def _on_stream_window(self, window: np.ndarray) -> None:
        try:
            new_tokens = self._phonemes.stream_step(
                window,
                sample_rate=self.settings.sample_rate,
                min_confidence=self.settings.phoneme_min_confidence,
            )
        except Exception as e:  # model hot-path shouldn't kill the thread
            log.exception("phoneme stream step failed: %s", e)
            return
        if new_tokens:
            self.emit(PartialPhonemes(tokens=new_tokens))

    def _on_utterance(self, audio: np.ndarray) -> None:
        self._phonemes.reset()
        # Still emit the full utterance so downstream can run ASR if wanted.
        self.emit(UtteranceEnded(audio=audio, sample_rate=self.settings.sample_rate))

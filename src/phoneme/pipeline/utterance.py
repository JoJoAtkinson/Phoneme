"""Philosophy B — utterance-at-a-time.

Wait for VAD to fire end-of-utterance, then run ASR + phoneme on the full
clip. Simpler and more accurate because the models see full context, but no
live feedback during speech.
"""

from __future__ import annotations

import logging

import numpy as np

from ..models.asr_whisper import FasterWhisperASR
from ..models.phoneme import Wav2Vec2PhonemeRecognizer
from .base import Pipeline
from .events import Transcription

log = logging.getLogger(__name__)


class UtterancePipeline(Pipeline):
    def __init__(self, settings, emit):
        super().__init__(settings, emit)
        self._phonemes = Wav2Vec2PhonemeRecognizer(
            device="mps" if settings.prefer_mps else None,
        )
        self._asr = FasterWhisperASR(model_size=settings.whisper_model_size)

    def warmup(self) -> None:
        self._phonemes.load()
        self._asr.load()

    def _on_utterance(self, audio: np.ndarray) -> None:
        sr = self.settings.sample_rate
        try:
            words = self._asr.transcribe(audio, sr)
            phs = self._phonemes.transcribe(
                audio, sr, min_confidence=self.settings.phoneme_min_confidence
            )
        except Exception as e:
            log.exception("utterance processing failed: %s", e)
            return

        self.emit(
            Transcription(
                text=" ".join(w.text for w in words),
                words=words,
                phonemes=phs,
                phoneme_groups=_group_phonemes_by_words(phs, words),
            )
        )


def _group_phonemes_by_words(phonemes, words) -> list[list[int]]:
    """Simple time-overlap alignment of phoneme tokens to word intervals."""
    groups: list[list[int]] = [[] for _ in words]
    if not words:
        return groups
    for i, ph in enumerate(phonemes):
        mid = (ph.start_s + ph.end_s) / 2
        best = 0
        best_overlap = -1.0
        for j, w in enumerate(words):
            if w.end_s <= w.start_s:
                continue
            overlap = max(0.0, min(w.end_s, ph.end_s) - max(w.start_s, ph.start_s))
            # bias toward word containing midpoint
            if w.start_s <= mid <= w.end_s:
                overlap += 0.05
            if overlap > best_overlap:
                best_overlap = overlap
                best = j
        groups[best].append(i)
    return groups

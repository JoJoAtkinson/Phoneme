"""Phoneme-compression mode.

User says phonemes one at a time with silence between (e.g. /k/ .. /æ/ .. /t/).
Each VAD-gated chunk is run through Wav2Vec2 and we append the most confident
phoneme(s) to a running buffer. On a longer pause (≥ 1.5 s) or on hold-to-talk
release, we "compress" the buffer by looking it up in CMU dict and emitting
candidate words.

This is separate from the "did I pronounce the whole word correctly" flow —
it's a building-blocks mode for learners who assemble words one sound at a
time.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from ..data.reverse_phoneme import lookup
from ..models.phoneme import Wav2Vec2PhonemeRecognizer
from ..runtime import torch_device
from .base import Pipeline
from .events import CompressedWord, PartialPhonemes

log = logging.getLogger(__name__)


class CompressPipeline(Pipeline):
    def __init__(self, settings, emit):
        super().__init__(settings, emit)
        self._phonemes = Wav2Vec2PhonemeRecognizer(
            model_key=settings.phoneme_model.catalog_key(),
            device=torch_device() if settings.prefer_mps else "cpu",
        )
        self._buffer: list[str] = []
        self._lock = threading.Lock()
        self._word_gap_s = 1.5  # longer pause ⇒ compress & emit word
        self._last_end_ts = 0.0

    def warmup(self) -> None:
        self._phonemes.load()

    def _on_utterance(self, audio: np.ndarray) -> None:
        import time as _t

        sr = self.settings.sample_rate
        phs = self._phonemes.transcribe(
            audio, sr, min_confidence=self.settings.phoneme_min_confidence
        )
        if not phs:
            return

        now = _t.monotonic()
        gap = now - self._last_end_ts if self._last_end_ts else 0.0
        self._last_end_ts = now

        with self._lock:
            if gap >= self._word_gap_s and self._buffer:
                candidates = lookup(self._buffer)
                self.emit(CompressedWord(phonemes=list(self._buffer), candidates=candidates))
                self._buffer = []
            # For compression mode we typically expect ONE phoneme per
            # utterance-gated burst, but if user said several we take the
            # most-confident-unique ones.
            taken = _pick_isolated(phs)
            self._buffer.extend(t.ipa for t in taken)
            self.emit(PartialPhonemes(tokens=taken))

    def finalize_word(self) -> None:
        """User explicitly ended a word (e.g. released hold-to-talk long press)."""
        with self._lock:
            if not self._buffer:
                return
            candidates = lookup(self._buffer)
            self.emit(CompressedWord(phonemes=list(self._buffer), candidates=candidates))
            self._buffer = []


def _pick_isolated(tokens) -> list:
    """Keep at most 3 most-confident, and drop immediate duplicates."""
    if not tokens:
        return []
    # Already merged in transcribe(), just cap length.
    sorted_by_conf = sorted(tokens, key=lambda t: -t.confidence)
    return sorted_by_conf[:3]

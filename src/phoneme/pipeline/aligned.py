"""Philosophy C (hybrid A+C) — streaming phonemes during speech, aligned word
layout after the pause.

Best of both worlds: user sees phonemes appear in real time (streaming
Wav2Vec2), and on utterance end we run full-context Whisper for accurate words
and re-run Wav2Vec2 on the complete clip for clean phoneme timestamps, then
group the phonemes under their words.
"""

from __future__ import annotations

import logging

import numpy as np

from ..models.asr import build_asr
from ..models.phoneme import Wav2Vec2PhonemeRecognizer
from ..runtime import torch_device
from .base import Pipeline
from .events import PartialPhonemes, Transcription
from .utterance import _group_phonemes_by_words

log = logging.getLogger(__name__)


class AlignedPipeline(Pipeline):
    def __init__(self, settings, emit):
        super().__init__(settings, emit)
        self._phonemes = Wav2Vec2PhonemeRecognizer(
            model_key=settings.phoneme_model.catalog_key(),
            device=torch_device() if settings.prefer_mps else "cpu",
        )
        self._asr = build_asr(settings)

    def _wants_streaming(self) -> bool:
        # Accuracy-over-latency: don't run phoneme inference on rolling
        # windows during the hold. Just wait for VAD/force_endpoint to
        # flush the full utterance, then do one clean phoneme + ASR pass.
        return False

    def warmup(self) -> None:
        self._phonemes.load()
        self._asr.load()

    def _on_stream_window(self, window: np.ndarray) -> None:
        try:
            new_tokens = self._phonemes.stream_step(
                window,
                sample_rate=self.settings.sample_rate,
                min_confidence=self.settings.phoneme_min_confidence,
            )
        except Exception as e:
            log.exception("stream step failed: %s", e)
            return
        if new_tokens:
            self.emit(PartialPhonemes(tokens=new_tokens))

    def _on_utterance(self, audio: np.ndarray) -> None:
        import time as _t

        sr = self.settings.sample_rate
        dur_ms = len(audio) * 1000 // sr
        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2) + 1e-12))
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        log.info(
            "[ALIGNED] _on_utterance start audio=%dms rms=%.4f peak=%.3f",
            dur_ms, rms, peak,
        )
        self._phonemes.reset()
        try:
            t0 = _t.perf_counter()
            words = self._asr.transcribe(audio, sr)
            t1 = _t.perf_counter()
            phs = self._phonemes.transcribe(
                audio, sr, min_confidence=self.settings.phoneme_min_confidence
            )
            t2 = _t.perf_counter()
            log.info(
                "[ALIGNED] inference asr=%dms phoneme=%dms words=%d phones=%d",
                int((t1 - t0) * 1000), int((t2 - t1) * 1000), len(words), len(phs),
            )
        except Exception as e:
            log.exception("aligned post-processing failed: %s", e)
            return

        text = " ".join(w.text for w in words)
        log.info("[ALIGNED] emit Transcription text=%r phones=%d", text, len(phs))
        self.emit(
            Transcription(
                text=text,
                words=words,
                phonemes=phs,
                phoneme_groups=_group_phonemes_by_words(phs, words),
            )
        )

"""Pipeline base — owns the audio capture, VAD, and model workers.

Subclasses override ``_on_utterance`` to define what happens when VAD fires.
For the streaming mode we also consume the ring buffer directly on a worker
thread to emit partial phonemes while the user is still speaking.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ..audio.capture import MicCapture
from ..audio.vad import StreamingVAD
from ..config import Settings

log = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, settings: Settings, emit: Callable[[Any], None]):
        self.settings = settings
        self.emit = emit
        self.mic: MicCapture | None = None
        self.vad: StreamingVAD | None = None
        self._stop = threading.Event()
        self._drain_thread: threading.Thread | None = None
        self._stream_thread: threading.Thread | None = None
        self._force_endpoint = threading.Event()

    # ---- lifecycle -----------------------------------------------------

    def start(self) -> None:
        s = self.settings
        self.vad = StreamingVAD(
            sample_rate=s.sample_rate,
            threshold=s.vad_threshold,
            min_silence_ms=s.vad_min_silence_ms,
            min_speech_ms=s.vad_min_speech_ms,
            on_speech_start=self._fire_utterance_started,
            on_speech_end=self._on_utterance,
            on_prob=self._on_vad_prob,
        )
        self.mic = MicCapture(sample_rate=s.sample_rate, block_ms=32)
        self.mic.start()
        self._stop.clear()
        self._drain_thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._drain_thread.start()
        if self._wants_streaming():
            self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
            self._stream_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.mic is not None:
            self.mic.stop()
        if self._drain_thread is not None:
            self._drain_thread.join(timeout=1.0)
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=1.0)

    # ---- hooks ---------------------------------------------------------

    def _wants_streaming(self) -> bool:
        return False

    def _on_utterance(self, audio: np.ndarray) -> None:
        """Subclasses override."""
        from .events import UtteranceEnded

        self.emit(UtteranceEnded(audio=audio, sample_rate=self.settings.sample_rate))

    def _on_stream_window(self, window: np.ndarray) -> None:
        """Subclasses override for streaming phoneme emission."""

    # ---- hold-to-talk helpers -----------------------------------------

    def force_endpoint_now(self) -> None:
        """Called when the user releases hold-to-talk: end utterance immediately."""
        if self.vad is not None:
            self.vad.force_end()

    # ---- internals -----------------------------------------------------

    def _fire_utterance_started(self) -> None:
        from .events import UtteranceStarted

        self.emit(UtteranceStarted())

    def _on_vad_prob(self, prob: float) -> None:
        # Hook for UI meter; override if needed.
        pass

    def _drain_loop(self) -> None:
        """Copy audio from mic ring buffer into the VAD in chunks."""
        assert self.mic is not None and self.vad is not None
        drained = 0
        while not self._stop.is_set():
            total = self.mic.buffer.total_written
            if total <= drained:
                time.sleep(0.005)
                continue
            new_n = min(total - drained, StreamingVAD.CHUNK * 8)
            window = self.mic.buffer.read_latest(int(total - drained))[:new_n]
            self.vad.feed(window)
            drained += new_n

    def _stream_loop(self) -> None:
        """Slide a rolling window for streaming phoneme recognition."""
        assert self.mic is not None
        sr = self.settings.sample_rate
        win = int(sr * self.settings.phoneme_window_ms / 1000)
        hop = int(sr * self.settings.phoneme_hop_ms / 1000)
        last_read = 0
        while not self._stop.is_set():
            total = self.mic.buffer.total_written
            if total - last_read < hop:
                time.sleep(0.01)
                continue
            window = self.mic.buffer.read_latest(win)
            last_read = total
            self._on_stream_window(window)

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
        # True when the drain/stream loops should skip VAD and model inference.
        # In hold-to-talk mode we default to paused and only unpause while the
        # user is holding the key — otherwise streaming inference floods the
        # Qt event queue and the main thread stops processing key-release
        # events, which manifests as the spacebar appearing stuck.
        self._paused = threading.Event()
        if settings.hold_to_talk:
            self._paused.set()

    def set_paused(self, paused: bool) -> None:
        # Don't touch VAD state from the caller thread — that can race with
        # an in-flight force_endpoint on a worker. The drain/stream loops
        # advance their own cursors while paused so unpausing is clean.
        was = self._paused.is_set()
        if paused:
            self._paused.set()
        else:
            self._paused.clear()
        if was != paused:
            log.info("[PIPE] set_paused %s -> %s", was, paused)

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
        """Called when the user releases hold-to-talk: end utterance immediately.

        If VAD never entered the speech state during the hold (e.g. user spoke
        quietly or briefly), fall back to the last ~3 s of mic audio so the
        user still gets feedback for every hold regardless of VAD opinion."""
        log.info("[PIPE] force_endpoint_now")
        if self.vad is None:
            return
        if self.vad._in_speech:
            self.vad.force_end()
            return
        # VAD never triggered — fall back to a slice of recent audio.
        if self.mic is None:
            return
        recent = self.mic.buffer.read_latest(int(self.settings.sample_rate * 3.0))
        # Strip leading silence using raw RMS above a low floor.
        thresh = 0.005
        rms_window = 320  # 20 ms at 16 kHz
        for i in range(0, len(recent) - rms_window, rms_window):
            if float(np.sqrt(np.mean(recent[i : i + rms_window] ** 2))) > thresh:
                recent = recent[i:]
                break
        if len(recent) < int(self.settings.sample_rate * 0.1):
            log.info("[PIPE] force_endpoint_now: no audible audio in fallback window")
            return
        log.info(
            "[PIPE] force_endpoint_now fallback: VAD didn't trigger — using %dms of mic buffer",
            len(recent) * 1000 // self.settings.sample_rate,
        )
        self._on_utterance(recent)

    # ---- internals -----------------------------------------------------

    def _fire_utterance_started(self) -> None:
        from .events import UtteranceStarted

        log.info("[PIPE] utterance_started")
        self.emit(UtteranceStarted())

    def _on_vad_prob(self, prob: float) -> None:
        # Hook for UI meter; override if needed.
        pass

    def _drain_loop(self) -> None:
        """Copy audio from mic ring buffer into the VAD in chunks."""
        assert self.mic is not None and self.vad is not None
        drained = 0
        while not self._stop.is_set():
            if self._paused.is_set():
                # Keep the drain pointer current so we don't replay stale
                # audio into the VAD when unpausing mid-stream.
                drained = self.mic.buffer.total_written
                time.sleep(0.02)
                continue
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
            if self._paused.is_set():
                last_read = self.mic.buffer.total_written
                time.sleep(0.02)
                continue
            total = self.mic.buffer.total_written
            if total - last_read < hop:
                time.sleep(0.01)
                continue
            window = self.mic.buffer.read_latest(win)
            last_read = total
            self._on_stream_window(window)

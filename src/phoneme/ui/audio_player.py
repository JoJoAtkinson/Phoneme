"""Non-blocking audio playback via sounddevice."""

from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger(__name__)


class AudioPlayer:
    def __init__(self):
        self._lock = threading.Lock()
        self._current = None

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        import sounddevice as sd

        def _run():
            try:
                sd.stop()
                sd.play(samples.astype(np.float32, copy=False), samplerate=sample_rate)
            except Exception as e:
                log.warning("playback failed: %s", e)

        threading.Thread(target=_run, daemon=True).start()

    def stop(self) -> None:
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass

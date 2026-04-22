"""Microphone capture via sounddevice → RingBuffer.

PortAudio gives us a callback thread; we push samples into a lock-free-ish ring
buffer that pipeline workers pull from.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import numpy as np

from .ring_buffer import RingBuffer

log = logging.getLogger(__name__)


class MicCapture:
    def __init__(
        self,
        sample_rate: int = 16000,
        block_ms: int = 20,
        input_device: int | None = None,
        buffer_seconds: float = 30.0,
        on_level: Callable[[float], None] | None = None,
    ):
        self.sample_rate = sample_rate
        self.block_size = int(sample_rate * block_ms / 1000)
        self.input_device = input_device
        self.buffer = RingBuffer(int(sample_rate * buffer_seconds))
        self._stream = None
        self._stopped = threading.Event()
        self._on_level = on_level

    def _callback(self, indata, frames, time, status):  # pragma: no cover - audio thread
        if status:
            log.debug("sounddevice status: %s", status)
        mono = indata[:, 0] if indata.ndim > 1 else indata
        mono = mono.astype(np.float32, copy=False)
        self.buffer.write(mono)
        if self._on_level is not None:
            rms = float(np.sqrt(np.mean(mono * mono) + 1e-12))
            self._on_level(rms)

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            callback=self._callback,
            device=self.input_device,
        )
        self._stream.start()
        log.info("mic capture started @ %d Hz, block=%d", self.sample_rate, self.block_size)

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        self._stopped.set()

    def __enter__(self) -> "MicCapture":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

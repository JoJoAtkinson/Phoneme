"""Thread-safe float32 mono ring buffer sized in samples."""

from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    def __init__(self, capacity: int):
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._cap = capacity
        self._write = 0
        self._size = 0
        self._total_written = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def write(self, samples: np.ndarray) -> None:
        samples = samples.reshape(-1).astype(np.float32, copy=False)
        n = len(samples)
        with self._cond:
            if n >= self._cap:
                self._buf[:] = samples[-self._cap :]
                self._write = 0
                self._size = self._cap
            else:
                end = self._write + n
                if end <= self._cap:
                    self._buf[self._write : end] = samples
                else:
                    first = self._cap - self._write
                    self._buf[self._write :] = samples[:first]
                    self._buf[: n - first] = samples[first:]
                self._write = end % self._cap
                self._size = min(self._cap, self._size + n)
            self._total_written += n
            self._cond.notify_all()

    def read_latest(self, n: int) -> np.ndarray:
        """Return the most recent n samples (zero-padded at front if not enough)."""
        with self._lock:
            available = min(self._size, n)
            out = np.zeros(n, dtype=np.float32)
            if available == 0:
                return out
            start = (self._write - available) % self._cap
            end = start + available
            if end <= self._cap:
                out[-available:] = self._buf[start:end]
            else:
                first = self._cap - start
                out[-available : -available + first] = self._buf[start:]
                out[-available + first :] = self._buf[: available - first]
            return out

    @property
    def total_written(self) -> int:
        with self._lock:
            return self._total_written

    def wait_for(self, target_total: int, timeout: float | None = None) -> bool:
        """Block until total_written >= target_total."""
        with self._cond:
            return self._cond.wait_for(
                lambda: self._total_written >= target_total, timeout=timeout
            )

"""Streaming magnitude spectrogram for the live display.

We compute a short-time Fourier transform over a rolling audio window, then
fold the linear frequency bins into a log-mel scale so vowel formants (the
pedagogically useful part) pop out. The output is [num_frames, num_mels]
with values in dB, ready to be color-mapped to an image.

Design notes:
  * Hann window, 512-sample FFT at 16 kHz → 32 ms frames, 16 Hz resolution.
  * 8 ms hop → 125 frames/sec, so a 4-second strip is 500 columns — a good
    fit for a ~500 px wide widget.
  * We cap the top frequency at 5 kHz on the mel axis. Speech formants F1–F3
    live well below that; higher bins are mostly fricative noise and just
    add visual clutter for a learner.
  * Numpy-only. No torch, no torchaudio, no scipy required at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    num_mels: int,
    n_fft: int,
    sample_rate: int,
    f_min: float = 50.0,
    f_max: float | None = None,
) -> np.ndarray:
    """Classic triangular mel filterbank → shape [num_mels, n_fft//2 + 1]."""
    if f_max is None:
        f_max = sample_rate / 2
    mel_edges = np.linspace(_hz_to_mel(np.array(f_min)), _hz_to_mel(np.array(f_max)), num_mels + 2)
    hz_edges = _mel_to_hz(mel_edges)
    bin_edges = np.floor((n_fft + 1) * hz_edges / sample_rate).astype(int)
    bin_edges = np.clip(bin_edges, 0, n_fft // 2)

    fb = np.zeros((num_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(num_mels):
        lo, mid, hi = bin_edges[m], bin_edges[m + 1], bin_edges[m + 2]
        if mid > lo:
            fb[m, lo:mid] = np.linspace(0, 1, mid - lo, endpoint=False)
        if hi > mid:
            fb[m, mid:hi] = np.linspace(1, 0, hi - mid, endpoint=False)
    # Normalize so each filter sums to 1 (Slaney style).
    sums = fb.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return fb / sums


@dataclass
class SpectrogramConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    hop: int = 128  # 8 ms
    num_mels: int = 64
    f_min: float = 50.0
    f_max: float = 5000.0  # cap for speech-formant display
    db_floor: float = -80.0
    db_ceiling: float = 0.0


class StreamingSpectrogram:
    """Maintains a ring of column samples for a rolling display.

    Call ``process(samples)`` with new audio; new spectrogram frames are
    computed, normalized to [0, 1], and written to the ring. Call
    ``image_view()`` to read the current state as a [num_mels, num_cols]
    float32 array (oldest column first), ready to be colormapped.
    """

    def __init__(self, num_cols: int = 512, config: SpectrogramConfig | None = None):
        self.cfg = config or SpectrogramConfig()
        self.num_cols = num_cols
        self._filterbank = mel_filterbank(
            self.cfg.num_mels, self.cfg.n_fft, self.cfg.sample_rate, self.cfg.f_min, self.cfg.f_max
        )
        self._window = np.hanning(self.cfg.n_fft).astype(np.float32)
        # ring buffer of mel-log columns, [num_mels, num_cols]
        self._img = np.zeros((self.cfg.num_mels, num_cols), dtype=np.float32)
        self._write_col = 0
        # pending audio that didn't fit into a whole frame yet
        self._pending = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self._img[:] = 0.0
        self._write_col = 0
        self._pending = np.zeros(0, dtype=np.float32)

    def process(self, samples: np.ndarray) -> int:
        """Append samples; return number of new spectrogram columns written."""
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        self._pending = np.concatenate([self._pending, samples])
        n_new = 0
        while len(self._pending) >= self.cfg.n_fft:
            frame = self._pending[: self.cfg.n_fft] * self._window
            # slide by hop
            self._pending = self._pending[self.cfg.hop :]
            spec = np.abs(np.fft.rfft(frame)).astype(np.float32)  # [n_fft//2 + 1]
            mel = self._filterbank @ spec  # [num_mels]
            mel_db = 20.0 * np.log10(mel + 1e-8)
            mel_db = np.clip(mel_db, self.cfg.db_floor, self.cfg.db_ceiling)
            norm = (mel_db - self.cfg.db_floor) / (self.cfg.db_ceiling - self.cfg.db_floor)
            self._img[:, self._write_col] = norm
            self._write_col = (self._write_col + 1) % self.num_cols
            n_new += 1
        return n_new

    def image_view(self) -> np.ndarray:
        """Return a contiguous [num_mels, num_cols] array, oldest column first."""
        if self._write_col == 0:
            return self._img
        return np.concatenate(
            [self._img[:, self._write_col :], self._img[:, : self._write_col]], axis=1
        )

    @property
    def num_mels(self) -> int:
        return self.cfg.num_mels

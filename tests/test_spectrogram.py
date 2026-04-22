"""Unit tests for the spectrogram pipeline.

We verify correctness by feeding synthetic sines and checking that the
brightest mel bin lands where we expect — no audio hardware needed.
"""

from __future__ import annotations

import numpy as np

from phoneme.audio.spectrogram import (
    SpectrogramConfig,
    StreamingSpectrogram,
    mel_filterbank,
)


def test_mel_filterbank_shape():
    fb = mel_filterbank(num_mels=64, n_fft=512, sample_rate=16000)
    assert fb.shape == (64, 257)
    # Each filter should sum to 1 (Slaney normalization)
    sums = fb.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6)


def _make_sine(freq_hz: float, seconds: float, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def test_single_sine_peaks_at_expected_mel_bin():
    cfg = SpectrogramConfig(num_mels=64, f_min=50.0, f_max=5000.0)
    spec = StreamingSpectrogram(num_cols=256, config=cfg)
    # Pure 1 kHz tone for 500 ms
    spec.process(_make_sine(1000.0, 0.5))
    img = spec.image_view()
    # After the last column written, find the mel bin with max average power
    mean_per_bin = img.mean(axis=1)
    peak_bin = int(np.argmax(mean_per_bin))

    # Expected mel index for 1 kHz with f_min=50, f_max=5000, num_mels=64
    from phoneme.audio.spectrogram import _hz_to_mel

    mel_lo = float(_hz_to_mel(np.array(50.0)))
    mel_hi = float(_hz_to_mel(np.array(5000.0)))
    mel_1k = float(_hz_to_mel(np.array(1000.0)))
    expected = int(round(64 * (mel_1k - mel_lo) / (mel_hi - mel_lo)))

    assert abs(peak_bin - expected) <= 2, f"peak at bin {peak_bin}, expected ~{expected}"


def test_two_sines_show_two_bands():
    # 500 Hz + 2000 Hz simultaneously — mimics a two-formant vowel
    cfg = SpectrogramConfig(num_mels=64)
    spec = StreamingSpectrogram(num_cols=256, config=cfg)
    audio = _make_sine(500.0, 0.5) + _make_sine(2000.0, 0.5)
    spec.process(audio.astype(np.float32))
    img = spec.image_view()
    mean_per_bin = img.mean(axis=1)
    # Identify the two strongest peaks (separated by at least 5 bins).
    top = np.argsort(mean_per_bin)[::-1][:10]
    top_sorted = sorted(int(b) for b in top[:3])
    # Should span a wide range (not clustered to one formant)
    assert top_sorted[-1] - top_sorted[0] >= 5


def test_silence_produces_near_floor_image():
    cfg = SpectrogramConfig()
    spec = StreamingSpectrogram(num_cols=64, config=cfg)
    spec.process(np.zeros(16000, dtype=np.float32))
    img = spec.image_view()
    # With db_floor normalization, silence should map near 0.
    assert float(img.mean()) < 0.05


def test_process_returns_new_column_count():
    cfg = SpectrogramConfig(n_fft=512, hop=128)
    spec = StreamingSpectrogram(num_cols=128, config=cfg)
    # 1024 samples should yield about (1024-512)/128 + 1 = 5 frames
    n = spec.process(np.zeros(1024, dtype=np.float32))
    assert n in (4, 5)


def test_ring_wraps_after_num_cols_frames():
    cfg = SpectrogramConfig(n_fft=256, hop=128)
    spec = StreamingSpectrogram(num_cols=8, config=cfg)
    # Produce enough frames to wrap the ring multiple times
    total = spec.process(np.zeros(256 + 128 * 20, dtype=np.float32))
    assert total > 8
    img = spec.image_view()
    assert img.shape[1] == 8

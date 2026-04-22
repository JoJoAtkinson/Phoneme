"""RingBuffer unit tests — no audio hardware required."""

from __future__ import annotations

import numpy as np

from phoneme.audio.ring_buffer import RingBuffer


def test_write_and_read_within_capacity():
    rb = RingBuffer(10)
    rb.write(np.arange(4, dtype=np.float32))
    latest = rb.read_latest(4)
    assert np.array_equal(latest, np.arange(4, dtype=np.float32))


def test_wrap_around():
    rb = RingBuffer(5)
    rb.write(np.array([1, 2, 3, 4, 5], dtype=np.float32))
    rb.write(np.array([6, 7], dtype=np.float32))
    # buffer now holds 3,4,5,6,7 as "latest"
    latest = rb.read_latest(5)
    assert np.array_equal(latest, np.array([3, 4, 5, 6, 7], dtype=np.float32))


def test_over_capacity_write():
    rb = RingBuffer(3)
    rb.write(np.arange(10, dtype=np.float32))
    latest = rb.read_latest(3)
    assert np.array_equal(latest, np.array([7, 8, 9], dtype=np.float32))


def test_read_latest_pads_front_with_zero():
    rb = RingBuffer(10)
    rb.write(np.array([1, 2], dtype=np.float32))
    latest = rb.read_latest(5)
    assert latest.shape == (5,)
    assert np.array_equal(latest[-2:], np.array([1, 2], dtype=np.float32))
    assert np.all(latest[:-2] == 0)


def test_total_written_monotonic():
    rb = RingBuffer(10)
    assert rb.total_written == 0
    rb.write(np.zeros(3, dtype=np.float32))
    rb.write(np.zeros(100, dtype=np.float32))
    assert rb.total_written == 103

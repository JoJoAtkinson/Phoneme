"""Rolling mel-spectrogram widget.

Polls a ``RingBuffer`` via a QTimer at ~30 fps, advances a
``StreamingSpectrogram``, and paints the resulting log-mel image.

Usage:
    w = SpectrogramWidget()
    w.set_source(lambda n: ring_buffer.read_latest(n))
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QWidget

from ..audio.spectrogram import SpectrogramConfig, StreamingSpectrogram
from .colormap import apply_colormap

log = logging.getLogger(__name__)


class SpectrogramWidget(QWidget):
    def __init__(self, parent: QWidget | None = None, num_cols: int = 512):
        super().__init__(parent)
        self._cfg = SpectrogramConfig()
        self._spec = StreamingSpectrogram(num_cols=num_cols, config=self._cfg)
        self._source: Callable[[int], np.ndarray] | None = None
        self._last_drained_total = 0
        self._get_total: Callable[[], int] | None = None

        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30 fps
        self._timer.timeout.connect(self._tick)

        self.setMinimumHeight(80)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setStyleSheet("background: #0b0e13; border: 1px solid #232a36; border-radius: 10px;")

    # ---- wiring --------------------------------------------------------

    def set_source(
        self,
        read_latest: Callable[[int], np.ndarray],
        total_written: Callable[[], int],
    ) -> None:
        """Hook up the audio source. ``read_latest(n)`` returns the most
        recent n samples (float32 @ 16 kHz); ``total_written()`` returns
        the running total sample count so we can tell how much is new."""
        self._source = read_latest
        self._get_total = total_written
        self._last_drained_total = total_written()
        if not self._timer.isActive():
            self._timer.start()

    def clear_source(self) -> None:
        self._timer.stop()
        self._source = None
        self._get_total = None
        self._spec.reset()
        self.update()

    def reset(self) -> None:
        self._spec.reset()
        self.update()

    # ---- per-frame update ---------------------------------------------

    def _tick(self) -> None:
        if self._source is None or self._get_total is None:
            return
        total = self._get_total()
        new_samples = total - self._last_drained_total
        if new_samples <= 0:
            return
        # Cap how much we pull per tick so we don't blow up on hiccups.
        take = min(new_samples, self._cfg.sample_rate)  # up to 1 s
        samples = self._source(int(take))
        # read_latest pads with zeros at the front if buffer is short; that
        # pre-recording silence will just be quiet columns, which is fine.
        self._last_drained_total = total
        n_cols = self._spec.process(samples)
        if n_cols > 0:
            self.update()

    # ---- painting ------------------------------------------------------

    def paintEvent(self, _event) -> None:
        img = self._spec.image_view()  # [mels, cols]
        if img.size == 0:
            return

        # Flip vertically so low frequencies are at the bottom (standard).
        flipped = np.ascontiguousarray(img[::-1, :])
        rgba = apply_colormap(flipped)  # [mels, cols, 4]
        h, w, _ = rgba.shape
        qimg = QImage(rgba.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888).copy()

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        # Scale to widget, preserving nothing (fill).
        target = self.rect().adjusted(1, 1, -1, -1)
        p.drawImage(target, qimg)

        # Draw frequency tick labels (Hz) on the right edge.
        from PySide6.QtGui import QColor, QFont

        p.setPen(QColor("#6a7687"))
        f = QFont()
        f.setPointSize(8)
        p.setFont(f)
        # 4 tick marks: f_min, 25%, 50%, 100% of mel axis
        ticks = [
            (self._cfg.f_min, 1.0),
            (self._cfg.f_min + 0.25 * (self._cfg.f_max - self._cfg.f_min), 0.75),
            (self._cfg.f_min + 0.5 * (self._cfg.f_max - self._cfg.f_min), 0.5),
            (self._cfg.f_max, 0.0),
        ]
        for hz, y_frac in ticks:
            y = int(target.top() + y_frac * target.height())
            p.drawText(target.right() - 40, y + 4, f"{hz/1000:.1f} kHz")
        p.end()

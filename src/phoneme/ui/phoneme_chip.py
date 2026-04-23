"""Colored pill widget for a single phoneme."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..data.ipa import articulation_hint, color_for, is_vowel, simple_label


class PhonemeChip(QWidget):
    def __init__(
        self,
        ipa: str,
        duration_s: float = 0.0,
        confidence: float = 1.0,
        word_below: str | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.ipa = ipa
        self.duration_s = duration_s
        self.confidence = max(0.0, min(1.0, confidence))
        self.word_below = word_below
        # Fixed chip size — without this, a single chip inside a stretchy
        # parent layout balloons to fill all available space.
        self.setFixedSize(72, 96)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setToolTip(self._tooltip())

    def _tooltip(self) -> str:
        parts = [f"/{self.ipa}/"]
        hint = articulation_hint(self.ipa)
        if hint:
            parts.append(hint)
        if self.duration_s:
            parts.append(f"{self.duration_s * 1000:.0f} ms")
        parts.append(f"conf {self.confidence:.2f}")
        return "  ·  ".join(parts)

    def sizeHint(self):
        from PySide6.QtCore import QSize

        return QSize(72, 96)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        r = self.rect().adjusted(4, 4, -4, -4)
        # Split: top 2/3 = chip, bottom 1/3 = word (if any)
        chip_h = int(r.height() * (0.75 if self.word_below else 1.0))
        chip = r.adjusted(0, 0, 0, -(r.height() - chip_h))

        base = QColor(color_for(self.ipa))
        # Confidence → alpha on fill
        fill = QColor(base)
        fill.setAlphaF(0.25 + 0.55 * self.confidence)
        stroke = QColor(base)
        stroke.setAlphaF(0.4 + 0.6 * self.confidence)

        path = QPainterPath()
        radius = 14.0
        path.addRoundedRect(chip, radius, radius)
        p.fillPath(path, fill)
        pen = QPen(stroke, 1.5)
        p.setPen(pen)
        p.drawPath(path)

        # Elementary-school glyph (ā, ĕ, sh, th, ...). The raw IPA still
        # lives in the tooltip for anyone who wants it.
        p.setPen(QColor("#0e1116"))
        font = QFont()
        font.setPointSize(22 if not is_vowel(self.ipa) else 24)
        font.setBold(True)
        font.setFamily("Helvetica Neue")
        p.setFont(font)
        p.drawText(chip, Qt.AlignmentFlag.AlignCenter, simple_label(self.ipa))

        # Word below
        if self.word_below:
            word_rect = r.adjusted(0, chip_h, 0, 0)
            p.setPen(QColor("#cfd6e3"))
            f2 = QFont()
            f2.setPointSize(11)
            p.setFont(f2)
            p.drawText(word_rect, Qt.AlignmentFlag.AlignCenter, self.word_below)

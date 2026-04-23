"""The main display: big word on top, aligned phoneme chips underneath."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..models.phoneme import PhonemeToken
from .flow_layout import FlowLayout
from .phoneme_chip import PhonemeChip


class ChipStrip(QFrame):
    """Single-line flowing strip of phoneme chips (used for streaming mode)."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("ChipStrip")
        self._layout = FlowLayout(self, h_spacing=6, v_spacing=6)
        self.setMinimumHeight(110)

    def append(self, token: PhonemeToken) -> None:
        chip = PhonemeChip(
            ipa=token.ipa, duration_s=token.duration_s, confidence=token.confidence
        )
        self._layout.addWidget(chip)
        self.update()

    def clear(self) -> None:
        self._layout.clear()
        self.update()


class AlignedWordView(QWidget):
    """Row of word columns, each with the word on top and its phonemes beneath."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(12, 8, 12, 8)
        self._row.setSpacing(18)
        self._row.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._row.addStretch(1)

    def show_transcription(
        self, words, phonemes: list[PhonemeToken], groups: list[list[int]]
    ) -> None:
        # Clear
        while self._row.count():
            item = self._row.takeAt(0)
            if item is None:
                continue
            w = item.widget()
            if w:
                w.deleteLater()
        if not words:
            return

        for wi, word in enumerate(words):
            col = QWidget()
            col_layout = QVBoxLayout(col)
            col_layout.setContentsMargins(0, 0, 0, 0)
            col_layout.setSpacing(8)

            label = QLabel(word.text)
            label.setObjectName("WordLabel")
            label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            col_layout.addWidget(label)

            chips_row = QHBoxLayout()
            chips_row.setSpacing(4)
            chips_row.addStretch(1)
            indices = groups[wi] if wi < len(groups) else []
            for idx in indices:
                tok = phonemes[idx]
                chips_row.addWidget(
                    PhonemeChip(ipa=tok.ipa, duration_s=tok.duration_s, confidence=tok.confidence)
                )
            chips_row.addStretch(1)

            col_layout.addLayout(chips_row)
            self._row.addWidget(col)
        self._row.addStretch(1)

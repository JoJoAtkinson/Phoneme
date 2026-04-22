"""Phoneme main window.

Layout (top → bottom):
  1. Toolbar: pipeline mode, echo toggle, settings button
  2. Big word + live caption area
  3. Aligned word/phoneme view (shown on utterance end for aligned/utterance modes)
  4. Streaming phoneme chip strip (shown for streaming/aligned/compress modes)
  5. Footer: mic level, VAD indicator, hold-to-talk hint
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeyEvent
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..config import PipelineMode, Settings, TTSBackend
from ..models.tts_kokoro import KokoroTTS
from ..pipeline.events import (
    CompressedWord,
    EchoAudio,
    PartialPhonemes,
    Transcription,
    UtteranceEnded,
    UtteranceStarted,
)
from .audio_player import AudioPlayer
from .pipeline_runner import PipelineRunner
from .settings_dialog import SettingsDialog
from .spectrogram_widget import SpectrogramWidget
from .styles import APP_QSS
from .word_display import AlignedWordView, ChipStrip

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    hold_state_changed = Signal(bool)

    def __init__(self):
        super().__init__()
        self.settings = Settings.load()
        self.setWindowTitle("Phoneme")
        self.resize(1080, 620)
        self.setStyleSheet(APP_QSS)

        self.runner = PipelineRunner(self.settings)
        self.runner.event.connect(self._on_event)
        self.runner.pipeline_ready.connect(self._on_pipeline_ready)

        self.player = AudioPlayer()
        self._tts: KokoroTTS | None = None

        self._hold_active = False

        central = QWidget()
        central.setObjectName("Central")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        self._build_toolbar()

        # Word + live caption
        header = QWidget()
        header_l = QVBoxLayout(header)
        header_l.setContentsMargins(16, 8, 16, 0)
        header_l.setSpacing(2)
        self.word_label = QLabel("—")
        self.word_label.setObjectName("WordLabel")
        header_l.addWidget(self.word_label)

        self.live_caption = QLabel("Press and hold Space to speak")
        self.live_caption.setObjectName("LiveCaption")
        header_l.addWidget(self.live_caption)
        root.addWidget(header)

        # Aligned view (appears after utterance end)
        self.aligned_view = AlignedWordView()
        root.addWidget(self.aligned_view)

        # Streaming chip strip
        self.strip = ChipStrip()
        strip_wrap = QWidget()
        wrap_l = QHBoxLayout(strip_wrap)
        wrap_l.setContentsMargins(16, 0, 16, 8)
        wrap_l.addWidget(self.strip)
        root.addWidget(strip_wrap, 1)

        # Spectrogram strip (formants / vowel signatures)
        self.spectrogram = SpectrogramWidget()
        self.spectrogram.setVisible(self.settings.show_spectrogram)
        spec_wrap = QWidget()
        spec_l = QHBoxLayout(spec_wrap)
        spec_l.setContentsMargins(16, 0, 16, 16)
        spec_l.addWidget(self.spectrogram)
        root.addWidget(spec_wrap)

        # Footer
        self._build_statusbar()

        # Start pipeline after show
        QTimer.singleShot(50, self._kickoff)

    # ---- UI scaffolding -----------------------------------------------

    def _build_toolbar(self) -> None:
        tb = QToolBar("main")
        tb.setMovable(False)
        self.addToolBar(tb)

        mode_label = QLabel("  Mode:  ")
        tb.addWidget(mode_label)
        self.mode_select = QComboBox()
        for m in PipelineMode:
            self.mode_select.addItem(m.value, m)
        self.mode_select.setCurrentText(self.settings.pipeline_mode.value)
        self.mode_select.currentIndexChanged.connect(self._on_mode_changed)
        tb.addWidget(self.mode_select)

        tb.addSeparator()
        self.echo_btn = QPushButton("Echo: on")
        self.echo_btn.setCheckable(True)
        self.echo_btn.setChecked(self.settings.echo_after_utterance)
        self.echo_btn.toggled.connect(self._on_echo_toggled)
        tb.addWidget(self.echo_btn)

        self.hold_btn = QPushButton("Hold-to-talk: on")
        self.hold_btn.setCheckable(True)
        self.hold_btn.setChecked(self.settings.hold_to_talk)
        self.hold_btn.toggled.connect(self._on_hold_toggled)
        tb.addWidget(self.hold_btn)

        self.spec_btn = QPushButton("Spectrogram: on")
        self.spec_btn.setCheckable(True)
        self.spec_btn.setChecked(self.settings.show_spectrogram)
        self.spec_btn.toggled.connect(self._on_spectrogram_toggled)
        tb.addWidget(self.spec_btn)

        tb.addSeparator()
        clear_act = QAction("Clear", self)
        clear_act.triggered.connect(self._clear_displays)
        tb.addAction(clear_act)

        settings_act = QAction("Settings…", self)
        settings_act.triggered.connect(self._open_settings)
        tb.addAction(settings_act)

    def _build_statusbar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.hold_hint = QLabel("● idle")
        self.hold_hint.setObjectName("HoldHint")
        self.mic_label = QLabel("mic: —")
        self.mic_label.setObjectName("HoldHint")
        sb.addPermanentWidget(self.hold_hint)
        sb.addPermanentWidget(self.mic_label)

    # ---- pipeline control ---------------------------------------------

    def _kickoff(self) -> None:
        self.runner.start()

    def _restart_pipeline(self) -> None:
        self.spectrogram.clear_source()
        self.runner.settings = self.settings
        self.runner.start()
        self._clear_displays()

    def _on_pipeline_ready(self) -> None:
        src = self.runner.mic_source()
        if src is not None and self.settings.show_spectrogram:
            read_latest, total_written = src
            self.spectrogram.set_source(read_latest, total_written)

    def _on_mode_changed(self) -> None:
        new_mode = self.mode_select.currentData()
        if new_mode == self.settings.pipeline_mode:
            return
        self.settings.pipeline_mode = new_mode
        self.settings.save()
        self._restart_pipeline()

    def _on_echo_toggled(self, on: bool) -> None:
        self.settings.echo_after_utterance = on
        self.settings.save()
        self.echo_btn.setText(f"Echo: {'on' if on else 'off'}")

    def _on_hold_toggled(self, on: bool) -> None:
        self.settings.hold_to_talk = on
        self.settings.save()
        self.hold_btn.setText(f"Hold-to-talk: {'on' if on else 'off'}")

    def _on_spectrogram_toggled(self, on: bool) -> None:
        self.settings.show_spectrogram = on
        self.settings.save()
        self.spec_btn.setText(f"Spectrogram: {'on' if on else 'off'}")
        self.spectrogram.setVisible(on)
        if on:
            src = self.runner.mic_source()
            if src is not None:
                read_latest, total_written = src
                self.spectrogram.set_source(read_latest, total_written)
        else:
            self.spectrogram.clear_source()

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec():
            prev_mode = self.settings.pipeline_mode
            self.settings = dlg.apply_to(self.settings)
            self.settings.save()
            if self.settings.pipeline_mode != prev_mode:
                self.mode_select.setCurrentText(self.settings.pipeline_mode.value)
            self._restart_pipeline()

    def _clear_displays(self) -> None:
        self.strip.clear()
        self.aligned_view.show_transcription([], [], [])
        self.word_label.setText("—")

    # ---- events from pipeline -----------------------------------------

    def _on_event(self, ev: Any) -> None:
        if isinstance(ev, UtteranceStarted):
            self.hold_hint.setText("● listening")
            self.hold_hint.setProperty("active", "true")
            self.hold_hint.style().unpolish(self.hold_hint)
            self.hold_hint.style().polish(self.hold_hint)
            self.live_caption.setText("listening…")
            if self.settings.pipeline_mode in {PipelineMode.STREAMING, PipelineMode.ALIGNED}:
                self.strip.clear()

        elif isinstance(ev, PartialPhonemes):
            for tok in ev.tokens:
                self.strip.append(tok)
            seq = "".join(t.ipa for t in ev.tokens)
            if seq:
                self.live_caption.setText(f"{self.live_caption.text()} {seq}".strip())

        elif isinstance(ev, UtteranceEnded):
            self.hold_hint.setText("● idle")
            self.hold_hint.setProperty("active", "false")
            self.hold_hint.style().unpolish(self.hold_hint)
            self.hold_hint.style().polish(self.hold_hint)

        elif isinstance(ev, Transcription):
            self.word_label.setText(ev.text or "—")
            self.live_caption.setText(" ".join(p.ipa for p in ev.phonemes))
            self.aligned_view.show_transcription(ev.words, ev.phonemes, ev.phoneme_groups)
            if self.settings.echo_after_utterance and ev.text.strip():
                self._speak_echo(ev.text)

        elif isinstance(ev, CompressedWord):
            joined = " ".join(ev.phonemes)
            candidates = ", ".join(ev.candidates[:5]) if ev.candidates else "(no match)"
            self.word_label.setText(candidates.split(",")[0].strip() or "—")
            self.live_caption.setText(f"/{joined}/  →  {candidates}")

        elif isinstance(ev, EchoAudio):
            self.player.play(ev.audio, ev.sample_rate)

    # ---- TTS echo ------------------------------------------------------

    def _speak_echo(self, text: str) -> None:
        if self.settings.tts_backend is not TTSBackend.KOKORO:
            # Other backends not hooked up yet; fall back.
            pass
        import threading

        def _run():
            try:
                if self._tts is None:
                    self._tts = KokoroTTS(
                        voice=self.settings.tts_voice, speed=self.settings.tts_speed
                    )
                audio, sr = self._tts.synthesize(text)
                self.player.play(audio, sr)
            except Exception as e:
                log.exception("TTS failed: %s", e)

        threading.Thread(target=_run, daemon=True).start()

    # ---- hold-to-talk --------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self.settings.hold_to_talk and event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._hold_active = True
            self.hold_hint.setText("● HOLDING — speak")
            self.hold_hint.setProperty("active", "true")
            self.hold_hint.style().unpolish(self.hold_hint)
            self.hold_hint.style().polish(self.hold_hint)
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if self.settings.hold_to_talk and event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._hold_active = False
            self.runner.force_endpoint()
            return
        super().keyReleaseEvent(event)

    # ---- lifecycle -----------------------------------------------------

    def closeEvent(self, event) -> None:
        self.runner.stop()
        return super().closeEvent(event)

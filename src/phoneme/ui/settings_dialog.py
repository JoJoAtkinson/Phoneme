"""Settings dialog — tweak pipeline, models, thresholds, and voices."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QSpinBox,
    QVBoxLayout,
)

from ..config import ASRBackend, PhonemeModel, PipelineMode, Settings, TTSBackend


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Phoneme — Settings")
        self.setMinimumWidth(520)
        self.settings = settings.model_copy()

        root = QVBoxLayout(self)

        # Pipeline
        pipe = QGroupBox("Pipeline")
        pipe_form = QFormLayout(pipe)
        self.pipe_mode = QComboBox()
        for m in PipelineMode:
            self.pipe_mode.addItem(m.value, m)
        self.pipe_mode.setCurrentText(self.settings.pipeline_mode.value)
        pipe_form.addRow("Mode", self.pipe_mode)

        self.hold_to_talk = QCheckBox("Hold Space to talk")
        self.hold_to_talk.setChecked(self.settings.hold_to_talk)
        pipe_form.addRow(self.hold_to_talk)
        root.addWidget(pipe)

        # Phoneme
        ph = QGroupBox("Phoneme Model")
        ph_form = QFormLayout(ph)
        self.phoneme_model = QComboBox()
        for m in PhonemeModel:
            self.phoneme_model.addItem(m.value, m)
        self.phoneme_model.setCurrentText(self.settings.phoneme_model.value)
        ph_form.addRow("Model", self.phoneme_model)

        self.min_conf = QDoubleSpinBox()
        self.min_conf.setRange(0.0, 1.0)
        self.min_conf.setSingleStep(0.05)
        self.min_conf.setValue(self.settings.phoneme_min_confidence)
        ph_form.addRow("Min confidence", self.min_conf)

        self.win_ms = QSpinBox()
        self.win_ms.setRange(160, 2000)
        self.win_ms.setValue(self.settings.phoneme_window_ms)
        ph_form.addRow("Streaming window (ms)", self.win_ms)

        self.hop_ms = QSpinBox()
        self.hop_ms.setRange(40, 500)
        self.hop_ms.setValue(self.settings.phoneme_hop_ms)
        ph_form.addRow("Streaming hop (ms)", self.hop_ms)

        self.mark_vowels = QCheckBox("Mark long/short vowels (¯ / ˘)")
        self.mark_vowels.setChecked(self.settings.mark_long_short_vowels)
        ph_form.addRow(self.mark_vowels)
        root.addWidget(ph)

        # ASR
        asr = QGroupBox("Speech Recognition")
        asr_form = QFormLayout(asr)
        self.asr_backend = QComboBox()
        for b in ASRBackend:
            self.asr_backend.addItem(b.value, b)
        self.asr_backend.setCurrentText(self.settings.asr_backend.value)
        asr_form.addRow("Backend", self.asr_backend)

        self.whisper_size = QComboBox()
        for s in ["tiny", "base", "small", "medium", "large-v3"]:
            self.whisper_size.addItem(s)
        self.whisper_size.setCurrentText(self.settings.whisper_model_size)
        asr_form.addRow("Whisper size", self.whisper_size)
        root.addWidget(asr)

        # TTS
        tts = QGroupBox("Text-to-Speech (Echo)")
        tts_form = QFormLayout(tts)
        self.tts_backend = QComboBox()
        for b in TTSBackend:
            self.tts_backend.addItem(b.value, b)
        self.tts_backend.setCurrentText(self.settings.tts_backend.value)
        tts_form.addRow("Backend", self.tts_backend)

        self.tts_speed = QDoubleSpinBox()
        self.tts_speed.setRange(0.5, 2.0)
        self.tts_speed.setSingleStep(0.05)
        self.tts_speed.setValue(self.settings.tts_speed)
        tts_form.addRow("Speed", self.tts_speed)

        self.echo_on = QCheckBox("Echo after utterance")
        self.echo_on.setChecked(self.settings.echo_after_utterance)
        tts_form.addRow(self.echo_on)
        root.addWidget(tts)

        # VAD
        vad = QGroupBox("Voice Activity")
        vad_form = QFormLayout(vad)
        self.vad_thresh = QDoubleSpinBox()
        self.vad_thresh.setRange(0.1, 0.95)
        self.vad_thresh.setSingleStep(0.05)
        self.vad_thresh.setValue(self.settings.vad_threshold)
        vad_form.addRow("VAD threshold", self.vad_thresh)

        self.silence_ms = QSpinBox()
        self.silence_ms.setRange(100, 3000)
        self.silence_ms.setValue(self.settings.vad_min_silence_ms)
        vad_form.addRow("End-of-utterance silence (ms)", self.silence_ms)
        root.addWidget(vad)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def apply_to(self, s: Settings) -> Settings:
        s.pipeline_mode = self.pipe_mode.currentData()
        s.hold_to_talk = self.hold_to_talk.isChecked()
        s.phoneme_model = self.phoneme_model.currentData()
        s.phoneme_min_confidence = self.min_conf.value()
        s.phoneme_window_ms = self.win_ms.value()
        s.phoneme_hop_ms = self.hop_ms.value()
        s.mark_long_short_vowels = self.mark_vowels.isChecked()
        s.asr_backend = self.asr_backend.currentData()
        s.whisper_model_size = self.whisper_size.currentText()
        s.tts_backend = self.tts_backend.currentData()
        s.tts_speed = self.tts_speed.value()
        s.echo_after_utterance = self.echo_on.isChecked()
        s.vad_threshold = self.vad_thresh.value()
        s.vad_min_silence_ms = self.silence_ms.value()
        return s

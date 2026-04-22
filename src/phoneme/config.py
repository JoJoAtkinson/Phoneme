"""Persisted user settings. Load/save YAML at ~/Library/Application Support/Phoneme/settings.yaml."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .paths import config_file


class PipelineMode(str, Enum):
    """Which end-to-end pipeline drives the UI."""

    STREAMING = "streaming"  # A: live phoneme chips as you speak
    UTTERANCE = "utterance"  # B: wait for pause, then process full clip
    ALIGNED = "aligned"  # C: dual-stream, phonemes aligned under words
    COMPRESS = "compress"  # Phoneme-compression word-building mode


class ASRBackend(str, Enum):
    FASTER_WHISPER = "faster_whisper"
    MLX_WHISPER = "mlx_whisper"
    PARAKEET_MLX = "parakeet_mlx"


class TTSBackend(str, Enum):
    KOKORO = "kokoro"
    OPENVOICE = "openvoice"  # optional, bring-your-own-install
    ELEVENLABS = "elevenlabs"  # bring-your-own-key


class PhonemeModel(str, Enum):
    WAV2VEC2_ESPEAK = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    WAV2VEC2_GRUUT_EN = "bookbot/wav2vec2-ljspeech-gruut"


class Settings(BaseModel):
    # Audio
    sample_rate: int = 16000
    input_device: int | None = None  # None = default mic
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 600  # silence length to end an utterance
    vad_min_speech_ms: int = 80

    # Hold-to-talk
    hold_to_talk: bool = True  # default on per user request
    hold_to_talk_key: str = "space"

    # Pipeline
    pipeline_mode: PipelineMode = PipelineMode.ALIGNED

    # Phoneme model
    phoneme_model: PhonemeModel = PhonemeModel.WAV2VEC2_ESPEAK
    phoneme_window_ms: int = 480  # rolling window for streaming mode
    phoneme_hop_ms: int = 160
    phoneme_min_confidence: float = 0.35
    merge_repeats: bool = True
    mark_long_short_vowels: bool = True  # render ˙/¯ based on duration

    # ASR
    asr_backend: ASRBackend = ASRBackend.FASTER_WHISPER
    whisper_model_size: str = "small"  # tiny, base, small, medium, large-v3
    parakeet_model: str = "mlx-community/parakeet-tdt-0.6b-v2"

    # TTS
    tts_backend: TTSBackend = TTSBackend.KOKORO
    tts_voice: str = "af_sky"  # Kokoro voice id
    tts_speed: float = 0.9
    echo_after_utterance: bool = True

    # Device
    prefer_mps: bool = True  # use Apple Metal when available

    # Keys for paid services
    elevenlabs_api_key_env: str = "ELEVENLABS_API_KEY"
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY"

    # UI
    show_spectrogram: bool = True
    show_mouth_hint: bool = True
    color_by_category: bool = True

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or config_file()
        if path.exists():
            with path.open() as f:
                data = yaml.safe_load(f) or {}
            return cls.model_validate(data)
        return cls()

    def save(self, path: Path | None = None) -> None:
        path = path or config_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)

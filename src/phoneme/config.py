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
    WAV2VEC2_SPEECH31_EN = "speech31/wav2vec2-large-english-phoneme-v2"
    WAV2VEC2_XLSR53_ESPEAK = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    WAV2VEC2_ARPA39_EN = "mostafaashahin/wav2vec2-base-timit-phoneme-arpa-39"

    def catalog_key(self) -> str:
        """Map the HF repo id (stored in settings) to its downloader catalog key."""
        return {
            PhonemeModel.WAV2VEC2_ESPEAK: "phoneme/wav2vec2-espeak",
            PhonemeModel.WAV2VEC2_GRUUT_EN: "phoneme/wav2vec2-gruut-en",
            PhonemeModel.WAV2VEC2_SPEECH31_EN: "phoneme/wav2vec2-speech31-en",
            PhonemeModel.WAV2VEC2_XLSR53_ESPEAK: "phoneme/wav2vec2-xlsr53-espeak",
            PhonemeModel.WAV2VEC2_ARPA39_EN: "phoneme/wav2vec2-base-arpa39",
        }[self]

    def display_name(self) -> str:
        """User-facing label shown in the settings dialog."""
        return {
            PhonemeModel.WAV2VEC2_ESPEAK: "Facebook wav2vec2-lv60 (IPA, multilingual, 315M)",
            PhonemeModel.WAV2VEC2_GRUUT_EN: "Bookbot LJSpeech gruut (IPA, English, 95M)",
            PhonemeModel.WAV2VEC2_SPEECH31_EN: "speech31 (letter+phoneme hybrid, 315M)",
            PhonemeModel.WAV2VEC2_XLSR53_ESPEAK: "Facebook wav2vec2-xlsr53 (IPA, multilingual, 315M)",
            PhonemeModel.WAV2VEC2_ARPA39_EN: "mostafaashahin wav2vec2-base ARPA-39 (English, 95M)  ★ best A/B",
        }[self]


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

    # Phoneme model. facebook/wav2vec2-lv-60-espeak-cv-ft is multilingual
    # but the IPA it emits is the same symbol set English uses, and its
    # CommonVoice training data (thousands of real mics, noisy environments)
    # makes it far more robust to real-world input than the studio-trained
    # "English-only" alternatives. The other options in the enum are kept
    # for experimentation but tend to fold to garbage on laptop-mic audio.
    phoneme_model: PhonemeModel = PhonemeModel.WAV2VEC2_ESPEAK
    phoneme_window_ms: int = 720  # rolling window for streaming mode (longer = more context = more accurate)
    phoneme_hop_ms: int = 160
    # Permissive threshold: short/soft real-world utterances produce phonemes
    # with per-token confidence in the 0.2-0.5 range. Filtering at 0.35 silently
    # drops nearly everything on single-word holds ("hi", "no", "why"). 0.15
    # keeps the obvious junk out while letting legitimate-but-uncertain phonemes
    # render so the user sees what the model heard.
    phoneme_min_confidence: float = 0.15
    merge_repeats: bool = True
    mark_long_short_vowels: bool = True  # render ˙/¯ based on duration

    # ASR
    asr_backend: ASRBackend = ASRBackend.FASTER_WHISPER
    # English-only Whisper by default: .en checkpoints are smaller and more
    # accurate on English than the same-size multilingual models.
    whisper_model_size: str = "small.en"  # tiny.en, base.en, small.en, medium.en, small, large-v3
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

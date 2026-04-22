"""Typed events emitted by pipelines. UI subscribes via Qt signals."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..models.phoneme import PhonemeToken


@dataclass
class PartialPhonemes:
    """Newly-stable phonemes detected during a streaming window."""

    tokens: list[PhonemeToken]


@dataclass
class UtteranceStarted:
    pass


@dataclass
class UtteranceEnded:
    audio: np.ndarray
    sample_rate: int


@dataclass
class Transcription:
    text: str
    words: list  # list[WordToken]
    phonemes: list[PhonemeToken]
    # When provided, indices into ``phonemes`` grouped by word
    phoneme_groups: list[list[int]] = field(default_factory=list)


@dataclass
class EchoAudio:
    audio: np.ndarray
    sample_rate: int


@dataclass
class CompressedWord:
    """Result of phoneme-compression mode."""

    phonemes: list[str]
    candidates: list[str]  # best matching words from CMU lookup

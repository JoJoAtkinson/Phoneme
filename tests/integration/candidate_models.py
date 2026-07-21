"""Shared fixtures + scoring for phoneme-model A/B tests.

The baseline (facebook/wav2vec2-lv-60-espeak-cv-ft) lives in the app as the
default. Candidates declared here are tested alongside it — without touching
the app's settings or the baseline tests in test_tts_to_phoneme.py.

We score each candidate on the same audio inputs (clean TTS, attenuated +
noisy TTS, multi-word) and record:

    * coverage    — expected phonemes present (0.0-1.0 per word)
    * spurious    — fraction of output phonemes that weren't expected
    * count       — how many phonemes came out (filter for "collapsed to 1")
    * latency_ms  — inference time on MPS
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

import numpy as np
import pytest

# ---- candidate roster ----------------------------------------------------


@dataclass
class Candidate:
    """A phoneme model under evaluation."""

    catalog_key: str
    label: str  # short name for reports
    notation: str  # "ipa" or "arpa"
    approx_params_m: int  # for size-aware scoring
    tonal_variant_vocab: bool  # True if vocab has Mandarin tones


CANDIDATES: list[Candidate] = [
    # Current app default — our reference baseline.
    Candidate(
        catalog_key="phoneme/wav2vec2-espeak",
        label="lv60-espeak-BASELINE",
        notation="ipa",
        approx_params_m=315,
        tonal_variant_vocab=True,
    ),
    # XLSR-53 pretrain + same espeak decoder — sibling of baseline.
    Candidate(
        catalog_key="phoneme/wav2vec2-xlsr53-espeak",
        label="xlsr53-espeak",
        notation="ipa",
        approx_params_m=315,
        tonal_variant_vocab=True,
    ),
    # English-only IPA, 42-token focused vocab.
    Candidate(
        catalog_key="phoneme/wav2vec2-xls-r-timit",
        label="xls-r-timit-en",
        notation="ipa",
        approx_params_m=315,
        tonal_variant_vocab=False,
    ),
    # Smaller wav2vec2-base with ARPA-39 English notation.
    Candidate(
        catalog_key="phoneme/wav2vec2-base-arpa39",
        label="base-arpa39-en",
        notation="arpa",
        approx_params_m=95,
        tonal_variant_vocab=False,
    ),
]


# ---- pytest guard (integration only, macOS `say`) -----------------------

_INTEGRATION = os.environ.get("PHONEME_RUN_INTEGRATION") == "1"
_SAY = sys.platform == "darwin" and shutil.which("say") is not None

require_integration = pytest.mark.skipif(
    not _INTEGRATION,
    reason="set PHONEME_RUN_INTEGRATION=1 (loads ~4 GB of models, 1-2 min)",
)
require_say = pytest.mark.skipif(
    not _SAY,
    reason="requires macOS `say` for deterministic TTS",
)


# ---- audio synthesis -----------------------------------------------------


def say_audio(text: str, voice: str = "Samantha") -> tuple[np.ndarray, int]:
    """Synthesize text via `say` at 16 kHz / float32."""
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        subprocess.run(
            ["say", "-v", voice, "-o", wav_path,
             "--file-format=WAVE", "--data-format=LEF32@16000", text],
            check=True,
        )
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        return np.ascontiguousarray(audio), sr
    finally:
        os.unlink(wav_path)


def roughen(audio: np.ndarray, peak: float = 0.18, noise_db: float = -26.0) -> np.ndarray:
    """Attenuate + add white noise to approximate real laptop-mic audio."""
    if len(audio) == 0:
        return audio
    scaled = audio / max(float(np.max(np.abs(audio))), 1e-8) * peak
    noise_amp = peak * (10 ** (noise_db / 20.0))
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(len(scaled)).astype(np.float32) * noise_amp
    return (scaled + noise).astype(np.float32)


# ---- phoneme equivalence classes ---------------------------------------
#
# Different models emit different symbols for the same sound. A model that
# says "hello" is /h eɪ l ow/ is just as correct as one that says /h ɛ l oʊ/.
# Score against equivalence sets so notation differences don't penalize.

# IPA equivalence classes (per English phoneme)
IPA_CLASSES = {
    "H":  {"h"},
    "EH": {"ɛ", "e", "ə", "ɐ", "æ"},       # hello, bed — lax mid vowels
    "EY": {"eɪ", "eː", "e"},                # say — long a
    "IH": {"ɪ", "i", "ᵻ"},
    "IY": {"i", "iː"},                      # see — long e
    "L":  {"l", "ɫ"},
    "OW": {"oʊ", "əʊ", "o", "oː", "ow"},    # hello, go — long o
    "AO": {"ɔ", "ɔː", "ɑ", "ɑː"},
    "R":  {"ɹ", "r", "ɚ", "ɝ", "ɜː"},       # world, butter
    "W":  {"w"},
    "D":  {"d"},
    "T":  {"t"},
    "N":  {"n"},
    "M":  {"m"},
    "AY": {"aɪ"},
    "AW": {"aʊ"},
}

# ARPA equivalence (2-letter codes = same phoneme label; identity mapping
# except we need to uppercase the model's lowercase output).
ARPA_CLASSES = {
    "H":  {"hh"},
    "EH": {"eh", "ah"},
    "EY": {"ey"},
    "IH": {"ih"},
    "IY": {"iy"},
    "L":  {"l"},
    "OW": {"ow"},
    "AO": {"ao", "aa"},
    "R":  {"r", "er"},
    "W":  {"w"},
    "D":  {"d"},
    "T":  {"t"},
    "N":  {"n"},
    "M":  {"m"},
    "AY": {"ay"},
    "AW": {"aw"},
}


def classify_phoneme(ipa: str, notation: str) -> str | None:
    """Return the equivalence-class tag for a model-emitted phoneme, or None
    if it doesn't match any expected English phoneme."""
    classes = IPA_CLASSES if notation == "ipa" else ARPA_CLASSES
    # Case-normalize ARPA tokens (some models emit lowercase).
    needle = ipa.lower() if notation == "arpa" else ipa
    for tag, variants in classes.items():
        if needle in variants:
            return tag
    return None


# ---- test phrases and their expected phoneme tags ----------------------


@dataclass
class Phrase:
    text: str
    tags: list[str]  # expected equivalence-class tags, in order


PHRASES: dict[str, Phrase] = {
    "hello":       Phrase("Hello.",       ["H", "EH", "L", "OW"]),
    "hello_world": Phrase("Hello world.", ["H", "EH", "L", "OW", "W", "R", "L", "D"]),
    "why_me":      Phrase("Why me?",      ["W", "AY", "M", "IY"]),
    "goodbye":     Phrase("Goodbye.",     ["D", "AY"]),  # model often loses /g ʊ/
    "okay_then":   Phrase("Okay then.",   ["OW", "K", "EY", "EH", "N"]),
}


# ---- scorecard ---------------------------------------------------------


@dataclass
class Score:
    candidate: str
    phrase: str
    raw_ipa: list[str]
    raw_tags: list[str | None]  # None for spurious (unmapped) phonemes
    expected_tags: list[str]
    latency_ms: float
    rough: bool = False

    @property
    def coverage(self) -> float:
        """Fraction of expected phoneme tags present in output."""
        if not self.expected_tags:
            return 1.0
        output = set(t for t in self.raw_tags if t)
        hit = sum(1 for t in self.expected_tags if t in output)
        return hit / len(set(self.expected_tags))

    @property
    def spurious_rate(self) -> float:
        """Fraction of output tokens that didn't map to any expected tag."""
        if not self.raw_tags:
            return 0.0
        return sum(1 for t in self.raw_tags if t is None) / len(self.raw_tags)

    @property
    def count(self) -> int:
        return len(self.raw_ipa)


# Per-process scorecard registry — tests append, a final summary prints it.
_ALL_SCORES: list[Score] = []


def record(score: Score) -> None:
    _ALL_SCORES.append(score)


def summarize() -> str:
    if not _ALL_SCORES:
        return "(no scores recorded)"
    lines = ["", "=" * 100]
    lines.append(f"{'candidate':<28}{'phrase':<14}{'rough':<7}{'n':<4}{'cov':<7}{'spur':<7}{'ms':<8}  output")
    lines.append("-" * 100)
    for s in _ALL_SCORES:
        lines.append(
            f"{s.candidate:<28}{s.phrase:<14}{str(s.rough):<7}"
            f"{s.count:<4}{s.coverage:<7.2f}{s.spurious_rate:<7.2f}{s.latency_ms:<8.0f}  {s.raw_ipa}"
        )
    lines.append("=" * 100)
    # Aggregate by candidate
    from collections import defaultdict

    agg: dict[str, list[Score]] = defaultdict(list)
    for s in _ALL_SCORES:
        agg[s.candidate].append(s)
    lines.append("\nAGGREGATE (mean across all phrases):")
    lines.append(f"{'candidate':<28}{'avg_cov':<10}{'avg_spur':<10}{'avg_ms':<10}{'n_collapse':<12}")
    lines.append("-" * 70)
    rows = []
    for name, scores in agg.items():
        avg_cov = sum(s.coverage for s in scores) / len(scores)
        avg_spur = sum(s.spurious_rate for s in scores) / len(scores)
        avg_ms = sum(s.latency_ms for s in scores) / len(scores)
        n_collapse = sum(1 for s in scores if s.count <= 1)
        rows.append((name, avg_cov, avg_spur, avg_ms, n_collapse))
    # Sort by coverage desc, then spurious asc, then latency asc.
    rows.sort(key=lambda r: (-r[1], r[2], r[3]))
    for name, cov, spur, ms, col in rows:
        lines.append(f"{name:<28}{cov:<10.2f}{spur:<10.2f}{ms:<10.0f}{col:<12}")
    lines.append("=" * 100)
    return "\n".join(lines)


# ---- inference helper --------------------------------------------------


def run_candidate(cand: Candidate, audio: np.ndarray, sr: int, phrase_key: str,
                  rough: bool = False) -> Score:
    from phoneme.models.phoneme import Wav2Vec2PhonemeRecognizer

    rec = Wav2Vec2PhonemeRecognizer(model_key=cand.catalog_key)
    rec.load()  # warm separately so latency measures pure inference

    t0 = time.perf_counter()
    tokens = rec.transcribe(audio, sample_rate=sr, min_confidence=0.15)
    latency_ms = (time.perf_counter() - t0) * 1000

    raw_ipa = [t.ipa for t in tokens]
    raw_tags = [classify_phoneme(ipa, cand.notation) for ipa in raw_ipa]

    phrase = PHRASES[phrase_key]
    score = Score(
        candidate=cand.label,
        phrase=phrase_key,
        raw_ipa=raw_ipa,
        raw_tags=raw_tags,
        expected_tags=phrase.tags,
        latency_ms=latency_ms,
        rough=rough,
    )
    record(score)
    return score

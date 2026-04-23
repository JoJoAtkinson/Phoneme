"""End-to-end smoke test: synthesize known words via macOS `say`, feed the
audio through the exact same code path `make run` uses (minus mic capture
and Qt widgets), and assert the output is sensible.

Covers:
  * `Wav2Vec2PhonemeRecognizer.transcribe` on the app's default model.
  * `FasterWhisperASR.transcribe` on the app's default size.
  * `AlignedPipeline._on_utterance`, which is what `force_endpoint_now`
    dispatches to on a hold release.
  * A realistic end-to-end run: feed audio into the `Pipeline`'s mic
    ring buffer, let `StreamingVAD.feed` process it, and verify VAD +
    utterance + Transcription fires — i.e. the drain-loop path.
  * A hostile variant with attenuated gain + white noise, matching real
    laptop-mic audio characteristics (peaks < 0.2, SNR ~20 dB).

The phoneme model on multilingual CommonVoice-trained checkpoints emits
IPA that's identical whether the speaker is English or not, so we assert
on the presence of core English phonemes — not exact sequences, since
real-audio recognition is stochastic on the edges.

Skipped unless ``PHONEME_RUN_INTEGRATION=1`` — this loads ~500 MB of
models and takes 10-30 s. Invoke locally with::

    PHONEME_RUN_INTEGRATION=1 make test
    # or:
    PHONEME_RUN_INTEGRATION=1 .venv/bin/python -m pytest tests/integration/ -s
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pytest

INTEGRATION_ENABLED = os.environ.get("PHONEME_RUN_INTEGRATION") == "1"
_SAY_AVAILABLE = sys.platform == "darwin" and shutil.which("say") is not None

pytestmark = [
    pytest.mark.skipif(
        not INTEGRATION_ENABLED,
        reason="set PHONEME_RUN_INTEGRATION=1 to run (loads ~500 MB of models)",
    ),
    pytest.mark.skipif(
        not _SAY_AVAILABLE,
        reason="requires macOS `say` for deterministic TTS",
    ),
]


# ---------- audio fixtures -------------------------------------------------


def _say(text: str, voice: str = "Samantha") -> tuple[np.ndarray, int]:
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


def _roughen(audio: np.ndarray, peak: float = 0.18, noise_db: float = -26.0) -> np.ndarray:
    """Scale audio to a realistic laptop-mic peak and add white noise.

    Real mic audio in the logs was peak ~0.12-0.20, rms ~0.04 with ambient
    noise. We approximate that here so the integration test exercises what
    the app actually sees at runtime, not the abnormally clean TTS output."""
    if len(audio) == 0:
        return audio
    scaled = audio / max(float(np.max(np.abs(audio))), 1e-8) * peak
    noise_amp = peak * (10 ** (noise_db / 20.0))
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(len(scaled)).astype(np.float32) * noise_amp
    return (scaled + noise).astype(np.float32)


@pytest.fixture(scope="module")
def hello():
    return _say("Hello.")


@pytest.fixture(scope="module")
def hello_rough(hello):
    audio, sr = hello
    return _roughen(audio), sr


@pytest.fixture(scope="module")
def hello_world():
    return _say("Hello world.")


@pytest.fixture(scope="module")
def why_me():
    return _say("Why me?")


# ---------- phoneme recognizer -------------------------------------------


def _assert_has_any(tokens_ipa, expected_sets, label=""):
    """Assert at least one IPA from each `expected` set is present."""
    found = set(tokens_ipa)
    for expected in expected_sets:
        assert any(e in found for e in expected), (
            f"[{label}] missing any of {expected} in {tokens_ipa}"
        )


def test_phoneme_hello_clean(hello):
    """Clean TTS 'hello' must give us /l/ plus a mid vowel plus an o-family
    vowel. The espeak model emits `oː` (length-marked) for the final vowel
    of hello, which is the same sound as `oʊ` — accept either."""
    from phoneme.models.phoneme import Wav2Vec2PhonemeRecognizer

    audio, sr = hello
    rec = Wav2Vec2PhonemeRecognizer()  # app default (espeak lv-60)
    tokens = rec.transcribe(audio, sample_rate=sr, min_confidence=0.15)
    ipas = [t.ipa for t in tokens]
    print(f"\n[clean Hello]  → {ipas}")
    assert tokens, "zero phonemes on clean TTS hello"
    _assert_has_any(
        ipas,
        [{"l"},
         {"ɛ", "ə", "eɪ", "e"},
         {"oʊ", "əʊ", "o", "ow", "oː", "ɔ", "ɔː"}],
        label="hello/clean",
    )


def test_phoneme_hello_rough(hello_rough):
    """Attenuated + noisy audio (like real laptop mic) must still produce
    multiple phonemes, not just one — regression against the bookbot
    LJSpeech-only failure mode where real-mic audio collapsed to a single
    chip. Phoneme identity is not asserted here because white noise +
    attenuation genuinely degrades recognition; we only assert that the
    model doesn't give up entirely."""
    from phoneme.models.phoneme import Wav2Vec2PhonemeRecognizer

    audio, sr = hello_rough
    rec = Wav2Vec2PhonemeRecognizer()
    tokens = rec.transcribe(audio, sample_rate=sr, min_confidence=0.15)
    ipas = [t.ipa for t in tokens]
    print(f"\n[rough Hello]  → {ipas}  peak={np.max(np.abs(audio)):.2f}")
    # Regression assertion: must not collapse to 0 or 1 phoneme.
    assert len(tokens) >= 2, f"only {len(tokens)} phoneme(s) on rough audio: {ipas}"


def test_phoneme_hello_world(hello_world):
    from phoneme.models.phoneme import Wav2Vec2PhonemeRecognizer

    audio, sr = hello_world
    rec = Wav2Vec2PhonemeRecognizer()
    tokens = rec.transcribe(audio, sample_rate=sr, min_confidence=0.15)
    ipas = [t.ipa for t in tokens]
    print(f"\n[Hello world]  → {ipas}")
    assert tokens
    # At least /l/ (hello, world), and some r-colored or /w/ from "world".
    _assert_has_any(ipas, [{"l"}, {"w", "ɹ", "r", "ɚ", "ɝ"}], label="hello world")


# ---------- ASR ----------------------------------------------------------


def test_asr_hello_world(hello_world):
    from phoneme.models.asr_whisper import FasterWhisperASR

    audio, sr = hello_world
    asr = FasterWhisperASR(model_size="tiny.en")  # matches app default after last round
    words = asr.transcribe(audio, sample_rate=sr)
    text = " ".join(w.text for w in words).lower().strip(" .,!")
    print(f"\n[ASR hello world] → {text!r}")
    assert "hello" in text
    assert "world" in text


# ---------- full aligned pipeline (no mic) --------------------------------


def test_aligned_on_utterance_path(hello_world):
    """Exercise AlignedPipeline._on_utterance — the exact function
    `force_endpoint_now` invokes on hold-release."""
    from phoneme.config import Settings
    from phoneme.pipeline.aligned import AlignedPipeline
    from phoneme.pipeline.events import Transcription

    settings = Settings()
    events: list = []
    pipeline = AlignedPipeline(settings, emit=events.append)

    audio, _ = hello_world
    pipeline._on_utterance(audio)

    transcriptions = [e for e in events if isinstance(e, Transcription)]
    assert transcriptions, (
        f"no Transcription emitted; got {[type(e).__name__ for e in events]}"
    )
    t = transcriptions[0]
    text = t.text.lower().strip(" .,!")
    print(f"\n[pipeline hello world] text={text!r} phones={[p.ipa for p in t.phonemes]}")
    assert "hello" in text and "world" in text
    assert t.phonemes, "aligned pipeline emitted zero phonemes on clean input"
    assert len(t.phoneme_groups) == len(t.words)


# ---------- full drain-loop path through VAD ------------------------------


def test_end_to_end_through_vad_and_pipeline(hello_world):
    """Highest-fidelity test: feed TTS audio into the mic ring buffer the
    same way `MicCapture._callback` does, let the drain loop pass it to
    Silero VAD, and assert the full chain (VAD → utterance buffering →
    _on_utterance → Transcription) produces the expected event.

    This is the closest we can get to `make run` without opening an
    actual mic stream or running Qt. It will catch regressions in the
    interaction between MicCapture, RingBuffer, StreamingVAD, force_end,
    and the pipeline's _on_utterance."""
    from phoneme.config import Settings
    from phoneme.pipeline.aligned import AlignedPipeline
    from phoneme.pipeline.events import Transcription, UtteranceStarted

    settings = Settings()
    events: list = []
    pipeline = AlignedPipeline(settings, emit=events.append)
    pipeline.start()
    try:
        # Unpause so drain_loop feeds audio into VAD (hold-to-talk default
        # starts paused).
        pipeline.set_paused(False)

        audio, sr = hello_world
        # Pad with 400 ms of silence front/back so VAD sees a clean onset
        # and has enough trailing silence to auto-fire speech_end.
        silence = np.zeros(int(sr * 0.4), dtype=np.float32)
        padded = np.concatenate([silence, audio.astype(np.float32), silence])

        # Write in 32 ms chunks at wall-clock pace to simulate mic streaming.
        chunk = int(sr * 0.032)
        assert pipeline.mic is not None
        for i in range(0, len(padded), chunk):
            pipeline.mic.buffer.write(padded[i : i + chunk])
            time.sleep(0.005)  # let drain_loop catch up; doesn't need realtime

        # Wait up to 5 s for the Transcription event to land. VAD + ASR +
        # phoneme can each take a few hundred ms on first pass.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if any(isinstance(e, Transcription) for e in events):
                break
            time.sleep(0.05)
    finally:
        pipeline.stop()

    names = [type(e).__name__ for e in events]
    print(f"\n[end-to-end] events: {names}")
    assert any(isinstance(e, UtteranceStarted) for e in events), (
        f"VAD never fired speech_start: {names}"
    )
    transcriptions = [e for e in events if isinstance(e, Transcription)]
    assert transcriptions, f"no Transcription emitted end-to-end: {names}"
    t = transcriptions[-1]
    text = t.text.lower().strip(" .,!")
    print(f"[end-to-end] text={text!r} phones={[p.ipa for p in t.phonemes]}")
    assert "hello" in text and "world" in text
    assert t.phonemes, "end-to-end emitted a Transcription with zero phonemes"


def test_end_to_end_rough_audio(hello_world):
    """End-to-end with attenuated + noisy audio (laptop mic conditions).
    Asserts the pipeline still produces a usable Transcription."""
    from phoneme.config import Settings
    from phoneme.pipeline.aligned import AlignedPipeline
    from phoneme.pipeline.events import Transcription

    settings = Settings()
    events: list = []
    pipeline = AlignedPipeline(settings, emit=events.append)
    pipeline.start()
    try:
        pipeline.set_paused(False)
        audio, sr = hello_world
        rough = _roughen(audio, peak=0.18, noise_db=-26)
        silence = np.zeros(int(sr * 0.4), dtype=np.float32)
        padded = np.concatenate([silence, rough, silence])

        chunk = int(sr * 0.032)
        assert pipeline.mic is not None
        for i in range(0, len(padded), chunk):
            pipeline.mic.buffer.write(padded[i : i + chunk])
            time.sleep(0.005)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if any(isinstance(e, Transcription) for e in events):
                break
            time.sleep(0.05)
    finally:
        pipeline.stop()

    transcriptions = [e for e in events if isinstance(e, Transcription)]
    assert transcriptions, "no Transcription on rough audio end-to-end"
    t = transcriptions[-1]
    text = t.text.lower().strip(" .,!")
    print(f"\n[end-to-end rough] text={text!r} phones={[p.ipa for p in t.phonemes]}")
    assert "hello" in text or "world" in text, f"ASR lost both words: {text!r}"
    assert t.phonemes, "zero phonemes on rough end-to-end"

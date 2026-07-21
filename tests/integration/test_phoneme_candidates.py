"""A/B evaluation across phoneme model candidates.

Each candidate runs the same scorecard (coverage, spurious rate, latency,
count) over the same synthesized phrases — clean and rough variants. The
baseline (the app's current default) is included for reference.

At the end, `test_zzz_summary` prints a comparative leaderboard and asserts
the baseline doesn't silently regress.

This file deliberately does NOT touch the app's default model or overwrite
the baseline tests in test_tts_to_phoneme.py.

Run::

    PHONEME_RUN_INTEGRATION=1 .venv/bin/python -m pytest \\
        tests/integration/test_phoneme_candidates.py -s
"""

from __future__ import annotations

import pytest

from .candidate_models import (
    CANDIDATES,
    PHRASES,
    Candidate,
    require_integration,
    require_say,
    roughen,
    run_candidate,
    say_audio,
    summarize,
)

pytestmark = [require_integration, require_say]


# Module-scoped audio so each phrase is synthesized once and reused across
# all candidates.
@pytest.fixture(scope="module")
def audio_fixtures():
    out: dict[str, tuple] = {}
    for key, phrase in PHRASES.items():
        audio, sr = say_audio(phrase.text)
        out[key] = (audio, sr)
    return out


@pytest.fixture(scope="module")
def rough_fixtures(audio_fixtures):
    return {k: (roughen(a), sr) for k, (a, sr) in audio_fixtures.items()}


# Parametrize phrase × candidate. pytest gives us clean test IDs.
@pytest.mark.parametrize("cand", CANDIDATES, ids=lambda c: c.label)
@pytest.mark.parametrize("phrase_key", list(PHRASES.keys()))
def test_clean(cand: Candidate, phrase_key: str, audio_fixtures):
    audio, sr = audio_fixtures[phrase_key]
    score = run_candidate(cand, audio, sr, phrase_key, rough=False)
    print(f"\n  {cand.label:<28} clean {phrase_key:<14} → {score.raw_ipa}  "
          f"cov={score.coverage:.2f} spur={score.spurious_rate:.2f} "
          f"n={score.count} {score.latency_ms:.0f}ms")
    # Minimal correctness gate — don't let a candidate silently give empty
    # output. We leave strict gating to the aggregate summary test.
    assert score.count > 0, f"{cand.label} returned zero phonemes on clean {phrase_key}"


@pytest.mark.parametrize("cand", CANDIDATES, ids=lambda c: c.label)
@pytest.mark.parametrize("phrase_key", ["hello", "hello_world"])
def test_rough(cand: Candidate, phrase_key: str, rough_fixtures):
    """Attenuated + noisy audio. This is the real-world robustness test."""
    audio, sr = rough_fixtures[phrase_key]
    score = run_candidate(cand, audio, sr, phrase_key, rough=True)
    print(f"\n  {cand.label:<28} rough {phrase_key:<14} → {score.raw_ipa}  "
          f"cov={score.coverage:.2f} spur={score.spurious_rate:.2f} "
          f"n={score.count} {score.latency_ms:.0f}ms")


def test_zzz_summary():
    """Printed leaderboard + baseline non-regression gate.

    pytest runs tests alphabetically per file by default, but with parametrize
    this may run first — the module-level cache guarantees scores accumulate
    regardless of ordering. The "zzz" name is a hint, not a requirement."""
    report = summarize()
    print(report)

    # Non-regression check: baseline must not score below absolute floors that
    # we already validated in test_tts_to_phoneme.py. Failing here means the
    # eval harness itself regressed (bad fixture, bad classifier, etc.), not
    # that we should abandon the baseline.
    from .candidate_models import _ALL_SCORES

    baseline_scores = [s for s in _ALL_SCORES if "BASELINE" in s.candidate]
    assert baseline_scores, "no baseline scores recorded — harness bug"
    avg_cov = sum(s.coverage for s in baseline_scores) / len(baseline_scores)
    avg_count = sum(s.count for s in baseline_scores) / len(baseline_scores)
    assert avg_cov >= 0.5, f"baseline avg coverage {avg_cov:.2f} below floor"
    assert avg_count >= 3, f"baseline avg phoneme count {avg_count:.1f} below floor"

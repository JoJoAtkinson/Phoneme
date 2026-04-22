"""Unit tests for the IPA helpers — no audio, no models, no network."""

from __future__ import annotations

from phoneme.data.ipa import classify, color_for, is_vowel, vowel_length_marker


def test_classify_vowels():
    assert classify("iː") == "vowel_long"
    assert classify("ɪ") == "vowel_short"
    assert classify("aɪ") == "vowel_diphthong"


def test_classify_consonants():
    assert classify("p") == "stop_unvoiced"
    assert classify("b") == "stop_voiced"
    assert classify("ʃ") == "fricative_unvoiced"
    assert classify("z") == "fricative_voiced"
    assert classify("m") == "nasal"
    assert classify("tʃ") == "affricate"
    assert classify("w") == "approximant"


def test_color_for_returns_hex():
    c = color_for("iː")
    assert c.startswith("#") and len(c) == 7


def test_vowel_length_marker():
    assert vowel_length_marker("iː", 0.05) == "¯"  # ipa says long regardless
    assert vowel_length_marker("ɪ", 0.05) == "˘"  # short
    assert vowel_length_marker("ɪ", 0.25) == "¯"  # held long → upgrade
    assert vowel_length_marker("p", 0.1) is None  # non-vowel


def test_is_vowel():
    assert is_vowel("ə")
    assert not is_vowel("t")

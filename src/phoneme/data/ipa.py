"""IPA helpers: category coloring, long/short vowel detection, articulation hints."""

from __future__ import annotations

# Category → fill color (QSS-friendly hex)
CATEGORY_COLOR = {
    "vowel_long": "#f6a623",
    "vowel_short": "#f5d742",
    "vowel_diphthong": "#f78f3e",
    "nasal": "#8e6bd9",
    "stop_voiced": "#4c8bf5",
    "stop_unvoiced": "#8fa7c7",
    "fricative_voiced": "#2aa6a0",
    "fricative_unvoiced": "#7dc3be",
    "affricate": "#d36bd0",
    "approximant": "#62c478",
    "other": "#bbbbbb",
}

# Strip diacritics like ː, ˈ, ˌ before classifying
_STRIP = str.maketrans("", "", "ˈˌ")


# Canonical espeak/IPA vowel sets, plus ARPA-39 equivalents so the same
# classifier works regardless of the emitting model's notation.
LONG_VOWELS = {"iː", "uː", "ɑː", "ɔː", "ɜː", "eː", "oː", "aː", "yː",
               "iy", "uw", "er"}
SHORT_VOWELS = {"ɪ", "ʊ", "ʌ", "ə", "æ", "ɛ", "ɒ", "ɐ", "ɨ", "ʉ",
                "ih", "uh", "ah", "eh", "ae", "aa", "ao"}
DIPHTHONGS = {"aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ", "ɪə", "eə", "ʊə", "əʊ",
              "ay", "aw", "ey", "ow", "oy"}

VOICED_STOPS = {"b", "d", "ɡ", "g"}
UNVOICED_STOPS = {"p", "t", "k", "ʔ"}
NASALS = {"m", "n", "ŋ", "ɲ", "ng"}
VOICED_FRICATIVES = {"v", "z", "ʒ", "ð", "ɣ", "ʁ", "h",
                     "zh", "dh", "hh"}
UNVOICED_FRICATIVES = {"f", "s", "ʃ", "θ", "x", "ç",
                       "sh", "th"}
AFFRICATES = {"tʃ", "dʒ", "ts", "dz", "ch", "jh"}
APPROXIMANTS = {"j", "w", "l", "ɹ", "r", "ɻ", "y"}


def classify(ipa: str) -> str:
    t = ipa.translate(_STRIP)
    if t in LONG_VOWELS or t.endswith("ː"):
        return "vowel_long"
    if t in DIPHTHONGS:
        return "vowel_diphthong"
    if t in SHORT_VOWELS:
        return "vowel_short"
    if t in VOICED_STOPS:
        return "stop_voiced"
    if t in UNVOICED_STOPS:
        return "stop_unvoiced"
    if t in NASALS:
        return "nasal"
    if t in VOICED_FRICATIVES:
        return "fricative_voiced"
    if t in UNVOICED_FRICATIVES:
        return "fricative_unvoiced"
    if t in AFFRICATES:
        return "affricate"
    if t in APPROXIMANTS:
        return "approximant"
    return "other"


def color_for(ipa: str) -> str:
    return CATEGORY_COLOR[classify(ipa)]


def is_vowel(ipa: str) -> bool:
    cat = classify(ipa)
    return cat.startswith("vowel_")


# Long/short marker for pedagogical display. We mark:
#   * IPA-native long vowels (contain ː) → macron: ā
#   * Duration-observed long vowels (held > threshold) → macron: ¯
#   * Short vowels → breve: ̆
# For non-vowels we return None.
LONG_DURATION_S = 0.18


def vowel_length_marker(ipa: str, duration_s: float) -> str | None:
    if not is_vowel(ipa):
        return None
    if "ː" in ipa or ipa in LONG_VOWELS or duration_s >= LONG_DURATION_S:
        return "¯"
    return "˘"


# Place-of-articulation hint for the mouth diagram
ARTICULATION = {
    "p": "bilabial stop",
    "b": "bilabial stop (voiced)",
    "t": "alveolar stop",
    "d": "alveolar stop (voiced)",
    "k": "velar stop",
    "g": "velar stop (voiced)",
    "ɡ": "velar stop (voiced)",
    "m": "bilabial nasal",
    "n": "alveolar nasal",
    "ŋ": "velar nasal",
    "f": "labiodental fricative",
    "v": "labiodental fricative (voiced)",
    "θ": "dental fricative",
    "ð": "dental fricative (voiced)",
    "s": "alveolar fricative",
    "z": "alveolar fricative (voiced)",
    "ʃ": "postalveolar fricative",
    "ʒ": "postalveolar fricative (voiced)",
    "h": "glottal fricative",
    "tʃ": "postalveolar affricate",
    "dʒ": "postalveolar affricate (voiced)",
    "j": "palatal approximant",
    "w": "labial-velar approximant",
    "l": "alveolar lateral",
    "ɹ": "alveolar approximant",
    "r": "alveolar trill",
}


def articulation_hint(ipa: str) -> str:
    base = ipa.translate(_STRIP).rstrip("ː")
    return ARTICULATION.get(base, "")


# Multilingual espeak-trained wav2vec2 models emit tokens outside the English
# phoneme set: Mandarin tone numbers ("iɛ5", "ə1"), French nasals ("ɑ̃"),
# other-language consonants ("ʁ", "β", "ɣ"). On ambiguous English audio the
# model sometimes prefers these variants over the plain English equivalent,
# which shows up in the UI as mysterious "chinese tone 5" chips. Normalize
# these down to their nearest English phoneme before display.
_NON_ENGLISH_TO_ENGLISH = {
    "ʁ": "ɹ",     # French/German R → English r
    "r":  "ɹ",    # alveolar trill → approximant r (most English speakers)
    "ɾ": "ɾ",     # flap (keep — valid English "butter" allophone)
    "ɑ̃": "ɑ",     # French nasal a → a
    "ɔ̃": "ɔ",     # French nasal ɔ → ɔ
    "β": "v",     # Spanish β → closest English consonant
    "ɣ": "ɡ",     # voiced velar fricative → g
    "ç": "h",     # palatal fricative → h (as in "huge")
    "ɐ": "ə",     # near-open → schwa (often interchangeable)
    "ᵻ": "ɪ",     # unstressed barred-i → ɪ
    "i.5": "i",
    "i̪5": "i",
    "i̪": "i",
}


def normalize_to_english(ipa: str) -> str:
    """Map a potentially non-English IPA token to its closest English phoneme.

    Strips Mandarin tone numbers (digits 1-9), remaps French/German/Spanish
    consonants to their English counterparts. Returns the normalized token;
    caller can then decide whether to keep or drop it (empty string means
    the token was non-phonetic / should be dropped)."""
    if not ipa:
        return ipa
    # Strip trailing tone digits (Mandarin/Cantonese/Vietnamese markers).
    stripped = ipa.rstrip("0123456789")
    if stripped != ipa:
        ipa = stripped
    # Direct remaps for whole-token non-English forms.
    if ipa in _NON_ENGLISH_TO_ENGLISH:
        return _NON_ENGLISH_TO_ENGLISH[ipa]
    return ipa


# Elementary-school style labels. Legible to anyone who learned phonics in
# primary school — uses macron (ā ē ī ō ū) for long vowels and breve
# (ă ĕ ĭ ŏ ŭ) for short, plus common digraphs (sh, ch, th, ng). The chip
# widget shows this as the main glyph; the raw IPA stays in the tooltip.
#
# Includes both IPA tokens (from espeak/gruut/xlsr models) and ARPA-39
# tokens (from mostafaashahin/wav2vec2-base-timit-phoneme-arpa-39), so we
# can render clean elementary labels regardless of which model is selected.
SIMPLE_LABEL = {
    # ---- vowels ------------------------------------------------------
    # long vowels and diphthongs
    "i":  "ē",  "iː": "ē",
    "u":  "ū",  "uː": "ū",
    "eɪ": "ā",
    "aɪ": "ī",
    "oʊ": "ō",  "əʊ": "ō",
    # short vowels
    "ɛ":  "ĕ",
    "ɪ":  "ĭ",
    "æ":  "ă",
    "ʌ":  "ŭ",
    "ə":  "ə",       # schwa — keep distinct; common and worth learning
    "ʊ":  "oo",      # book-style short oo
    "ɑ":  "ŏ",  "ɑː": "ŏ",
    "ɒ":  "ŏ",
    "ɔ":  "aw", "ɔː": "aw",
    # other diphthongs / r-colored
    "aʊ": "ow",
    "ɔɪ": "oy",
    "ɚ":  "ər",
    "ɝ":  "ər",
    # ---- consonants --------------------------------------------------
    "b": "b", "d": "d", "f": "f", "g": "g", "ɡ": "g",
    "h": "h", "k": "k", "l": "l", "m": "m", "n": "n",
    "p": "p", "s": "s", "t": "t", "v": "v", "w": "w",
    "z": "z",
    "j": "y",        # IPA /j/ is English "y" sound, not the "j" in "jump"
    "ɹ": "r", "r": "r",
    "ŋ": "ng",
    "ʃ": "sh", "ʒ": "zh",
    "θ": "th", "ð": "th",  # English spelling doesn't distinguish voicing
    "tʃ": "ch",
    "dʒ": "j", "d͡ʒ": "j",  # the consonant in "jump"
    "ɾ": "t",        # flap t (as in "butter") — show as plain t for kids
    # ---- ARPA-39 (two-letter codes, from base-arpa39 model) ----------
    # Long vowels
    "iy": "ē",  "uw": "ū",
    "ey": "ā",  "ay": "ī",  "ow": "ō",  "oy": "oy",  "aw": "ow",
    # Short vowels
    "ih": "ĭ",  "eh": "ĕ",  "ae": "ă",  "ah": "ŭ",
    "uh": "oo", "aa": "ŏ",  "ao": "aw",
    # R-colored
    "er": "ər",
    # Consonants (lowercase ARPA → letter or digraph)
    "hh": "h",
    "ng": "ng",
    "sh": "sh", "zh": "zh",
    "th": "th",  "dh": "th",
    "ch": "ch", "jh": "j",
    # Single-letter ARPA tokens already map naturally (b, d, f, g, k, l,
    # m, n, p, r, s, t, v, w, y, z) — the dict-fallthrough returns them
    # unchanged, which is what we want.
}


def simple_label(ipa: str) -> str:
    """Return an elementary-school-style label for an IPA phoneme.

    Falls back to the raw IPA when we don't have a mapping — so rare
    phonemes still render, just in their IPA form."""
    return SIMPLE_LABEL.get(ipa.translate(_STRIP), ipa)

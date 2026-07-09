"""The fixed, rule-based, deterministic cleaning pipeline.

Pure functions only: every function takes text (or a document) and returns a new
value; nothing is mutated in place. The same chain runs in the Phase 0 smoke test
and in the Phase 1 at-scale clean, so what we eyeball on 10 docs is exactly what
runs on billions of tokens.

Order is cheapest-check-first: a drop ends the chain, and every drop is attributed
to exactly one reason so the Phase 1 report can tally them.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from config import CLEAN

# --------------------------------------------------------------------------- #
# Boilerplate line patterns (whole-line matches, case-insensitive)
# --------------------------------------------------------------------------- #

_BOILERPLATE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^\s*form\s+10[-\s]?[kq]\b.*$",          # SEC form headers
        r"^\s*page\s+\d+\s+of\s+\d+\s*$",          # Page N of M
        r"^\s*table\s+of\s+contents\s*$",
        r"^\s*/s/\s*.*$",                          # /s/ electronic signatures
        r"^\s*all\s+rights\s+reserved.*$",
        r"^\s*united\s+states\s+securities\s+and\s+exchange\s+commission\s*$",
        r"^\s*securities\s+and\s+exchange\s+commission\s*$",
        r"^\s*washington,?\s+d\.?\s?c\.?\s+\d{5}\s*$",
        r"^\s*\[?\s*x\s*\]?\s*$",                  # lone checkbox lines
    )
)

_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-z]+")
_ALNUM = re.compile(r"[A-Za-z0-9]")


@dataclass(frozen=True)
class CleanResult:
    """Immutable outcome of running one document through the chain."""

    kept: bool
    text: str          # cleaned text (empty when dropped)
    reason: str        # "kept" or the drop reason
    raw_chars: int
    clean_chars: int


# --------------------------------------------------------------------------- #
# Step 1: per-line filter + whitespace collapse
# --------------------------------------------------------------------------- #


def _nonalnum_ratio(line: str) -> float:
    if not line:
        return 1.0
    alnum = sum(1 for c in line if _ALNUM.match(c))
    return 1.0 - alnum / len(line)


def filter_lines(text: str) -> str:
    """Drop short / mostly-symbol lines and collapse internal whitespace."""
    out: list[str] = []
    for raw in text.splitlines():
        line = _WHITESPACE.sub(" ", raw).strip()
        if len(line) < CLEAN.min_line_chars:
            continue
        if _nonalnum_ratio(line) > CLEAN.max_nonalnum_ratio:
            continue
        out.append(line)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Step 2: strip known boilerplate lines
# --------------------------------------------------------------------------- #


def strip_boilerplate(text: str) -> str:
    """Delete whole lines matching any known boilerplate regex."""
    return "\n".join(
        line
        for line in text.splitlines()
        if not any(p.match(line) for p in _BOILERPLATE_PATTERNS)
    )


# --------------------------------------------------------------------------- #
# Step 4: repetition test (top-K n-grams dominate)
# --------------------------------------------------------------------------- #


def is_repetitive(text: str) -> bool:
    """True if the top-K n-grams cover more than the allowed fraction."""
    words = text.split()
    n = CLEAN.ngram_n
    if len(words) < n * 2:
        return False
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return False
    counts = Counter(grams)
    top = sum(c for _, c in counts.most_common(CLEAN.repetition_top_k))
    return top / len(grams) > CLEAN.max_repetition_ratio


# --------------------------------------------------------------------------- #
# Step 5: English detection (langdetect, ASCII-ratio fallback)
# --------------------------------------------------------------------------- #


def _ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) < 128) / len(text)


def is_english(text: str) -> bool:
    """Cheap-first English check.

    ASCII ratio decides the easy cases for free (near-pure-ASCII text is English,
    very-non-ASCII text is not); ``langdetect`` is only paid for the ambiguous
    middle band. This keeps the expensive per-doc detector off the hot path for
    the hundreds of thousands of clean English web/legal docs, where it was the
    Phase 1 throughput bottleneck.
    """
    sample = text[: CLEAN.lang_sample_chars]
    ratio = _ascii_ratio(sample)
    if ratio >= 0.99:
        return True   # near-pure ASCII -> English, no detector needed
    if ratio < 0.90:
        return False  # heavily non-ASCII -> not English
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0  # deterministic
        return detect(sample) == "en"
    except Exception:
        return ratio > 0.95


# --------------------------------------------------------------------------- #
# Step 6: strict OCR gate (dictionary-based, scanned sources only)
# --------------------------------------------------------------------------- #

_OCR_TOKEN = re.compile(r"[A-Za-z]{3,}")
_ENGLISH_WORDS: frozenset[str] | None = None  # lazily loaded, cached per process


def _english_words() -> frozenset[str]:
    """Load the system English wordlist once (cached). Empty set if unavailable."""
    global _ENGLISH_WORDS
    if _ENGLISH_WORDS is None:
        try:
            with open(CLEAN.dict_path, encoding="utf-8", errors="ignore") as fh:
                _ENGLISH_WORDS = frozenset(
                    w.strip().lower() for w in fh if w.strip().isalpha()
                )
        except OSError:
            _ENGLISH_WORDS = frozenset()  # no wordlist -> gate becomes a no-op
    return _ENGLISH_WORDS


def nonword_ratio(text: str) -> float:
    """Fraction of alphabetic word-tokens (len>=3) not in the English wordlist."""
    words = _english_words()
    if not words:
        return 0.0
    toks = [t.lower() for t in _OCR_TOKEN.findall(text)]
    if len(toks) < CLEAN.ocr_min_tokens:
        return 0.0
    nonword = sum(1 for t in toks if t not in words)
    return nonword / len(toks)


def is_ocr_garble(text: str) -> bool:
    """True if the doc's non-dictionary-word ratio exceeds the configured max."""
    return nonword_ratio(text) > CLEAN.nonword_ratio_max


# --------------------------------------------------------------------------- #
# The full chain
# --------------------------------------------------------------------------- #


def clean_document(text: str, *, strict_ocr: bool = False) -> CleanResult:
    """Run one document through the full deterministic chain.

    Returns an immutable CleanResult carrying the outcome and the single drop
    reason (or "kept"). Pure: does not touch the input.
    """
    raw_chars = len(text)

    step1 = filter_lines(text)
    step2 = strip_boilerplate(step1)

    if len(step2) < CLEAN.min_doc_chars:
        return CleanResult(False, "", "too_short", raw_chars, len(step2))

    if is_repetitive(step2):
        return CleanResult(False, "", "repetitive", raw_chars, len(step2))

    if not is_english(step2):
        return CleanResult(False, "", "non_english", raw_chars, len(step2))

    if strict_ocr and is_ocr_garble(step2):
        return CleanResult(False, "", "ocr", raw_chars, len(step2))

    return CleanResult(True, step2, "kept", raw_chars, len(step2))

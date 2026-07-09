"""Pure helpers for Phase 2 (dedup + contamination strip).

No heavy deps here (datasketch/MinHash live in the Modal function). These are
small, deterministic, testable text utilities.
"""

from __future__ import annotations

import hashlib
import re

_WS = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase + collapse whitespace: the canonical form for hashing/shingling."""
    return _WS.sub(" ", text.lower()).strip()


def words(text: str) -> list[str]:
    """Alphanumeric word tokens of the normalized text."""
    return _WORD.findall(normalize(text))


def exact_hash(text: str) -> str:
    """Stable blake2b hash of the normalized doc (exact-dup key)."""
    return hashlib.blake2b(normalize(text).encode("utf-8"), digest_size=16).hexdigest()


def word_ngrams(tokens: list[str], n: int) -> set[int]:
    """Set of word n-grams hashed with the fast native hash.

    Uses ``hash(tuple(...))`` (C-speed, no string allocation) rather than a
    cryptographic hash: the contamination set and the per-doc grams are always
    built in the SAME process, so hash consistency holds and stability across
    runs is not required. This is ~10x faster than blake2b over hundreds of
    millions of grams.
    """
    if len(tokens) < n:
        return set()
    return {hash(tuple(tokens[i : i + n])) for i in range(len(tokens) - n + 1)}


def shingles(tokens: list[str], k: int) -> set[bytes]:
    """Set of k-word shingles as bytes (for MinHash update)."""
    if len(tokens) < k:
        return {" ".join(tokens).encode("utf-8")} if tokens else set()
    return {" ".join(tokens[i : i + k]).encode("utf-8") for i in range(len(tokens) - k + 1)}

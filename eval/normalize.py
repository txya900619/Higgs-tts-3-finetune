# coding=utf-8
"""Text normalization for WER/CER, shared by every system under comparison.

Lowercase, strip a fixed five-character punctuation whitelist, collapse
whitespace. Deliberately NOT a blanket Unicode punctuation strip (the shape
`transformers`' BasicTextNormalizer takes), because two characters that look
like punctuation carry meaning in this orthography and would be silently
corrupted:

  ":"  marks vowel length in several dialects (Amis, Saisiyat) -- `bae:iw`,
       `sapi:ihin`. Across the 284,606 sentences in this corpus it sits
       between two letters 79.7% of the time (6,302 of 7,903).
  "⌃"  (U+2303) is an orthographic marker; the FormoG2P table maps it to the
       glottal stop ʔ, so it is a phoneme rather than punctuation even in the
       87.5% of cases where it is not strictly word-internal.

The five that are stripped were each confirmed to occur mid-word exactly zero
times in the same scan, and no other non-letter, non-space character appears
in the corpus at all. The apostrophe needs no normalization either: all
176,655 occurrences are U+02BC MODIFIER LETTER APOSTROPHE, which Unicode
classes as a letter, and no variant form appears.

Re-run `scripts/eval/scan_punctuation.py` if the corpus ever changes.
"""

import re

STRIPPED_PUNCTUATION = ".,!?;"

_STRIP_RE = re.compile(f"[{re.escape(STRIPPED_PUNCTUATION)}]")
_SPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, drop the five safe punctuation marks, collapse whitespace."""
    text = text.lower()
    text = _STRIP_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()

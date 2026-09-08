# coding=utf-8
"""Orthography -> IPA for Formosan languages, for the F5-TTS baseline.

Follows the reference implementation in hungshinlee/formospeech-data
(`g2p/formosan/g2p_formosan.py`) rather than the older copy embedded in the
ILRDF/formosan-f5-tts Space, which differs in three ways that all matter:

  - `EXTRA_FORMOSAN_G2P` supplies a language-independent fallback for
    graphemes a dialect's own row does not define (c->ʦ, y->j, j->ɟ, the rest
    to themselves). The Space has no such table and simply discards any
    sentence containing one, which threw away 1.43% of this evaluation set --
    a single loanword or place name (`kognkoan`, `sakizaya`) took down a
    sentence whose other seventeen words converted cleanly.
  - Kavalan maps BOTH `R` and `r` to `R`; the Space preserved only uppercase,
    so lowercase r became an unmapped character.
  - Keys are matched longest-first. The Space iterates the dict in CSV column
    order and takes the first hit, so "ng" (listed after "n") gets split into
    n + g.

`load_g2p_from_csv` registers every row under its lang_tag ("阿美_南勢"), its
full name ("南勢阿美語") and its lang_code ("ami-x-iams"), so callers can index
however they like and no dialect map is needed.

Note the reference `convert_to_ipa` joins phones within a word with "-",
producing the NeMo phone-tokenizer format the corpus's `ipa` column is stored
in. `join_char` defaults to "" here because downstream consumers want the
plain phone sequence, which is what `common.strip_ipa_dashes` recovers.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional

G2P_CSV = Path(__file__).with_name("g2p_new.csv")
G2P_CSV_LEGACY = Path(__file__).with_name("g2p_250402.csv")

END_PUNCTUATIONS = ["!", "?", ".", ";", ","]

META_KEYS = {"Language", "Dialect", "subgroup", "subgroup_eng", "lang_code", "language"}

# Applied to every language for graphemes its own row leaves undefined.
EXTRA_FORMOSAN_G2P = {
    "z": "z", "o": "o", "h": "h", "g": "g", "y": "j", "w": "w",
    "c": "ʦ", "u": "u", "f": "f", "v": "v", "j": "ɟ", "b": "b",
    "q": "q", "e": "e", "l": "l", "d": "d",
}

KEEP_CASE_SAISIYAT = {"xsy", "賽夏"}
KEEP_CASE_KAVALAN = {"ckv", "噶瑪蘭"}


def load_g2p_from_csv(
    g2p_path: Path = G2P_CSV, strip_values: bool = True,
) -> dict[str, dict[str, list[str]]]:
    """Load the table, indexed by lang_tag, full name and lang_code alike.

    `strip_values` trims whitespace off each phone. It matters for exactly one
    column: 15 rows of the newer table give `ʼ` the value "ʔ " with a trailing
    space, against a bare "ʔ" in ten other dialects and no whitespace anywhere
    else in 2,300+ cells. That is a data-entry slip, and it has to be removed
    at load time -- once conversion has run, the stray space is
    indistinguishable from a real word boundary.

    The slip reached the published corpora: `formospeech/*`'s own `ipa` column
    carries it, so 23,978 of 284,606 materialised rows (8.42%, all in the 15
    Atayal/Bunun/Paiwan dialects whose row has the trailing space) are damaged
    two ways -- 18,768 have a word split in half (`maʼun` -> `maʔ un`) and
    5,210 have a space before punctuation (`taknaʼ.` -> `taknaʔ .`).

    Hence the default. Pass `strip_values=False` only to reproduce the corpus
    column as published: with that setting `text_to_ipa` matches it on all
    284,606 rows byte-for-byte, and with the default it differs from it in
    whitespace alone (zero non-whitespace differences), which is what makes
    regenerating `ipa` from `text` a safe repair rather than a re-transcription.
    """
    g2p: dict[str, dict[str, list[str]]] = {}
    with open(g2p_path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            language = (row.get("Language") or "").strip()
            dialect = (row.get("Dialect") or "-").strip()
            lang_tag = language if dialect in ("-", "") else f"{language}_{dialect}"

            mapping: dict[str, list[str]] = {}
            for key, value in row.items():
                if not key or key in META_KEYS or value in (None, "", "-"):
                    continue
                parts = value.split(",")
                mapping[key] = [v.strip() for v in parts] if strip_values else parts
            for grapheme, phone in EXTRA_FORMOSAN_G2P.items():
                mapping.setdefault(grapheme, [phone])

            # Longest key first, so "ng" wins over "n".
            mapping = dict(sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True))

            for alias in (lang_tag, row.get("language"), row.get("lang_code")):
                if alias:
                    g2p[alias.strip()] = mapping

    # 臺/台 spelling varies between sources.
    for a, b in (("魯凱_霧台", "魯凱_霧臺"), ("霧臺魯凱語", "霧台魯凱語")):
        if a in g2p and b not in g2p:
            g2p[b] = g2p[a]
        if b in g2p and a not in g2p:
            g2p[a] = g2p[b]
    return g2p


g2p_object = load_g2p_from_csv()
g2p_object_legacy = load_g2p_from_csv(G2P_CSV_LEGACY)
# Reproduces the published `ipa` column, stray spaces and all. Only for
# provenance checks against the corpus -- never for generating training text.
g2p_object_corpus = load_g2p_from_csv(strip_values=False)


def lower_formosan_text(raw_text: str, language: str) -> str:
    text = list(raw_text.strip())
    if language in KEEP_CASE_SAISIYAT:
        for i, char in enumerate(text):
            if char == "S":
                if i == 0:
                    text[i] = char.lower()
            else:
                text[i] = char.lower()
    elif language in KEEP_CASE_KAVALAN:
        for i, char in enumerate(text):
            text[i] = "R" if char in ("R", "r") else char.lower()
    else:
        text = [c.lower() for c in text]
    return "".join(text)


def replace_to_list(text: str, g2p: dict[str, list[str]]) -> tuple[list, set]:
    """Greedy longest-match grapheme -> phone replacement."""
    result: list[str] = []
    buffer = ""
    oovs: set[str] = set()

    i = 0
    while i < len(text):
        match = next((k for k in g2p if text.startswith(k, i)), None)
        if match is not None:
            if buffer:
                result.append(buffer)
                buffer = ""
            result.append(g2p[match][0])
            i += len(match)
        else:
            buffer += text[i]
            oovs.add(text[i])
            i += 1

    if buffer:
        result.append(buffer)
    return result, oovs


def convert_to_ipa(
    text: str, g2p: dict[str, list[str]], join_char: str = "",
) -> tuple[Optional[str], list]:
    result_list: list[str] = []
    oovs_to_ipa: set[str] = set()

    for word in text.split():
        ending_punct = ""
        if word and word[-1] in END_PUNCTUATIONS:
            ending_punct = word[-1]
            word = word[:-1]

        ipa_list, oovs = replace_to_list(word, g2p)
        oovs_to_ipa.update(oovs)
        result_list.append(join_char.join(ipa_list) + ending_punct)

    if not result_list:
        return None, sorted(oovs_to_ipa)
    return " ".join(result_list), sorted(oovs_to_ipa)


def text_to_ipa(text: str, language: str, table: dict | None = None) -> str:
    """Convert one sentence. `language` may be a lang_tag, full name or lang_code.

    """
    table = table if table is not None else g2p_object
    if language not in table:
        raise KeyError(f"No g2p entry for {language!r}")

    text = lower_formosan_text(text, language)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"\-\“\”]", "", text)
    text = re.sub(r"[\ʼ\’\']", "ʼ", text)
    text = text.replace("^", "⌃")

    ipa, _oovs = convert_to_ipa(text, table[language])
    if ipa is None:
        raise ValueError(f"Nothing to convert for {language}: {text!r}")

    return ipa.replace("ʦ", "t͡s").replace("ʨ", "t͡ɕ").replace("ʤ", "d͡ʒ")

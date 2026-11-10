"""Sinhala text normalization.

Two forms are needed because they do different jobs:

- ``normalize_label`` is what the model trains on. It keeps zero-width joiners
  (U+200D), which Sinhala conjuncts such as "ශ්‍රී" need to display correctly.
- ``normalize_score`` is what gets compared when scoring. It also removes joiners,
  so a model isn't penalized for an invisible character.
"""
import re
import unicodedata

_KEEP = re.compile(r"[^඀-෿‍\s]")  # Sinhala block plus zero-width joiner


def normalize_label(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("‌", "")
    text = _KEEP.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_score(text: str) -> str:
    return normalize_label(text).replace("‍", "")

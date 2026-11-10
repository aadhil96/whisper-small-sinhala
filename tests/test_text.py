import unicodedata

from sinhala_asr.text import normalize_label, normalize_score

SRI = "ශ්‍රී"


def test_label_keeps_zero_width_joiner():
    assert "‍" in normalize_label(f"{SRI} ලංකාව")


def test_score_removes_zero_width_joiner():
    assert "‍" not in normalize_score(f"{SRI} ලංකාව")


def test_zero_width_non_joiner_is_removed():
    assert "‌" not in normalize_label("ල‌ංකාව")


def test_non_sinhala_text_is_removed():
    assert normalize_label("ලංකාව Lanka 2026, හොඳයි!") == "ලංකාව හොඳයි"


def test_whitespace_is_collapsed_and_stripped():
    assert normalize_label("  මම \n\t යනවා  ") == "මම යනවා"


def test_text_with_no_sinhala_becomes_empty():
    assert normalize_label("Hello 123") == ""


def test_output_is_nfc_normalized():
    decomposed = unicodedata.normalize("NFD", "කෝ")
    assert normalize_label(decomposed) == unicodedata.normalize("NFC", "කෝ")

import pytest

from sinhala_asr.metrics import score


def test_perfect_transcription_scores_zero():
    m = score(["මම යනවා"], ["මම යනවා"])
    assert (m["wer"], m["cer"], m["runaway_outputs"], m["n"]) == (0, 0, 0, 1)


def test_one_wrong_word_of_two_is_50_percent_wer():
    assert score(["මම යනවා"], ["මම එනවා"])["wer"] == pytest.approx(50)


def test_cer_counts_characters():
    assert score(["මම යනවා"], ["මම එනවා"])["cer"] == pytest.approx(100 / 7)


def test_zero_width_joiner_difference_is_not_an_error():
    assert score(["ශ්‍රී"], ["ශ්රී"])["cer"] == 0


def test_empty_references_are_skipped():
    assert score(["මම යනවා", "Hello"], ["මම යනවා", "මම"])["n"] == 1


def test_runaway_output_is_counted_and_median_stays_robust():
    refs = ["මම", "මම", "මම"]
    m = score(refs, ["මම", "මම", "මම " * 20])
    assert m["runaway_outputs"] == 1
    assert m["cer"] > 100
    assert m["median_utterance_cer"] == 0

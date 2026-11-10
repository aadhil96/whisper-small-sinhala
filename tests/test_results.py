"""The numbers in the README must be reproducible from the files in this repo."""
import csv
import json
from pathlib import Path

import pytest
from score_predictions import MODELS, score_file

RESULTS = Path(__file__).resolve().parent.parent / "results"
PUBLISHED = json.loads((RESULTS / "results.json").read_text())


@pytest.mark.parametrize("name", MODELS)
def test_scores_recompute_from_saved_predictions(name):
    recomputed = score_file(name)
    for key, value in recomputed.items():
        assert value == pytest.approx(PUBLISHED["test"][name][key], abs=0.01), key


def test_headline_numbers_are_the_ones_in_the_readme():
    assert round(PUBLISHED["test"]["fine-tuned"]["wer"], 2) == 36.47
    assert round(PUBLISHED["test"]["fine-tuned"]["cer"], 2) == 9.13


def test_results_come_from_the_full_run_not_a_smoke_test():
    assert PUBLISHED["training"]["smoke_test"] is False
    assert PUBLISHED["training"]["steps"] == 6000
    assert PUBLISHED["data"]["test_utterances"] == 3000


@pytest.mark.parametrize("name", MODELS)
def test_every_model_was_scored_on_the_same_clips(name):
    expected = json.loads((RESULTS / "test_ids.json").read_text())
    with open(RESULTS / f"predictions_{name}.csv", encoding="utf-8", newline="") as fh:
        assert [row["id"] for row in csv.DictReader(fh)] == expected


def test_training_log_agrees_with_the_best_validation_score():
    history = json.loads((RESULTS / "log_history.json").read_text())
    best = min(entry["eval_cer"] for entry in history if "eval_cer" in entry)
    assert best == pytest.approx(PUBLISHED["training"]["best_val_cer"])

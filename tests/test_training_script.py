"""The Kaggle script runs as one file, so check its copies of shared logic haven't drifted."""
import ast
import re
import unicodedata
from pathlib import Path

import pytest

from sinhala_asr.text import normalize_label, normalize_score

SCRIPT = Path(__file__).resolve().parent.parent / "training" / "train_whisper_sinhala.py"
TREE = ast.parse(SCRIPT.read_text(encoding="utf-8"))


def constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            return ast.literal_eval(node.value)
    raise KeyError(name)


def script_namespace():
    wanted = {"_KEEP", "normalize_label", "normalize_score"}
    nodes = [n for n in TREE.body
             if getattr(n, "name", None) in wanted
             or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in wanted)]
    namespace = {"re": re, "unicodedata": unicodedata}
    exec(compile(ast.Module(nodes, []), str(SCRIPT), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("text", ["ශ්‍රී ලංකාව", "මම Lanka 123 යනවා!", "  ල‌ංකාව \n", "Hello", ""])
def test_script_normalization_matches_the_package(text):
    namespace = script_namespace()
    assert namespace["normalize_label"](text) == normalize_label(text)
    assert namespace["normalize_score"](text) == normalize_score(text)


def test_script_is_configured_for_the_full_run():
    assert constant("SMOKE") is False
    assert constant("MODEL_NAME") == "openai/whisper-small"
    assert constant("LR") == 1e-5
    assert constant("BATCH_SIZE") == 16

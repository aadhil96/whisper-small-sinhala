"""Recompute the published test scores from the saved predictions in results/."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sinhala_asr.metrics import score  # noqa: E402

MODELS = ["fine-tuned", "whisper-large-v3", "whisper-small"]


def score_file(name):
    with open(ROOT / "results" / f"predictions_{name}.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return score([r["reference"] for r in rows], [r["hypothesis"] for r in rows])


if __name__ == "__main__":
    published = json.loads((ROOT / "results" / "results.json").read_text())["test"]
    print(f"{'model':18} {'WER':>8} {'CER':>8} {'median CER':>11} {'runaway':>8} {'n':>6}   matches results.json")
    for name in MODELS:
        m = score_file(name)
        same = all(abs(m[k] - published[name][k]) < 0.01 for k in m)
        print(f"{name:18} {m['wer']:8.2f} {m['cer']:8.2f} {m['median_utterance_cer']:11.2f} "
              f"{m['runaway_outputs']:8d} {m['n']:6d}   {same}")

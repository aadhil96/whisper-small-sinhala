"""WER and CER scoring, matching what the training run reported."""
import statistics

import jiwer

from sinhala_asr.text import normalize_score


def score(refs, hyps):
    """Score hypotheses against references; empty references are skipped.

    A model stuck repeating itself can push WER and CER far past 100%, so this also
    reports the typical utterance and how often that happened.
    """
    pairs = [(normalize_score(r), normalize_score(h)) for r, h in zip(refs, hyps)]
    pairs = [(r, h) for r, h in pairs if r]
    refs, hyps = [r for r, _ in pairs], [h for _, h in pairs]
    return {
        "wer": 100 * jiwer.wer(refs, hyps),
        "cer": 100 * jiwer.cer(refs, hyps),
        "median_utterance_cer": 100 * statistics.median(jiwer.cer(r, h) for r, h in pairs),
        "runaway_outputs": sum(len(h) > 3 * len(r) for r, h in pairs),
        "n": len(refs),
    }

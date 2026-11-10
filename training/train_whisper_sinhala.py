"""Fine-tune whisper-small for Sinhala on OpenSLR SLR52, then score it against baselines.

Adapted from sinhala_asr_whisper_kaggle.ipynb. Differences from the notebook:
  - Audio features are computed on the fly, so the cache doesn't fill Kaggle's disk.
  - Zero-width joiners stay in the training labels. Sinhala needs them for conjuncts
    such as the one in "Sri", so removing them would make the model output broken text.
    They are removed only for scoring, which keeps every model's comparison fair.
  - Every model is scored on the same fixed test utterances with the same code.
  - Training stops on a time limit, so results are saved before Kaggle ends the session.
"""
import csv
import glob
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SMOKE = False  # short end-to-end test; set to False for the real run

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # one GPU, even on a T4 x2 machine
os.environ["USE_TF"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "transformers>=4.46,<5", "accelerate", "jiwer", "soundfile", "librosa"])

import jiwer
import librosa
import soundfile as sf
import torch
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback,
                          WhisperForConditionalGeneration, WhisperProcessor)

START = time.time()
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------- CONFIG -----------------------------
MODEL_NAME = "openai/whisper-small"
BASELINES = ["openai/whisper-small", "openai/whisper-large-v3"]
LANGUAGE = "sinhala"
TASK = "transcribe"

NUM_ZIP_PARTS = 1 if SMOKE else 16  # hex parts 0..f, ~0.9 GB each
MAX_AUDIO_SEC = 20.0
MAX_LABEL_TOKENS = 440              # Whisper's decoder has 448 positions
MAX_NEW_TOKENS = 400

MAX_STEPS = 20 if SMOKE else 6000
EVAL_STEPS = 10 if SMOKE else 1000
BATCH_SIZE = 16
LR = 1e-5
VAL_EVAL_SIZE = 32 if SMOKE else 1000
TEST_EVAL_SIZE = 64 if SMOKE else 3000

# Conservative limits: outputs may be lost if Kaggle ends the session mid-run.
DOWNLOAD_BUDGET_H = 2.25  # stop downloading after this and use the parts we have
TRAIN_DEADLINE_H = 7.0    # stop training here, leaving time to score and save
BASELINE_DEADLINE_H = 8.0 # don't start another baseline after this
SCORING_DEADLINE_H = 8.75 # abandon any scoring still running at this point
MIN_FREE_GB = 5

DATA_DIR = "/tmp/openslr52"         # not /kaggle/working, which has a 20 GB limit
WORK_DIR = "/kaggle/working"
OUTPUT_DIR = f"{WORK_DIR}/whisper-sinhala"
CKPT_DIR = f"{WORK_DIR}/checkpoints"
BASE_URL = "https://www.openslr.org/resources/52"
os.makedirs(DATA_DIR, exist_ok=True)
# ------------------------------------------------------------------


def log(msg):
    print(f"[{(time.time() - START) / 3600:5.2f}h] {msg}", flush=True)


def hours():
    return (time.time() - START) / 3600


def free_gb():
    return shutil.disk_usage(DATA_DIR).free / 1e9


def fetch(url, dest):
    return subprocess.call(["wget", "-q", "-c", "--tries=5", "--timeout=60", url, "-O", dest]) == 0


log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
if not torch.cuda.is_available():
    raise SystemExit("No GPU available; stopping before any work is wasted.")

# --- download ---
tsv_path = f"{DATA_DIR}/utt_spk_text.tsv"
if not fetch(f"{BASE_URL}/utt_spk_text.tsv", tsv_path):
    raise SystemExit("Could not download the transcript table.")

parts_used = []
for part in "0123456789abcdef"[:NUM_ZIP_PARTS]:
    if hours() > DOWNLOAD_BUDGET_H or free_gb() < MIN_FREE_GB:
        log(f"Stopping downloads at part {part}: {free_gb():.0f} GB free")
        break
    zip_path = f"{DATA_DIR}/asr_sinhala_{part}.zip"
    if not fetch(f"{BASE_URL}/asr_sinhala_{part}.zip", zip_path):
        log(f"Part {part} failed to download; skipping it")
        continue
    if subprocess.call(["unzip", "-q", "-o", zip_path, "-d", DATA_DIR]) != 0:
        log(f"Part {part} failed to extract; skipping it")
        continue
    os.remove(zip_path)
    parts_used.append(part)
    log(f"Part {part} ready; {free_gb():.0f} GB free")
if not parts_used:
    raise SystemExit("No audio parts were downloaded.")


# --- text normalization ---
_KEEP = re.compile(r"[^඀-෿‍\s]")  # Sinhala block plus zero-width joiner


def normalize_label(text):
    text = unicodedata.normalize("NFC", text).replace("‌", "")
    text = _KEEP.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_score(text):
    return normalize_label(text).replace("‍", "")


# --- manifest ---
id2path = {Path(f).stem: f for f in glob.glob(f"{DATA_DIR}/**/*.flac", recursive=True)}
log(f"FLAC files: {len(id2path)}")

rows = []
with open(tsv_path, encoding="utf-8") as fh:
    for line in fh:
        cols = line.rstrip("\n").split("\t")
        if len(cols) >= 3 and cols[0] in id2path:
            text = normalize_label(cols[2])
            if text:
                rows.append({"id": cols[0], "speaker": cols[1], "path": id2path[cols[0]], "text": text})
matched = len(rows)


def duration(row):
    try:
        return sf.info(row["path"]).duration
    except Exception:
        return None


with ThreadPoolExecutor(8) as pool:
    durations = list(pool.map(duration, rows))
rows = [dict(r, seconds=d) for r, d in zip(rows, durations) if d is not None and d <= MAX_AUDIO_SEC]

processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
lengths = processor.tokenizer([r["text"] for r in rows]).input_ids
rows = [r for r, ids in zip(rows, lengths) if len(ids) <= MAX_LABEL_TOKENS]
log(f"Utterances: {matched} matched, {len(rows)} kept; "
    f"{sum(chr(0x200D) in r['text'] for r in rows)} labels contain a zero-width joiner")

# --- speaker-disjoint split, 90 / 5 / 5 ---
speakers = sorted({r["speaker"] for r in rows})
random.shuffle(speakers)
n_held_out = max(1, int(0.05 * len(speakers)))
val_speakers = set(speakers[:n_held_out])
test_speakers = set(speakers[n_held_out:2 * n_held_out])
train_rows = [r for r in rows if r["speaker"] not in val_speakers | test_speakers]
val_rows = [r for r in rows if r["speaker"] in val_speakers]
test_rows = [r for r in rows if r["speaker"] in test_speakers]
val_rows = random.sample(val_rows, min(VAL_EVAL_SIZE, len(val_rows)))
test_rows = random.sample(test_rows, min(TEST_EVAL_SIZE, len(test_rows)))

train_text = {normalize_score(r["text"]) for r in train_rows}
test_seen = sum(normalize_score(r["text"]) in train_text for r in test_rows)
data_summary = {
    "parts_used": parts_used,
    "speakers_total": len(speakers),
    "train_utterances": len(train_rows),
    "train_hours": round(sum(r["seconds"] for r in train_rows) / 3600, 1),
    "val_utterances": len(val_rows),
    "test_utterances": len(test_rows),
    "test_sentences_also_in_train": test_seen,
}
log(json.dumps(data_summary))
with open(f"{WORK_DIR}/test_ids.json", "w") as fh:
    json.dump([r["id"] for r in test_rows], fh)


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return librosa.resample(audio, orig_sr=sr, target_sr=16000) if sr != 16000 else audio


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, items, proc, with_labels=True):
        self.items, self.proc, self.with_labels = items, proc, with_labels

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        try:
            audio = load_audio(item["path"])
        except Exception:
            return self[(idx + 1) % len(self.items)]
        example = {"input_features": self.proc.feature_extractor(
            audio, sampling_rate=16000).input_features[0]}
        if self.with_labels:
            example["labels"] = self.proc.tokenizer(item["text"]).input_ids
        return example


@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: Any

    def __call__(self, features):
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # remove the prepended start-of-transcript token; the model adds it back
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def score(refs, hyps):
    pairs = [(normalize_score(r), normalize_score(h)) for r, h in zip(refs, hyps)]
    pairs = [(r, h) for r, h in pairs if r]
    refs, hyps = [r for r, _ in pairs], [h for _, h in pairs]
    # A model stuck repeating itself can push WER and CER far past 100%, so also report
    # the typical utterance and how often that happened.
    return {"wer": 100 * jiwer.wer(refs, hyps), "cer": 100 * jiwer.cer(refs, hyps),
            "median_utterance_cer": 100 * statistics.median(jiwer.cer(r, h) for r, h in pairs),
            "runaway_outputs": sum(len(h) > 3 * len(r) for r, h in pairs), "n": len(refs)}


def compute_metrics(pred):
    pad = processor.tokenizer.pad_token_id
    pred_ids, label_ids = pred.predictions, pred.label_ids
    pred_ids[pred_ids == -100] = pad
    label_ids[label_ids == -100] = pad
    hyps = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    refs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    metrics = score(refs, hyps)
    return {"wer": metrics["wer"], "cer": metrics["cer"]}


class TimeLimit(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if hours() > TRAIN_DEADLINE_H and not control.should_training_stop:
            log(f"Time limit reached at step {state.global_step}; stopping training")
            control.should_training_stop = True
            control.should_evaluate = True
            control.should_save = True
        return control


# --- train ---
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.generation_config.language = LANGUAGE
model.generation_config.task = TASK
model.generation_config.forced_decoder_ids = None
model.generation_config.use_cache = True
model.config.forced_decoder_ids = None
model.config.use_cache = False  # required with gradient checkpointing

training_args = Seq2SeqTrainingArguments(
    output_dir=CKPT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    warmup_steps=2 if SMOKE else 400,
    max_steps=MAX_STEPS,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    fp16=True,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=EVAL_STEPS,
    logging_steps=5 if SMOKE else 50,
    predict_with_generate=True,
    generation_max_length=MAX_NEW_TOKENS,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    remove_unused_columns=False,
    label_names=["labels"],
    dataloader_num_workers=3,
    report_to="none",
    seed=SEED,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=SpeechDataset(train_rows, processor),
    eval_dataset=SpeechDataset(val_rows, processor),
    data_collator=DataCollatorSpeechSeq2Seq(processor),
    compute_metrics=compute_metrics,
    processing_class=processor,
    callbacks=[TimeLimit()],
)
log("Training starts")
trainer.train()
log(f"Training done at step {trainer.state.global_step}; best CER {trainer.state.best_metric}")

# --- save before scoring, so a later failure can't lose the model ---
model.config.use_cache = True
model.gradient_checkpointing_disable()
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
with open(f"{WORK_DIR}/log_history.json", "w") as fh:
    json.dump(trainer.state.log_history, fh, indent=1)
shutil.rmtree(CKPT_DIR, ignore_errors=True)
log(f"Saved model to {OUTPUT_DIR}: {sorted(os.listdir(OUTPUT_DIR))}")


# --- score every model on the same utterances ---
def evaluate_model(net, proc, name, batch_size):
    net = net.to("cuda").eval()
    loader = torch.utils.data.DataLoader(
        SpeechDataset(test_rows, proc, with_labels=False), batch_size=batch_size, num_workers=3,
        collate_fn=lambda fs: proc.feature_extractor.pad(fs, return_tensors="pt"))
    hyps = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for batch in loader:
            if hours() > SCORING_DEADLINE_H:
                raise TimeoutError(f"{name} ran out of time after {len(hyps)} utterances")
            ids = net.generate(input_features=batch["input_features"].to("cuda"),
                               language=LANGUAGE, task=TASK, max_new_tokens=MAX_NEW_TOKENS)
            hyps += proc.tokenizer.batch_decode(ids, skip_special_tokens=True)
    refs = [r["text"] for r in test_rows]
    with open(f"{WORK_DIR}/predictions_{name}.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "reference", "hypothesis"])
        writer.writerows(zip([r["id"] for r in test_rows], refs, hyps))
    metrics = score(refs, hyps)
    log(f"{name}: {json.dumps(metrics)}")
    return metrics


results = {
    "data": data_summary,
    "training": {"model": MODEL_NAME, "steps": trainer.state.global_step, "batch_size": BATCH_SIZE,
                 "learning_rate": LR, "best_val_cer": trainer.state.best_metric, "smoke_test": SMOKE},
    "test": {},
}


def write_results():
    with open(f"{WORK_DIR}/results.json", "w") as fh:
        json.dump(results, fh, indent=1)


del trainer, model
torch.cuda.empty_cache()

# The fine-tuned model is reloaded from disk, so this tests the files we would publish.
for name, source in [("fine-tuned", OUTPUT_DIR)] + [(b.split("/")[-1], b) for b in BASELINES]:
    if hours() > BASELINE_DEADLINE_H:
        log(f"Skipping {name}: out of time")
        results["test"][name] = {"error": "skipped: out of time"}
        write_results()
        continue
    try:
        eval_proc = WhisperProcessor.from_pretrained(source)
        eval_model = WhisperForConditionalGeneration.from_pretrained(source)
        results["test"][name] = evaluate_model(eval_model, eval_proc, name, BATCH_SIZE)
        del eval_model
    except Exception as exc:  # the trained model is already safe; record this and carry on
        log(f"Scoring {name} failed: {exc}")
        results["test"][name] = {"error": str(exc)}
    write_results()
    torch.cuda.empty_cache()

log("ALL DONE " + json.dumps(results["test"]))

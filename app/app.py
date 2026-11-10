"""Gradio demo for the fine-tuned Sinhala Whisper model. Run with: python app/app.py"""
import os
from pathlib import Path

import gradio as gr
import librosa
import torch
from transformers import pipeline

MODEL_ID = os.environ.get("MODEL_ID", "Aadhil96/whisper-small-sinhala")
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

DESCRIPTION = f"""
Record or upload Sinhala speech to get a transcription from
[whisper-small-sinhala](https://huggingface.co/{MODEL_ID}), a `whisper-small` model
fine-tuned on the OpenSLR SLR52 corpus.

Output is Sinhala script only, with no digits, Latin text or punctuation.
"""


def load_pipeline():
    return pipeline("automatic-speech-recognition", model=MODEL_ID,
                    device=0 if torch.cuda.is_available() else -1, chunk_length_s=30)


def make_transcriber(asr):
    def transcribe(audio_path):
        if not audio_path:
            return "Please record or upload some audio first."
        try:
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as exc:
            raise gr.Error("Could not read that audio file. Please try a WAV, FLAC or MP3.") from exc
        result = asr({"raw": audio, "sampling_rate": 16000},
                     generate_kwargs={"language": "sinhala", "task": "transcribe"})
        return result["text"].strip()

    return transcribe


def build_demo(asr):
    examples = [[str(path)] for path in sorted(EXAMPLES_DIR.glob("*.flac"))]
    return gr.Interface(
        fn=make_transcriber(asr),
        inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="Sinhala audio"),
        outputs=gr.Textbox(label="Transcription", lines=4),
        examples=examples or None,
        title="Sinhala Speech-to-Text",
        description=DESCRIPTION,
        article="Example clips are from held-out test speakers in "
                "[OpenSLR SLR52](https://www.openslr.org/52/) (CC BY-SA 4.0).",
        flagging_mode="never",
        api_name="predict",
    )


if __name__ == "__main__":
    build_demo(load_pipeline()).launch()

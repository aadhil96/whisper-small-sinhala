"""Gradio app tests with a stand-in model, so they run quickly and without downloads."""
from pathlib import Path

import gradio as gr
import numpy as np
import pytest
import soundfile as sf
from gradio_client import Client, handle_file

from app import EXAMPLES_DIR, build_demo, make_transcriber

EXAMPLES = sorted(EXAMPLES_DIR.glob("*.flac"))


class FakeASR:
    def __init__(self):
        self.calls = []

    def __call__(self, inputs, generate_kwargs):
        self.calls.append((inputs, generate_kwargs))
        return {"text": "  මම යනවා "}


def test_empty_input_gets_a_friendly_message_and_skips_the_model():
    asr = FakeASR()
    assert "record or upload" in make_transcriber(asr)(None)
    assert asr.calls == []


def test_model_gets_16khz_audio_and_sinhala_settings():
    asr = FakeASR()
    assert make_transcriber(asr)(str(EXAMPLES[0])) == "මම යනවා"
    inputs, generate_kwargs = asr.calls[0]
    assert inputs["sampling_rate"] == 16000
    assert generate_kwargs == {"language": "sinhala", "task": "transcribe"}


def test_stereo_44khz_upload_becomes_mono_16khz(tmp_path):
    path = tmp_path / "stereo.wav"
    sf.write(path, np.zeros((44100, 2), dtype="float32"), 44100)
    asr = FakeASR()
    make_transcriber(asr)(str(path))
    audio = asr.calls[0][0]["raw"]
    assert audio.ndim == 1
    assert len(audio) == pytest.approx(16000, abs=10)


def test_unreadable_file_shows_an_error_instead_of_crashing(tmp_path):
    path = tmp_path / "not_audio.wav"
    path.write_text("this is not audio")
    with pytest.raises(gr.Error):
        make_transcriber(FakeASR())(str(path))


def test_three_example_clips_ship_with_the_app():
    assert [Path(p).stem for p in map(str, EXAMPLES)] == ["8b97848022", "9753dab5f9", "e41cb38d3f"]


@pytest.fixture(scope="module")
def running_demo():
    demo = build_demo(FakeASR())
    _, url, _ = demo.launch(prevent_thread_lock=True, quiet=True, share=False)
    yield url
    demo.close()


def test_server_transcribes_through_the_gradio_api(running_demo):
    client = Client(running_demo, verbose=False)
    assert client.predict(handle_file(str(EXAMPLES[0])), api_name="/predict") == "මම යනවා"


def test_server_handles_a_request_with_no_audio(running_demo):
    client = Client(running_demo, verbose=False)
    assert "record or upload" in client.predict(None, api_name="/predict")

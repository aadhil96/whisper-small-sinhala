"""Tests against the real published model. Downloads about 1 GB: pytest -m slow"""
import pytest
from gradio_client import Client, handle_file

from app import EXAMPLES_DIR, build_demo, load_pipeline, make_transcriber
from sinhala_asr.metrics import score
from sinhala_asr.text import normalize_score

pytestmark = pytest.mark.slow

REFERENCES = {
    "8b97848022": "ඇලෙක්සාර් රෑන්කින් කියන්නෙ හොදම එක නෙමෙයි",
    "9753dab5f9": "එක්තරා උදෑසනක ට පමණ කරන ලද ස්කෑන් එකකින්",
    "e41cb38d3f": "ඔබට දැන් ඉතාම නිවැරැදි පිළිතුරක් ලැබී තිබෙනවා",
}


@pytest.fixture(scope="module")
def asr():
    return load_pipeline()


@pytest.mark.parametrize("clip_id", REFERENCES)
def test_published_model_transcribes_held_out_clips(asr, clip_id):
    text = make_transcriber(asr)(str(EXAMPLES_DIR / f"{clip_id}.flac"))
    print(f"\n{clip_id}\n  REF: {REFERENCES[clip_id]}\n  HYP: {text}")
    assert normalize_score(text), "output should contain Sinhala script"
    assert score([REFERENCES[clip_id]], [text])["cer"] < 30


def test_published_model_works_through_the_gradio_server(asr):
    demo = build_demo(asr)
    _, url, _ = demo.launch(prevent_thread_lock=True, quiet=True, share=False)
    try:
        client = Client(url, verbose=False)
        clip_id = next(iter(REFERENCES))
        text = client.predict(handle_file(str(EXAMPLES_DIR / f"{clip_id}.flac")), api_name="/predict")
        assert score([REFERENCES[clip_id]], [text])["cer"] < 30
    finally:
        demo.close()

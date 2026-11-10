"""Drive the real Gradio page in Chrome with the real model: example, upload and microphone.

The microphone is simulated by Chrome, which loops a held-out clip as its capture device.
It is not a test of a person speaking into a real microphone.
Needs Google Chrome installed. Run with: python scripts/e2e_ui_check.py
"""
import sys
import tempfile
from pathlib import Path

import librosa
import soundfile as sf
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "app")]

from app import EXAMPLES_DIR, build_demo, load_pipeline  # noqa: E402
from sinhala_asr.metrics import score  # noqa: E402

IMAGES = ROOT / "docs" / "images"
UPLOAD_ID, MIC_ID = "9753dab5f9", "e41cb38d3f"
REFERENCES = {
    "8b97848022": "ඇලෙක්සාර් රෑන්කින් කියන්නෙ හොදම එක නෙමෙයි",
    UPLOAD_ID: "එක්තරා උදෑසනක ට පමණ කරන ලද ස්කෑන් එකකින්",
    MIC_ID: "ඔබට දැන් ඉතාම නිවැරැදි පිළිතුරක් ලැබී තිබෙනවා",
}
TIMEOUT_MS = 180_000


def wait_for_transcript(page, previous=""):
    box = page.get_by_label("Transcription")
    page.wait_for_function(
        "([el, old]) => el.value.trim() && el.value !== old",
        arg=[box.element_handle(), previous], timeout=TIMEOUT_MS)
    return box.input_value().strip()


def report(flow, clip_id, text):
    cer = score([REFERENCES[clip_id]], [text])["cer"]
    print(f"{flow:10} CER {cer:5.1f}%  HYP: {text}")
    return cer


def main():
    demo = build_demo(load_pipeline())
    _, url, _ = demo.launch(prevent_thread_lock=True, quiet=True, share=False)
    results = {}

    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
        mic_audio = Path(tmp) / "mic.wav"
        audio, _ = librosa.load(EXAMPLES_DIR / f"{MIC_ID}.flac", sr=48000, mono=True)
        sf.write(mic_audio, audio, 48000, subtype="PCM_16")
        duration = len(audio) / 48000

        browser = p.chromium.launch(channel="chrome", headless=True, args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-audio-capture={mic_audio}"])
        context = browser.new_context(viewport={"width": 1180, "height": 820},
                                      permissions=["microphone"])
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(url)
        page.get_by_label("Transcription").wait_for()
        page.get_by_role("button", name="Submit").wait_for()

        # 1. A built-in example clip.
        page.locator(".gallery-item, button.example, table.gr-samples-table tr").first.click()
        page.wait_for_timeout(1500)
        page.get_by_role("button", name="Submit").click()
        text = wait_for_transcript(page)
        results["example"] = report("example", "8b97848022", text)
        page.screenshot(path=IMAGES / "gradio-demo.png", full_page=True)

        # 2. An uploaded file.
        page.locator("button.secondary", has_text="Clear").click()
        page.wait_for_timeout(1000)
        page.get_by_label("Upload file").click()
        page.locator("input[type=file]").first.set_input_files(EXAMPLES_DIR / f"{UPLOAD_ID}.flac")
        page.wait_for_timeout(2000)
        page.get_by_role("button", name="Submit").click()
        text = wait_for_transcript(page, previous="")
        results["upload"] = report("upload", UPLOAD_ID, text)

        # 3. The microphone, simulated by Chrome.
        page.locator("button.secondary", has_text="Clear").click()
        page.wait_for_timeout(1000)
        page.get_by_label("Record audio").click()
        page.wait_for_timeout(1000)
        page.locator("button.record-button, button.record").first.click()
        # The clip loops from when the stream opens, so record long enough for a full pass.
        page.wait_for_timeout(int((2 * duration + 1) * 1000))
        page.locator("button.stop-button, button.stop").first.click()
        page.wait_for_timeout(2500)
        page.get_by_role("button", name="Submit").click()
        text = wait_for_transcript(page, previous="")
        results["microphone"] = report("microphone", MIC_ID, text[:len(REFERENCES[MIC_ID])])
        page.screenshot(path=IMAGES / "gradio-microphone.png", full_page=True)

        browser.close()
    demo.close()

    print("Page errors:", errors or "none")
    if errors or any(cer >= 50 for cer in results.values()):
        raise SystemExit("UI CHECK FAILED")
    print("UI CHECK PASSED")


if __name__ == "__main__":
    main()

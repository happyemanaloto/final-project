# voice_sanity.py — resilient mic → WAV → Whisper sanity check
# pip install sounddevice soundfile numpy openai-whisper pyttsx3
# (Windows/macOS/Linux) Make sure ffmpeg is on PATH: `ffmpeg -version`
# already there at bottom:
if __name__ == "__main__":
    main()

import argparse
import os
import sys
import shutil
import time
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

# Optional TTS
try:
    import pyttsx3
except Exception:
    pyttsx3 = None


# ------------------ Diagnostics ------------------

def _check_ffmpeg():
    if shutil.which("ffmpeg"):
        return True
    print("[error] ffmpeg not found on PATH. Install it and reopen terminal.")
    print("        Windows: https://github.com/BtbN/FFmpeg-Builds/releases  (add bin to PATH)")
    print("        macOS (brew):  brew install ffmpeg")
    print("        Linux:         sudo apt-get install ffmpeg")
    return False


def _check_whisper(model_name: str):
    try:
        import whisper  # noqa: F401
        return True
    except Exception:
        print("[error] `openai-whisper` not installed in this environment.")
        print("        pip install -U openai-whisper")
        print(f"        Then re-run: python voice_sanity.py --model {model_name}")
        return False


# ------------------ Devices ------------------

def list_input_devices():
    print("=== Input devices ===")
    for i, d in enumerate(sd.query_devices()):
        print(f"{i:>2}  {d['name']}   IN {d['max_input_channels']}  OUT {d['max_output_channels']}")


def pick_default_input_device():
    """Best-effort auto-pick of a real microphone."""
    candidates = []
    for i, d in enumerate(sd.query_devices()):
        ch = int(d.get("max_input_channels", 0))
        if ch <= 0:
            continue
        name = (d["name"] or "").lower()
        score = 10
        if any(k in name for k in ["mic", "microphone", "array", "usb"]):
            score += 10
        if "stereo mix" in name or "mapper" in name or "primary" in name or "speaker" in name:
            score -= 20
        candidates.append((score, i, d["name"]))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def set_audio_defaults(mic_index: int | None, samplerate: int, channels: int = 1):
    """Try requested samplerate; if it fails, fall back to device default."""
    if mic_index is None:
        mic_index = pick_default_input_device()
        if mic_index is None:
            raise RuntimeError("No suitable input device found. Use --list-devices and pass --mic <index>.")
        print(f"[auto] Using input device index {mic_index}")
    sd.default.device = (mic_index, None)  # (input, output)
    sd.default.channels = channels

    try:
        sd.default.samplerate = samplerate
        # probe by starting/aborting a short stream
        with sd.InputStream(samplerate=samplerate, channels=channels, device=mic_index):
            pass
    except Exception:
        # fall back to device's default samplerate
        dev = sd.query_devices(mic_index)
        dev_sr = int(dev.get("default_samplerate") or 16000)
        print(f"[warn] {samplerate} Hz unsupported on this mic. Falling back to {dev_sr} Hz.")
        sd.default.samplerate = dev_sr

    return mic_index, int(sd.default.samplerate)


# ------------------ Record / Playback / TTS ------------------

def beep():
    try:
        if sys.platform.startswith("win"):
            import winsound
            winsound.Beep(1000, 180)
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass


def record_to_wav(seconds: int = 6, samplerate: int = 16000, channels: int = 1, mic_index: int | None = None) -> Path:
    """Record from mic to a temp WAV and return its path."""
    mic_index, samplerate = set_audio_defaults(mic_index, samplerate, channels)
    print(f"Recording {seconds}s @ {samplerate} Hz from device {mic_index}… speak now!")
    beep()
    n = int(seconds * samplerate)
    audio = sd.rec(n, samplerate=samplerate, channels=channels, dtype="int16")
    sd.wait()

    # Loudness estimate (RMS)
    rms = float(np.sqrt((audio.astype(np.float32) ** 2).mean()))
    print("RMS level:", round(rms, 1), "(~0=very quiet, 300–2000=normal speech, >4000=very loud)")
    if rms < 50:
        print("[warn] Very low level detected (silence?). Check mic selection/permissions/gain.")

    tmp = Path(tempfile.gettempdir()) / f"voice_sanity_{int(time.time())}.wav"
    sf.write(str(tmp), audio, samplerate)
    print("Saved to:", tmp)
    return tmp


def playback(wav_path: Path):
    try:
        data, sr = sf.read(str(wav_path), dtype="float32")
        print("Playing back…")
        sd.play(data, sr)
        sd.wait()
    except Exception as e:
        print("[warn] playback failed:", e)


def tts_say(text: str, lang_hint: str | None = None, rate: int | None = None):
    if not pyttsx3:
        print("[warn] pyttsx3 not installed; skipping TTS.")
        return
    try:
        eng = pyttsx3.init()
        if rate:
            eng.setProperty("rate", rate)
        # try to pick a voice by lang tag fragment
        if lang_hint:
            want = lang_hint.lower()
            for v in eng.getProperty("voices"):
                # some engines expose .languages as bytes
                langs = []
                try:
                    langs = [L.decode("utf-8", "ignore") for L in (v.languages or [])]
                except Exception:
                    langs = [str(x) for x in (v.languages or [])]
                blob = " ".join([v.id or "", v.name or "", " ".join(langs)]).lower()
                if want in blob:
                    eng.setProperty("voice", v.id)
                    break
        eng.say(text)
        eng.runAndWait()
    except Exception as e:
        print("[warn] TTS playback failed:", e)


# ------------------ Whisper ------------------

def transcribe_whisper(wav_path: Path, model_name: str = "small", language: str | None = None) -> str:
    """Transcribe a WAV file with OpenAI Whisper (local)."""
    if not _check_ffmpeg():
        return ""
    if not _check_whisper(model_name):
        return ""

    import whisper
    print(f"Loading Whisper model: {model_name} … (first time may download)")
    model = whisper.load_model(model_name)
    print("Transcribing…")
    try:
        res = model.transcribe(str(wav_path), fp16=False, language=language)
    except Exception as e:
        print("[error] Whisper failed:", e)
        return ""
    text = (res.get("text") or "").strip()
    return text


# ------------------ CLI ------------------

def main():
    ap = argparse.ArgumentParser(description="Mic → Whisper transcription sanity check")
    ap.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    ap.add_argument("--mic", type=int, default=None, help="Input device index (see --list-devices)")
    ap.add_argument("--seconds", type=int, default=6, help="Recording length")
    ap.add_argument("--sr", type=int, default=16000, help="Requested sample rate")
    ap.add_argument("--model", type=str, default=os.getenv("CHEF_WHISPER_MODEL", "small"),
                    help="Whisper model: tiny/tiny.en/base/base.en/small/small.en/medium/large/large-v3 …")
    ap.add_argument("--lang", type=str, default=None, help="Language hint (e.g., en, tl). Optional.")
    ap.add_argument("--playback", action="store_true", help="Play back the recorded audio")
    ap.add_argument("--speak", action="store_true", help="Use TTS to read back the transcript")
    ap.add_argument("--keep", action="store_true", help="Keep the temp WAV file (don’t delete)")
    args = ap.parse_args()

    if args.list_devices:
        list_input_devices()
        return

    # Windows privacy gotcha reminder
    if sys.platform.startswith("win"):
        print("Tip: Windows → Settings → Privacy & security → Microphone (allow desktop apps)")

    wav_path = None
    try:
        wav_path = record_to_wav(seconds=args.seconds, samplerate=args.sr, mic_index=args.mic)
        if args.playback:
            playback(wav_path)
        text = transcribe_whisper(wav_path, model_name=args.model, language=args.lang)
        print("\n=== Transcript ===")
        print(text if text else "(no speech recognized)")
        if args.speak and text:
            tts_say(text, lang_hint=args.lang or "en")
    except KeyboardInterrupt:
        print("\nCanceled.")
    except Exception as e:
        print("[fatal] ", e)
        print("Run with --list-devices and try a different --mic index.")
    finally:
        if wav_path and not args.keep:
            try:
                Path(wav_path).unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()

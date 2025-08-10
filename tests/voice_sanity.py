# # voice_sanity.py
# import os
# from importlib.machinery import SourceFileLoader

# # 1) Point this to your actual file path
# BOT_PATH = r"C:\Users\happy\Documents\ironhack\kusina-bot\final-project\tests\kusina-bot.py"

# # 2) Pick a Whisper size; "small" is a good balance. Try "medium" if accuracy is poor.
# os.environ["CHEF_WHISPER_MODEL"] = "small"

# # Load your bot file as a module (won’t run main() because __name__ won’t be "__main__")
# kusina_bot = SourceFileLoader("kusina_bot", BOT_PATH).load_module()

# # Optional: pick input device if needed
# # import sounddevice as sd
# # print(sd.query_devices())
# # sd.default.device = (2, None)  # (INPUT_ID, None)

# print("Say something…")
# text = kusina_bot.stt_from_mic(max_seconds=6)  # remove lang_hint
# print("Heard:", text)
# voice_sanity.py
# Quick mic → WAV → Whisper transcription (+ optional TTS)
#
# Requires:
#   pip install sounddevice soundfile numpy openai-whisper pyttsx3
#   (and ffmpeg installed on your system PATH)
#
# Examples:
#   python voice_sanity.py --list-devices
#   python voice_sanity.py --mic 6 --seconds 6 --model small --lang en --playback --speak
#
# On Windows, if you hear nothing or get silence,
# check Windows Settings → Privacy & security → Microphone (allow desktop apps).
# python voice_sanity.py --mic 6 --seconds 6 --model small --lang en --playback --speak


import argparse
import os
import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

# --- Optional TTS (pyttsx3) ---
try:
    import pyttsx3
except Exception:
    pyttsx3 = None


def list_input_devices():
    print("=== Input devices ===")
    for i, d in enumerate(sd.query_devices()):
        print(f"{i:>2}  {d['name']}   IN {d['max_input_channels']}")


def pick_default_input_device():
    """Best-effort auto-pick of a real microphone."""
    candidates = []
    for i, d in enumerate(sd.query_devices()):
        name = (d["name"] or "").lower()
        ch = int(d.get("max_input_channels", 0))
        if ch <= 0:
            continue
        if any(k in name for k in ["stereo mix", "mapper", "primary", "speaker"]):
            continue
        # prefer explicit mics
        score = 10
        if "mic" in name or "microphone" in name or "array" in name:
            score += 10
        candidates.append((score, i, d["name"]))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def set_audio_defaults(mic_index: int | None, samplerate: int, channels: int = 1):
    if mic_index is None:
        mic_index = pick_default_input_device()
        if mic_index is None:
            raise RuntimeError("No suitable input device found. Use --list-devices and pass --mic <index>.")
        print(f"[auto] Using input device index {mic_index}")
    sd.default.device = (mic_index, None)  # (input, output)
    sd.default.samplerate = samplerate
    sd.default.channels = channels
    return mic_index


def beep():
    try:
        if sys.platform.startswith("win"):
            import winsound
            winsound.Beep(1000, 180)
        else:
            # terminal bell; may or may not work
            print("\a", end="", flush=True)
    except Exception:
        pass


def record_to_wav(seconds: int = 6, samplerate: int = 16000, channels: int = 1, mic_index: int | None = None) -> Path:
    """Record from mic to a temp WAV and return its path."""
    set_audio_defaults(mic_index, samplerate, channels)
    n = int(seconds * samplerate)
    print(f"Recording {seconds}s @ {samplerate} Hz… speak now!")
    beep()
    audio = sd.rec(n, samplerate=samplerate, channels=channels, dtype="int16")
    sd.wait()
    # Loudness estimate
    rms = float(np.sqrt((audio.astype(np.float32) ** 2).mean()))
    print("RMS level:", round(rms, 1), "(~0=very quiet, 300–2000=normal speech, >4000=very loud)")
    # Save
    tmp = Path(tempfile.gettempdir()) / f"voice_sanity_{int(time.time())}.wav"
    sf.write(str(tmp), audio, samplerate)
    print("Saved to:", tmp)
    return tmp


def transcribe_whisper(wav_path: Path, model_name: str = "small", language: str | None = None) -> str:
    """Transcribe a WAV file with OpenAI Whisper (local)."""
    # Lazy import so listing devices still works without whisper installed
    import whisper
    print(f"Loading Whisper model: {model_name} … (first time may take a bit)")
    model = whisper.load_model(model_name)
    # Force CPU-friendly unless you have a GPU with CUDA
    print("Transcribing…")
    res = model.transcribe(str(wav_path), fp16=False, language=language)
    text = (res.get("text") or "").strip()
    return text


def tts_say(text: str, lang_hint: str | None = None, rate: int | None = None):
    if not pyttsx3:
        print("[warn] pyttsx3 not installed; skipping TTS.")
        return
    eng = pyttsx3.init()
    if rate:
        eng.setProperty("rate", rate)
    if lang_hint:
        try:
            want = lang_hint.lower()
            for v in eng.getProperty("voices"):
                langs = getattr(v, "languages", []) or []
                tags = []
                for L in langs:
                    try:
                        tags.append(L.decode("utf-8", "ignore"))
                    except Exception:
                        tags.append(str(L))
                blob = " ".join([v.id, v.name or "", " ".join(tags)]).lower()
                if want in blob:
                    eng.setProperty("voice", v.id)
                    break
        except Exception:
            pass
    try:
        eng.say(text)
        eng.runAndWait()
    except Exception as e:
        print("[warn] TTS playback failed:", e)


def playback(wav_path: Path, samplerate: int = 16000):
    try:
        data, sr = sf.read(str(wav_path), dtype="float32")
        print("Playing back…")
        sd.play(data, sr)
        sd.wait()
    except Exception as e:
        print("[warn] playback failed:", e)


def main():
    ap = argparse.ArgumentParser(description="Mic → Whisper transcription sanity check")
    ap.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    ap.add_argument("--mic", type=int, default=None, help="Input device index (see --list-devices)")
    ap.add_argument("--seconds", type=int, default=6, help="Recording length")
    ap.add_argument("--sr", type=int, default=16000, help="Sample rate")
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

    try:
        wav_path = record_to_wav(seconds=args.seconds, samplerate=args.sr, mic_index=args.mic)
        if args.playback:
            playback(wav_path, samplerate=args.sr)
        text = transcribe_whisper(wav_path, model_name=args.model, language=args.lang)
        print("\n=== Transcript ===")
        print(text if text else "(no speech recognized)")
        if args.speak and text:
            # TTS language hint: use args.lang if provided
            tts_say(text, lang_hint=args.lang or "en")
    except KeyboardInterrupt:
        print("\nCanceled.")
        return
    finally:
        if not args.keep:
            try:
                if 'wav_path' in locals() and Path(wav_path).exists():
                    Path(wav_path).unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()

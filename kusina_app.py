import argparse, os, sys, traceback
from bot.voice_sanity import record_to_wav, transcribe_whisper, tts_say

print("[kusina_app] starting…")  # <-- loud
print("[kusina_app] sys.executable:", sys.executable)
print("[kusina_app] cwd:", os.getcwd())

try:
    from bot.io import run_cli
    print("[kusina_app] imported run_cli OK")
except Exception as e:
    print("[kusina_app] FAILED to import bot.io.run_cli")
    traceback.print_exc()
    sys.exit(1)

def main():
    print("[kusina_app] entering main()")
    if not os.getenv("OPENAI_API_KEY"):
        print("[kusina_app] OPENAI_API_KEY missing! set it in env or .env")
        raise RuntimeError("OPENAI_API_KEY missing. Put it in your .env")

    ap = argparse.ArgumentParser(description="Kusina Bot — Modular")
    ap.add_argument("--rebuild-vs", action="store_true")
    ap.add_argument("--force-reply-lang", type=str, default=None)
    ap.add_argument("--voice", action="store_true")
    ap.add_argument("--mic-index", type=int, default=None)
    ap.add_argument("--samplerate", type=int, default=16000)
    ap.add_argument("--mic-demo", action="store_true", help="Record mic and transcribe (sanity check).")
    ap.add_argument("--lang", type=str, default=None, help="Language hint for transcription.")

    args = ap.parse_args()
    print("[kusina_app] args:", args)

        # Quick mic sanity (optional path)
    if args.mic_demo:
        print("Listing/recording via voice_sanity helpers…")
        wav = record_to_wav(seconds=6, samplerate=args.samplerate, mic_index=args.mic_index)
        text = transcribe_whisper(wav, model_name="small.en", language=args.lang)
        print("\n=== Transcript ===\n", text)
        if text:
            try:
                tts_say(text, lang_hint=(args.lang or "en"))
            except Exception:
                pass
        return

    try:
        run_cli(
            rebuild_vs=args.rebuild_vs,
            force_reply_lang=args.force_reply_lang,
            voice=args.voice,
            mic_index=args.mic_index,
            samplerate=args.samplerate,
        )
    except Exception as e:
        print("[kusina_app] ERROR inside run_cli")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    print("[kusina_app] __main__ block executing")
    main()

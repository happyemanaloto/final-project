# bot/io.py
import argparse, os, tempfile
from pathlib import Path
import sounddevice as sd, soundfile as sf

from .data import load_all_docs, build_or_load_vectorstore
from .nlp import detect_language, parse_language_switch, ensure_reply_language, llm_zero
from .agentX import build_agent, chat_once
from .tools import bind_vectorstore, bind_session_hooks  # KeywordIndex binding optional
from .session import SessionMemory

def run_cli(rebuild_vs=False, force_reply_lang=None, voice=False, mic_index=None, samplerate=16000):
    # --- start a NEW session each run (ephemeral memory) ---
    user_id = os.getenv("CHEF_USER_ID", "HappyUserTest")
    session = SessionMemory(user_id=user_id)
    if force_reply_lang:
        session.reply_lang = force_reply_lang

    # --- load data + vector store ---
    docs = load_all_docs()
    print(f"Loaded {len(docs)} recipes.")
    vs = build_or_load_vectorstore(docs, rebuild=rebuild_vs)
    bind_vectorstore(vs)

    # wire tools to this session’s last_hits memory
    bind_session_hooks(get_hits=session.get_hits, set_hits=session.remember_hits)

    # OPTIONAL: keyword fallback index (only if your tools.py uses it)
    # from .tools import KeywordIndex, bind_keyword_index
    # bind_keyword_index(KeywordIndex(docs))

    # --- build agent WITH session-aware memory (ConversationBufferMemory lives inside agent) ---
    agent = build_agent(llm_zero(), session)

    # --- voice setup (optional) ---
    if voice:
        try:
            if mic_index is not None:
                sd.default.device = (mic_index, None)
            sd.default.samplerate = samplerate or 16000
            sd.default.channels = 1
            print("[voice] ready")
        except Exception as e:
            print(f"[voice] not available: {e}")
            voice = False

    print("\nKusina Bot ready. Ctrl+C to exit.\n")
    last_bot = ""

    # --- main loop ---
    while True:
        try:
            if voice:
                typed = input("You (Enter=voice, or type): ").strip()
                if typed:
                    user = typed
                else:
                    print("Listening… (8s)")
                    n = int(8 * (samplerate or 16000))
                    audio = sd.rec(n, samplerate=samplerate or 16000, channels=1, dtype="int16", device=mic_index)
                    sd.wait()
                    wav = Path(tempfile.gettempdir()) / "kusina_tmp.wav"
                    sf.write(str(wav), audio, samplerate or 16000)
                    from .tools import _whisper_model
                    res = _whisper_model().transcribe(str(wav), fp16=False)
                    user = (res.get("text") or "").strip()
                    print("You (transcribed):", user)
            else:
                user = input("You: ").strip()

            if not user:
                continue

            # language switch command (updates session.reply_lang)
            maybe = parse_language_switch(user)
            if maybe:
                session.reply_lang = maybe
                pretty = "Tagalog" if maybe == "tl" else maybe
                msg = ensure_reply_language(f"Okay! I’ll reply in {pretty} from now on.", session.reply_lang)
                print("\nAssistant:\n" + msg + "\n")
                last_bot = msg
                continue

            # per-turn ephemeral language: if user typed a long message in a different language,
            # answer in that language for this turn only (don’t change session default)
            det = detect_language(user)
            turn_lang = session.reply_lang if (det == session.reply_lang or len(user) < 60) else det

            ans = chat_once(agent, user, reply_lang=turn_lang)
            print("\nAssistant:\n" + ans + "\n")
            last_bot = ans

        except KeyboardInterrupt:
            print("\nBye!")
            break

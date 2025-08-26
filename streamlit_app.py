# streamlit_app.py — Kusina Bot Full Interface
#
# This Streamlit app implements a conversational cooking assistant with
# multiple capabilities:
#   • Loads the recipe vector store and builds the agent on first run.
#   • Supports a language dropdown and translation toggle.
#   • Offers a microphone recorder (via audio_recorder_streamlit) with
#     user‑controlled playback and manual transcription.
#   • Provides an upload button for audio/video files with preview and
#     transcription + summarization.
#   • Includes a speaker toggle that auto‑plays the last assistant reply
#     when enabled.  Speech synthesis respects the current reply language.
#   • Positions the LangSmith tracing toggle at the top left of the page.

from __future__ import annotations

import os
import io
import re
import json
import time
import base64
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from dotenv import load_dotenv  # 👈 add this
# Load environment variables from .env file
load_dotenv()

# Optional extras
try:
    from audio_recorder_streamlit import audio_recorder  # type: ignore
    HAS_REC = True
except Exception:
    HAS_REC = False
try:
    from gtts import gTTS  # type: ignore
    HAS_GTTS = True
except Exception:
    HAS_GTTS = False

# Import back‑end modules
from bot.data import load_all_docs, build_or_load_vectorstore
from bot.tools import (
    bind_vectorstore,
    bind_session_hooks,
    transcribe_local_media,
    summarize_transcript_file,
)
from bot.session import SessionMemory
from bot.agent import build_agent, chat_once
from bot.nlp import (
    detect_language,
    parse_language_switch,
    ensure_reply_language,
    llm_zero,
    _pick_gtts_lang,
)

# ---------- Visual assets ----------
PROJECT_ROOT = Path(__file__).resolve().parent
WELCOME_VIDEO = PROJECT_ROOT / "bot" / "favorites" / "welcomepage.mp4"
BACKGROUND_IMG = PROJECT_ROOT / "bot" / "favorites" / "background.jpg"

# ---------- Helpers ----------

def _b64_file(path: Path) -> Optional[str]:
    """Read a file and return a base64 string; return None on failure."""
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        return None


def _apply_background():
    """Apply a page background image and custom CSS for chat bubbles and controls."""
    b64 = _b64_file(BACKGROUND_IMG) if BACKGROUND_IMG.exists() else None
    mime = "image/png" if BACKGROUND_IMG.suffix.lower() == ".png" else "image/jpeg"
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            {"background: url('data:%s;base64,%s') center center / cover no-repeat fixed !important;" % (mime, b64) if b64 else ""}
        }}
        [data-testid="stHeader"], [data-testid="stToolbar"] {{ background: transparent !important; }}
        .block-container {{
            background: transparent !important;
            padding-top: 5.8rem !important;
            padding-bottom: 1rem !important;
        }}
        /* Chat bubbles */
        .chat-bubble {{
            max-width: 78%;
            padding: 14px 16px;
            border: 2px solid #000;
            border-radius: 18px;
            background: #f8fff0;
            color: #111;
            margin: 8px 0 8px 0;
        }}
        .chat-bubble.user {{
            margin-left: auto;
            font-style: italic;
            font-weight: 700;
        }}
        .chat-bubble.bot {{
            margin-right: auto;
            font-family: cursive;
        }}
        /* Right rail sticky */
        .right-rail {{ position: sticky; top: 6rem; }}
        /* Icon button styling */
        .icon-btn button {{
            height: 44px; font-size: 22px;
            background: #f8fff0 !important;
            color: #000 !important;
            border: 1px solid #000 !important;
        }}
        /* Dropdown height */
        .stSelectbox div[data-baseweb="select"] > div {{
            min-height: 28px !important;
            height: 28px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _video_html(path: Path):
    """Render a looping video if the file exists."""
    if not path.exists():
        return
    b64 = _b64_file(path)
    if not b64:
        return
    st.markdown(
        f'''
        <div style="height:86vh; display:flex; align-items:center; justify-content:center;">
          <video autoplay muted loop playsinline style="width:100%; height:100%; object-fit:cover; border-radius:14px;">
            <source src="data:video/mp4;base64,{b64}">
          </video>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _naturalize_for_tts(text: str) -> str:
    """Insert light punctuation and breaks so gTTS sounds more natural."""
    if not text:
        return ""
    t = text.strip()
    # Convert bullets/dashes into sentences
    t = re.sub(r"[•\-–]\s*", ". ", t)
    # Replace line breaks with stops
    t = re.sub(r"(?<![.!?])\n+", ". ", t)
    # Soften colons into commas
    t = t.replace(":", ",")
    # Compress whitespace
    t = re.sub(r"\s+", " ", t).strip()
    # Ensure sentence ends with punctuation
    if t and t[-1] not in ".!?":
        t += "."
    return t


def _gtts_to_b64(text: str) -> Optional[str]:
    """Generate a base64‑encoded MP3 from text using gTTS.  Applies naturalization and language selection."""
    if not HAS_GTTS or not text:
        return None
    try:
        # Choose gTTS language based on current reply_lang
        lang_code = _pick_gtts_lang(st.session_state.reply_lang)
        buf = io.BytesIO()
        gTTS(text=_naturalize_for_tts(text), lang=lang_code).write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")
    except Exception:
        return None


def _autoplay_audio(b64_mp3: str):
    """Render an HTML5 audio tag that plays automatically once."""
    if not b64_mp3:
        return
    st.markdown(
        f"""
        <audio autoplay>
          <source src="data:audio/mp3;base64,{b64_mp3}" type="audio/mpeg">
        </audio>
        """,
        unsafe_allow_html=True,
    )


def _play_audio_controls(b64_audio: str, mime: str):
    """Render an audio player with controls for user playback."""
    if not b64_audio:
        return
    st.markdown(
        f"""
        <audio controls>
          <source src="data:{mime};base64,{b64_audio}">
        </audio>
        """,
        unsafe_allow_html=True,
    )


def _save_temp_file(b64_data: str, suffix: str) -> str:
    """Write base64 data to a temporary file and return its path."""
    raw = base64.b64decode(b64_data)
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return path


def _is_audio(ext: str) -> bool:
    return ext.lower() in {"wav", "mp3", "m4a", "aac", "ogg", "flac", "opus"}


def _is_video(ext: str) -> bool:
    return ext.lower() in {"mp4", "mov", "mkv", "webm", "m4v", "mpeg", "mpg", "avi"}


# ---------- State ----------
def _init_state():
    """Initialize or restore Streamlit session state for persistent variables."""
    ss = st.session_state
    ss.setdefault("messages", [])           # chat history: list of dicts {role, content}
    ss.setdefault("reply_lang", "en")       # ISO code for replies
    ss.setdefault("translate_on", False)    # friend toggle (translate responses to reply_lang)
    ss.setdefault("langsmith_on", False)    # tracing toggle (LangSmith)
    ss.setdefault("speaker_on", False)      # speaker toggle
    ss.setdefault("last_bot_text", "")       # last assistant message
    ss.setdefault("last_tts_b64", None)     # base64 audio prepared for auto playback
    ss.setdefault("rec_b64", None)          # base64 of recorded audio via mic
    ss.setdefault("rec_mime", "audio/wav")  # MIME for mic recording
    ss.setdefault("show_rec_player", False) # show mic recording playback controls
    ss.setdefault("upload_tmp_path", None)  # path of uploaded media for preview/transcribe
    ss.setdefault("show_upload", False)     # toggle to show upload controls
    ss.setdefault("session", None)          # SessionMemory instance
    ss.setdefault("agent", None)            # agent instance
    ss.setdefault("boot_done", False)       # flag to indicate backend loaded


def _boot_once():
    """Load documents, build vectorstore, wire session hooks, and build agent once."""
    ss = st.session_state
    if ss.boot_done:
        return
    # Create a session; user id can be static since Streamlit sessions are per user
    session = SessionMemory(user_id=os.getenv("CHEF_USER_ID", "HappyUser"))
    session.reply_lang = ss.reply_lang
    # Load docs and build the vector store (Chroma)
    docs = load_all_docs()
    vs = build_or_load_vectorstore(docs, rebuild=False)
    bind_vectorstore(vs)
    # Bind session hooks to store hits for later calorie/shopping lookups
    bind_session_hooks(session.get_hits, session.remember_hits)
    # Build the agent with an LLM; use llm_zero for minimal temperature
    agent = build_agent(llm_zero(), session)
    ss.session = session
    ss.agent = agent
    ss.boot_done = True


# ---------- Main Page ----------

st.set_page_config(page_title="Kusina Bot", page_icon="🍳", layout="wide")
_apply_background()
_init_state()
_boot_once()

ss = st.session_state  # alias

# ---------- Top row: LangSmith toggle (top-left) ----------
with st.container():
    # Place the LangSmith toggle at the top-left above the left column
    cols = st.columns([0.26, 0.74])
    with cols[0]:
        ls = st.toggle("LangSmith tracing", value=ss.langsmith_on)
        if ls != ss.langsmith_on:
            ss.langsmith_on = ls
            # Set environment variable for LangChain tracing (persist across runs)
            os.environ["LANGCHAIN_TRACING_V2"] = "true" if ls else "false"
            os.environ["LANGCHAIN_PROJECT"] = st.text_input("LangSmith project", value="kusina-bot")
            # Rebuild agent to attach new tracer settings
            ss.boot_done = False
            _boot_once()
            st.rerun()
    with cols[1]:
        st.empty()


# ---------- Layout: left video, middle chat, right controls ----------
left, mid, right = st.columns([0.26, 0.60, 0.14], gap="large")

with left:
    # Show welcome video
    _video_html(WELCOME_VIDEO)

with mid:
    # Autoplay pending TTS if speaker is on and audio is prepared
    if ss.speaker_on and ss.last_tts_b64:
        _autoplay_audio(ss.last_tts_b64)
        # Clear after playing so it doesn't loop on subsequent reruns
        ss.last_tts_b64 = None

    # Render chat history
    for msg in ss.messages[-200:]:
        cls = "bot" if msg["role"] == "assistant" else "user"
        st.markdown(f'<div class="chat-bubble {cls}">{msg["content"]}</div>', unsafe_allow_html=True)

    # Chat input: user types here
    user_text = st.chat_input("Type a message…")
    if user_text:
        txt = user_text.strip()
        if txt:
            # Append user message
            ss.messages.append({"role": "user", "content": txt})
            # Check for language switch commands
            maybe = parse_language_switch(txt)
            if maybe:
                # Normalize to ISO codes via mapping or alias (maybe is alias or iso)
                iso_map = {
                    "english": "en", "tagalog": "tl", "filipino": "tl", "tl": "tl", "fil": "tl",
                    "korean": "ko", "ko": "ko",
                    "spanish": "es", "español": "es", "espanol": "es", "castellano": "es", "es": "es",
                    "dutch": "nl", "nederlands": "nl", "nl": "nl",
                    "french": "fr", "francais": "fr", "français": "fr", "fr": "fr",
                    "german": "de", "deutsch": "de", "de": "de",
                    "italian": "it", "italiano": "it", "it": "it",
                    "portuguese": "pt", "portugues": "pt", "português": "pt", "pt": "pt",
                    "japanese": "ja", "nihongo": "ja", "ja": "ja",
                    "chinese": "zh", "mandarin": "zh", "zh": "zh", "zh-cn": "zh", "zh-tw": "zh",
                }
                new_lang = iso_map.get(maybe.lower(), ss.reply_lang)
                ss.reply_lang = new_lang
                ss.session.reply_lang = new_lang
                pretty_map = {
                    "en": "English", "tl": "Tagalog", "ko": "Korean", "es": "Spanish", "nl": "Dutch",
                    "fr": "French", "de": "German", "it": "Italian", "pt": "Portuguese", "ja": "Japanese", "zh": "Chinese",
                }
                pretty = pretty_map.get(new_lang, new_lang)
                conf_msg = ensure_reply_language(f"Okay! I’ll reply in {pretty} from now on.", new_lang)
                ss.messages.append({"role": "assistant", "content": conf_msg})
                ss.last_bot_text = conf_msg
                # Pre‑generate TTS if speaker is on
                if ss.speaker_on:
                    b64 = _gtts_to_b64(conf_msg)
                    if b64:
                        ss.last_tts_b64 = b64
                st.rerun()

            # Determine language for this turn
            det = detect_language(txt)
            turn_lang = ss.reply_lang if ss.translate_on else (
                ss.session.reply_lang if (det == ss.session.reply_lang or len(txt) < 60) else det
            )
            # Call agent for answer
            ans = chat_once(ss.agent, txt, reply_lang=turn_lang)
            if ss.translate_on and turn_lang != ss.reply_lang:
                ans = ensure_reply_language(ans, ss.reply_lang)
            # Append assistant reply
            ss.messages.append({"role": "assistant", "content": ans})
            ss.last_bot_text = ans
            # Pre‑generate TTS if speaker is on
            if ss.speaker_on:
                b64 = _gtts_to_b64(ans)
                if b64:
                    ss.last_tts_b64 = b64
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


with right:
    st.markdown('<div class="right-rail icon-btn">', unsafe_allow_html=True)
    # Language dropdown
    lang_labels = ["English", "Tagalog", "Spanish", "French", "German", "Italian", "Dutch", "Portuguese", "Japanese", "Chinese"]
    lang_codes = ["en", "tl", "es", "fr", "de", "it", "nl", "pt", "ja", "zh"]
    try:
        idx = lang_codes.index(ss.reply_lang)
    except ValueError:
        idx = 0
    selected_label = st.selectbox(
        "Language", options=lang_labels, index=idx, key="reply_lang_select"
    )
    new_code = lang_codes[lang_labels.index(selected_label)]
    if new_code != ss.reply_lang:
        ss.reply_lang = new_code
        if ss.session:
            ss.session.reply_lang = new_code

    # Friend toggle (translate all replies)
    if st.button("友", use_container_width=True, help="Toggle translation of replies into chosen language"):
        ss.translate_on = not ss.translate_on

    # Upload toggle (show/hide uploader)
    if st.button("➕", use_container_width=True, help="Upload audio/video to transcribe & summarize"):
        ss.show_upload = not ss.show_upload
    # If uploader visible, render file uploader and transcribe button
    if ss.show_upload:
        up = st.file_uploader(
            "Upload audio/video", type=[
                "wav", "mp3", "m4a", "aac", "ogg", "flac", "opus",
                "mp4", "mov", "mkv", "webm", "m4v", "mpeg", "mpg", "avi"
            ],
            label_visibility="collapsed"
        )
        if up is not None:
            # Save upload to tmp path and preview
            tmp_dir = PROJECT_ROOT / "tmp_uploads"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / f"{int(time.time())}_{up.name}"
            tmp_path.write_bytes(up.read())
            ss.upload_tmp_path = str(tmp_path)
            ext = (up.name or "").split(".")[-1].lower()
            if _is_audio(ext):
                st.audio(str(tmp_path))
            elif _is_video(ext):
                st.video(str(tmp_path))
            else:
                st.info("Preview not supported, but I can still transcribe it.")
        if ss.upload_tmp_path and st.button("📝 Transcribe Upload", use_container_width=True):
            try:
                res = transcribe_local_media.invoke({"path": ss.upload_tmp_path})
                if isinstance(res, str):
                    res_json = json.loads(res)
                else:
                    res_json = res
                if res_json.get("error"):
                    ss.messages.append({"role": "assistant", "content": ensure_reply_language(f"Transcription failed: {res_json['error']}", ss.reply_lang)})
                    st.rerun()
                transcript_path = Path(res_json.get("transcript_path", ""))
                if not transcript_path.is_absolute():
                    transcript_path = (PROJECT_ROOT / transcript_path).resolve()
                summ = summarize_transcript_file.invoke({"transcript_path": str(transcript_path), "target_lang": ss.reply_lang})
                summary_text = summ if isinstance(summ, str) else str(summ)
                # Show summarization in chat
                ss.messages.append({"role": "user", "content": "(Uploaded media) Please summarize ingredients and steps."})
                ss.messages.append({"role": "assistant", "content": summary_text})
                ss.last_bot_text = summary_text
                if ss.speaker_on:
                    b64tts = _gtts_to_b64(summary_text)
                    if b64tts:
                        ss.last_tts_b64 = b64tts
                st.rerun()
            except Exception as e:
                ss.messages.append({"role": "assistant", "content": ensure_reply_language(f"Upload error: {e}", ss.reply_lang)})
                st.rerun()

    # Mic controls
    st.markdown("### 🎙️ Mic")
    st.caption("Record audio, play/stop it, then transcribe when ready.")
    if HAS_REC:
        mic_bytes = audio_recorder(text="", icon_size="2x")
        if mic_bytes:
            ss.rec_b64 = base64.b64encode(mic_bytes).decode("ascii")
            ss.rec_mime = "audio/wav"
            ss.show_rec_player = True
        # Playback controls
        if ss.show_rec_player and ss.rec_b64:
            pcols = st.columns(2)
            if pcols[0].button("▶️ Play Recording", use_container_width=True):
                _play_audio_controls(ss.rec_b64, ss.rec_mime)
            if pcols[1].button("⏹ Stop Playback", use_container_width=True):
                st.rerun()
            # Transcribe button
            if st.button("📝 Transcribe", use_container_width=True):
                try:
                    fpath = _save_temp_file(ss.rec_b64, suffix=".wav")
                    res = transcribe_local_media.invoke({"path": fpath})
                    if isinstance(res, str):
                        res_json = json.loads(res)
                    else:
                        res_json = res
                    if res_json.get("error"):
                        ss.messages.append({"role": "assistant", "content": ensure_reply_language(f"Transcription failed: {res_json['error']}", ss.reply_lang)})
                        st.rerun()
                    transcript_path = Path(res_json.get("transcript_path", ""))
                    if not transcript_path.is_absolute():
                        transcript_path = (PROJECT_ROOT / transcript_path).resolve()
                    summ = summarize_transcript_file.invoke({"transcript_path": str(transcript_path), "target_lang": ss.reply_lang})
                    summary_text = summ if isinstance(summ, str) else str(summ)
                    ss.messages.append({"role": "user", "content": "(Mic) Please summarize ingredients and steps."})
                    ss.messages.append({"role": "assistant", "content": summary_text})
                    ss.last_bot_text = summary_text
                    if ss.speaker_on:
                        b64tts = _gtts_to_b64(summary_text)
                        if b64tts:
                            ss.last_tts_b64 = b64tts
                    st.rerun()
                except Exception as e:
                    ss.messages.append({"role": "assistant", "content": ensure_reply_language(f"Mic processing error: {e}", ss.reply_lang)})
                    st.rerun()
    else:
        st.caption("Install 'audio-recorder-streamlit' to enable mic recording.")

    # Speaker toggle (shows ON/OFF; pre‑generates TTS when turning on)
    speaker_label = "🔊 Speaker: ON" if ss.speaker_on else "🔈 Speaker: OFF"
    if st.button(speaker_label, use_container_width=True):
        ss.speaker_on = not ss.speaker_on
        if ss.speaker_on:
            # Pre-generate audio for the most recent bot reply if available
            if ss.last_bot_text:
                b64 = _gtts_to_b64(ss.last_bot_text)
                if b64:
                    ss.last_tts_b64 = b64
        else:
            ss.last_tts_b64 = None
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
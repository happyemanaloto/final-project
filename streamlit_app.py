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
# --- hydrate env from Streamlit Secrets (cloud) + .env (local) ---
import os
try:
    import streamlit as st  # will be available in Streamlit Cloud
    for k, v in st.secrets.items():
        if isinstance(v, (str, int, float, bool)):
            os.environ.setdefault(str(k), str(v))
except Exception:
    pass

try:
    from dotenv import load_dotenv  # no-op in cloud if .env not present
    load_dotenv()
except Exception:
    pass

from __future__ import annotations

# import os
import io
import re
import json
import time
import base64
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
try:
    from yt_dlp import YoutubeDL
    _HAS_YTDLP = True
except Exception:
    YoutubeDL = None
    _HAS_YTDLP = False
        
# import streamlit as st

# from dotenv import load_dotenv  # 👈 add this
# # Load environment variables from .env file
# load_dotenv()

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
from bot.data import load_all_docs, build_or_load_vectorstore, DEFAULT_VS_DIR
from bot.tools import (
    bind_vectorstore,
    bind_session_hooks,
    transcribe_local_media,
    summarize_transcript_file, 
    ingest_link,
    summarize_video,
    transcribe_media,
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
YES_WORDS = {"yes","yeah","yep","sure","ok","okay","please","go ahead","do it"}
NO_WORDS  = {"no","nope","nah","not now","later"}

def _set_pending(action: str, payload: dict | None = None):
    st.session_state["pending_action"] = action
    st.session_state["pending_payload"] = payload or {}

def _consume_pending():
    st.session_state.pop("pending_action", None)
    st.session_state.pop("pending_payload", None)

def _maybe_mark_offer_from_last_assistant():
    msgs = st.session_state.get("messages", [])
    if not msgs:
        return
    last = msgs[-1]
    if last.get("role") != "assistant":
        return
    txt = (last.get("content") or "").lower()
    m = re.search(r"would you like (?:the )?recipe for ([^?]+)\?", txt)
    if m:
        dish = m.group(1).strip()
        _set_pending("offer_recipe", {"dish": dish})


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
            min-height: 44px !important;
            height: 44px !important;
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

def _yt_id(url: str) -> str | None:
    m = re.search(r"(?:v=|youtu\.be/)([\w\-]{6,})", url)
    return m.group(1) if m else None

@st.cache_data(show_spinner=False, ttl=24*3600)
def _yt_title(url: str) -> str:
    """Try to resolve the YouTube title; fall back to a friendly label."""
    # Try yt-dlp first (fast + no API key)
    if _HAS_YTDLP:
        try:
            with YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
                info = ydl.extract_info(url, download=False)
                t = (info or {}).get("title")
                if t:
                    return t
        except Exception:
            pass
    # Fallback: readable label from video id or URL
    vid = _yt_id(url)
    return f"YouTube video ({vid})" if vid else url

@st.cache_data(show_spinner=False, ttl=24*3600)
def _title_map(urls: list[str]) -> dict[str, str]:
    """Return {title: url} mapping for a list of URLs (titles must be unique; de-dupe if needed)."""
    mapping: dict[str, str] = {}
    seen = set()
    for u in urls:
        t = _yt_title(u)
        # de-dup titles by appending last 5 chars of id if needed
        if t in seen:
            suffix = (_yt_id(u) or u)[-5:]
            t = f"{t} · {suffix}"
        mapping[t] = u
        seen.add(t)
    return mapping


def remove_json_block(text: str) -> str:
    """
    Clean assistant replies or transcripts by removing any JSON/code blocks.
    Handles cases where multiple calorie summaries appear.
    """
    if not text:
        return ""

    t = text

    # Remove fenced ```json ... ``` blocks
    t = re.sub(r"```json\s*[\s\S]*?```", "", t, flags=re.IGNORECASE)

    # Remove any fenced ``` ... ``` code blocks (non-json too)
    t = re.sub(r"```[\s\S]*?```", "", t)

    # Remove inline or multiline {...} objects that look like JSON
    t = re.sub(r"\{[^{}]*\}", "", t)

    # Collapse repeated recipe calorie summaries into one (keep only first occurrence)
    lines = []
    seen_calorie = False
    for line in t.splitlines():
        if re.search(r"\bcalories?\b", line, flags=re.IGNORECASE):
            if seen_calorie:
                continue  # skip duplicates
            seen_calorie = True
        lines.append(line)

    # Join, normalize whitespace
    cleaned = " ".join(" ".join(lines).split())
    return cleaned.strip()

def kb_count() -> int:
    """Best-effort vector DB count; falls back to a session counter."""
    vs = st.session_state.get("vs")
    try:
        return int(getattr(getattr(vs, "_collection", None), "count", lambda: 0)())
    except Exception:
        return int(st.session_state.get("kb_docs", 0))

def _as_str(x) -> str:
    return x if isinstance(x, str) else str(x)

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
#
# Additional cleaning for text-to-speech
#
# The bot often uses emojis to express cooking themes (🍳🥗) and includes
# certain characters or substrings (e.g. "#", "*", apostrophes, or the word
# "json") that can cause gTTS to mispronounce or produce awkward pauses.
# To ensure smoother audio output while preserving the visual reply, we
# remove these items just before calling gTTS.  This function should be
# applied after naturalization but before gTTS invocation.  It leaves
# non‑problematic punctuation intact.
_EMOJI_RE = re.compile(
    "["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags
    u"\U00002700-\U000027BF"  # dingbats
    u"\U0001F900-\U0001F9FF"  # supplemental symbols
    "]+",
    flags=re.UNICODE,
)

def _strip_unwanted_for_tts(text: str) -> str:
    """Remove emojis and specific unwanted characters before TTS.

    This function strips out:
      • All Unicode emoji characters defined by _EMOJI_RE.
      • The substring "json" (case‑insensitive), as it may leak implementation details.
      • Hash symbols (#), asterisks (*), and apostrophes (') which can cause
        unnatural pauses or mispronunciations in speech.

    Args:
        text: The naturalized text string.

    Returns:
        A cleaned string with unwanted elements removed.
    """
    if not text:
        return text
    # Remove emojis
    t = _EMOJI_RE.sub("", text)
    # Remove the substring "json" in a case-insensitive manner
    t = re.sub(r"json", "", t, flags=re.IGNORECASE)
    # Remove specific characters (#, *, and apostrophes)
    t = t.replace("#", "").replace("*", "").replace("'", "").replace('`', '').replace('{', '').replace('}', '')
    return t

def _gtts_to_b64(text: str) -> Optional[str]:
    """Generate a base64‑encoded MP3 from text using gTTS.  Applies naturalization and language selection."""
    if not HAS_GTTS or not text:
        return None
    try:
        # Choose gTTS language based on current reply_lang
        lang_code = _pick_gtts_lang(st.session_state.reply_lang)
        buf = io.BytesIO()
        # Naturalize the text (insert pauses/punctuation) and strip unwanted symbols
        naturalized = _naturalize_for_tts(text)
        cleaned = _strip_unwanted_for_tts(naturalized)
        gTTS(text=cleaned, lang=lang_code).write_to_fp(buf)
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
    ss.setdefault("translate_on", True) 

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = build_or_load_vectorstore(docs=[], rebuild=False)

def _kb_count(vs):
    try:
        return int(vs._collection.count())
    except Exception:
        return 0
st.sidebar.caption(f"📚 KB docs: **{_kb_count(st.session_state.vectorstore)}**")
st.sidebar.caption(f"🗂 Persist dir: {DEFAULT_VS_DIR}")

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
    ss.vectorstore = vs
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

    with st.expander("🛠 RAG Debugger"):
        probe = st.text_input("Probe a query")
        k = st.slider("Top-k", 1, 10, 5)
        if st.button("Run probe"):
            vs = st.session_state.vectorstore
            hits = vs.similarity_search_with_score(probe, k=k)
            rows = []
            for i, (doc, dist) in enumerate(hits, 1):
                rows.append({
                    "rank": i,
                    "distance (↓ better)": round(float(dist), 3),
                    "preview": (getattr(doc, "page_content","") or "")[:160].replace("\n"," ") + "…",
                })
            st.dataframe(rows, use_container_width=True)


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
    user_text = st.chat_input("What are you craving for…")
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
            norm = txt.lower().strip()
            if any(k in norm for k in ["under 30", "sub 30", "30mins", "30 minutes"]):
                quick = (
                    "Here are fast meals under 30 minutes:\n"
                    "1) Garlic shrimp & rice (15 min)\n"
                    "2) Egg & tuna fried rice (20 min)\n"
                    "3) Chicken adobo express (25 min)\n"
                    "4) Veggie stir-fry noodles (20 min)\n"
                    "5) Caprese omelette & toast (12 min)"
                )
                st.session_state.messages.append({"role": "assistant", "content": quick})
                st.session_state.last_bot_text = quick
                if st.session_state.speaker_on:
                    _gtts_to_b64(quick)
                st.rerun()
                st.stop() 
            
            # inside your text submission handler, before calling chat_once / agent
            user_txt = (txt or "").strip().lower()
            pending = st.session_state.get("pending_action")

            if pending and user_txt in YES_WORDS:
                # Execute the pending action
                if pending == "offer_recipe":
                    dish = st.session_state.get("pending_payload", {}).get("dish", "")
                    q = f"Please give me the full recipe for {dish}."
                    ans = chat_once(st.session_state.agent, q, reply_lang=st.session_state.reply_lang)
                    # ans = remove_json_block(ans)
                    st.session_state.messages.append({"role": "assistant", "content": ans})
                    _consume_pending()
                    if st.session_state.get("speaker_on"):
                        _gtts_to_b64(ans)
                    st.rerun(); st.stop()

            elif pending and user_txt in NO_WORDS:
                _consume_pending()
                st.session_state.messages.append({"role":"assistant","content":"No problem — any other dish or cuisine?"})
                st.rerun(); st.stop()

# else: continue normal flow (call agent with the actual user text)



            # Call agent for answer
            ans = chat_once(ss.agent, txt, reply_lang=turn_lang)
            if ss.translate_on and turn_lang != ss.reply_lang:
                ans = ensure_reply_language(ans, ss.reply_lang)
            # Clean any ```json ...```, naked { ... }, stray `json` string, etc.
            # ans = remove_json_block(ans)
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

    # # Friend toggle (translate all replies)
    # if st.button("友", use_container_width=True, help="Toggle translation of replies into chosen language"):
    #     ss.translate_on = not ss.translate_on

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

                # Load the raw transcript and remove JSON nutrition blocks before summarization
                try:
                    transcript_obj = json.loads(transcript_path.read_text(encoding="utf-8"))
                    transcript_raw = " ".join(
                        s.get("text", "").strip() for s in transcript_obj.get("segments", []) if s.get("text")
                    )
                    clean_transcript = remove_json_block(transcript_raw)
                    print(clean_transcript)  # or use st.text(clean_transcript) to display in the app
                except Exception:
                    pass

                summ = summarize_transcript_file.invoke({"transcript_path": str(transcript_path), "target_lang": ss.reply_lang})
                summary_text = summ if isinstance(summ, str) else str(summ)
                summary_text = remove_json_block(summary_text)
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

                    # Load the raw transcript and remove JSON nutrition blocks before summarization
                    try:
                        transcript_obj = json.loads(transcript_path.read_text(encoding="utf-8"))
                        transcript_raw = " ".join(
                            s.get("text", "").strip() for s in transcript_obj.get("segments", []) if s.get("text")
                        )
                        clean_transcript = remove_json_block(transcript_raw)
                        print(clean_transcript)  # or use st.text(clean_transcript) to display in the app
                    except Exception:
                        pass

                    summ = summarize_transcript_file.invoke({"transcript_path": str(transcript_path), "target_lang": ss.reply_lang})
                    summary_text = summ if isinstance(summ, str) else str(summ)
                    summary_text = remove_json_block(summary_text)
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

    # with st.expander("📥 Seed knowledge base (YouTube/Web)"):
    #     default_urls = """\
    # https://www.youtube.com/watch?v=Gyz7s3cFjZU
    # https://www.youtube.com/watch?v=NTpzPZajtEU
    # https://www.youtube.com/watch?v=oPXfLnb8pFo
    # https://www.youtube.com/watch?v=zZNhVv7fmSE
    # https://www.youtube.com/watch?v=VRctr-tviIA
    # https://www.youtube.com/watch?v=Swkq2jc5AnA
    # https://www.youtube.com/watch?v=SkbOKonW6nU
    # https://www.youtube.com/watch?v=K9qJQmOeohU
    # https://www.youtube.com/watch?v=QlDzm8UXbk0
    # https://www.youtube.com/watch?v=u8bdtAUpvlA"""
    #     url_text = st.text_area("Paste one URL per line (YouTube or recipe webpages)", value=default_urls, height=180)
    #     urls = [u.strip() for u in url_text.splitlines() if u.strip()]

    #     col_a, col_b = st.columns(2)
    #     seed_yt  = col_a.button("🎬 Ingest YouTube", use_container_width=True)
    #     seed_web = col_b.button("🌐 Ingest Web Recipes (optional)", use_container_width=True)

    #     st.caption(f"📚 KB docs (approx): **{kb_count()}**")

    #     if seed_yt and urls:
    #         ok = 0
    #         with st.spinner("Seeding YouTube…"):
    #             for i, u in enumerate([u for u in urls if "youtu" in u.lower()], 1):
    #                 st.write(f"{i}. ▶️ {u}")
    #                 try:
    #                     out = summarize_video.invoke({"url": u, "target_lang": st.session_state.reply_lang})
    #                     ok += 1
    #                     st.session_state["kb_docs"] = st.session_state.get("kb_docs", 0) + 1
    #                     st.caption(_as_str(out)[:220] + "…")
    #                 except Exception as e:
    #                     st.warning(f"⚠️ {u} → {e}")
    #         st.success(f"Done: {ok} video(s) ingested. KB now ≈ {kb_count()} docs.")

    #     if seed_web and urls:
    #         ok = 0
    #         with st.spinner("Seeding web recipes…"):
    #             for i, u in enumerate([u for u in urls if "youtu" not in u.lower()], 1):
    #                 st.write(f"{i}. 🌐 {u}")
    #                 try:
    #                     out = ingest_link.invoke({"url": u, "target_lang": st.session_state.reply_lang})
    #                     ok += 1
    #                     st.session_state["kb_docs"] = st.session_state.get("kb_docs", 0) + 1
    #                     st.caption(_as_str(out)[:220] + "…")
    #                 except Exception as e:
    #                     st.warning(f"⚠️ {u} → {e}")
    #         st.success(f"Done: {ok} page(s) ingested. KB now ≈ {kb_count()} docs.")
    # imports (top of file)
    # optional title resolver (graceful if missing)


    # DEFAULT_YT = [
    #     "https://www.youtube.com/watch?v=Gyz7s3cFjZU",
    #     "https://www.youtube.com/watch?v=NTpzPZajtEU",
    #     "https://www.youtube.com/watch?v=oPXfLnb8pFo",
    #     "https://www.youtube.com/watch?v=zZNhVv7fmSE",
    #     "https://www.youtube.com/watch?v=VRctr-tviIA",
    #     "https://www.youtube.com/watch?v=Swkq2jc5AnA",
    #     "https://www.youtube.com/watch?v=SkbOKonW6nU",
    #     "https://www.youtube.com/watch?v=K9qJQmOeohU",
    #     "https://www.youtube.com/watch?v=QlDzm8UXbk0",
    #     "https://www.youtube.com/watch?v=u8bdtAUpvlA",
    # ]

    # with st.expander("🎬 Quick demo: pick a YouTube link"):
    #     title_to_url = _title_map(DEFAULT_YT)
    #     titles = list(title_to_url.keys())
    #     choice_title = st.selectbox("Choose a video", titles, index=0)
    #     choice_url = title_to_url[choice_title]

    #     # optional preview thumbnail
    #     vid = _yt_id(choice_url)
    #     if vid:
    #         st.image(f"https://img.youtube.com/vi/{vid}/hqdefault.jpg", width=320)

    #     c1, c2 = st.columns(2)
    #     btn_tx  = c1.button("📝 Transcript only", use_container_width=True)
    #     btn_sum = c2.button("🧾 Summarize & add to KB", use_container_width=True)

    #     if btn_tx:
    #         try:
    #             res = transcribe_media.invoke({"url_or_path": choice_url})
    #             transcript = res if isinstance(res, str) else str(res)
    #             # transcript = remove_json_block(transcript)
    #             st.session_state.messages.append({"role": "assistant", "content": transcript or "No transcript found."})
    #             st.session_state.last_bot_text = transcript
    #             if st.session_state.get("speaker_on") and transcript:
    #                 _gtts_to_b64(transcript)
    #             st.rerun()   # if in a function, return afterwards; at top-level use st.stop()
    #             st.stop()
    #         except Exception as e:
    #             st.warning(f"Transcript error: {e}")

    #     if btn_sum:
    #         try:
    #             out = summarize_video.invoke({"url": choice_url, "target_lang": st.session_state.reply_lang})
    #             summary = out if isinstance(out, str) else str(out)
    #             # summary = remove_json_block(summary)
    #             st.session_state.messages.append({"role": "assistant", "content": summary or "No summary available."})
    #             st.session_state["kb_docs"] = st.session_state.get("kb_docs", 0) + 1
    #             st.session_state.last_bot_text = summary
    #             if st.session_state.get("speaker_on") and summary:
    #                 _gtts_to_b64(summary)
    #             st.rerun()   # if in a function, return afterwards; at top-level use st.stop()
    #             st.stop()
    #         except Exception as e:
    #             st.warning(f"Summarize error: {e}")

    # with st.expander("🚀 Seed entire demo list"):
    #     if st.button("Seed all videos (summarize + upsert)", use_container_width=True):
    #         ok = 0
    #         prog = st.progress(0)
    #         for i, url in enumerate(DEFAULT_YT, 1):
    #             try:
    #                 summarize_video.invoke({"url": url, "target_lang": st.session_state.reply_lang})
    #                 ok += 1
    #                 st.session_state["kb_docs"] = st.session_state.get("kb_docs", 0) + 1
    #             except Exception as e:
    #                 st.warning(f"⚠️ {url} → {e}")
    #             prog.progress(int(i / len(DEFAULT_YT) * 100))
    #         st.success(f"Seeded {ok}/{len(DEFAULT_YT)} videos. KB ≈ {st.session_state.get('kb_docs', 0)} docs.")

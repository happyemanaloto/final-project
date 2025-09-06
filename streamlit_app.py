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
# --- ensure modern sqlite for Chroma ---
try:
    import pysqlite3  # provided by pysqlite3-binary
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass
# --------------------------------------

import os, io, re, json, time, base64, tempfile, hashlib
from pathlib import Path
from typing import Dict, List, Optional
try:
    from yt_dlp import YoutubeDL
    _HAS_YTDLP = True
except Exception:
    YoutubeDL = None
    _HAS_YTDLP = False
        
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
from bot.data import load_all_docs, build_or_load_vectorstore, DEFAULT_VS_DIR
from bot.tools import (
    bind_vectorstore,
    bind_session_hooks,
    transcribe_local_media,
    summarize_transcript_file, 
    ingest_link,
    summarize_video,
    transcribe_media,
    estimate_nutrition,
    make_shopping_list
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
YES_WORDS = {"yes","yeah","yep","sure","ok","okay","please","go ahead","do it","yup", "yessir", "please do", "yes please"}
NO_WORDS  = {"no","nope","nah","not now","later", "no thanks", "nah thanks"}
PLEASE_WORDS = {"please", "pls", "kindly", "please continue", "go on"}

def _recent_dialog(n: int = 5) -> str:
    """Return a compact transcript of the last n messages (oldest → newest)."""
    msgs = st.session_state.get("messages", [])[-n:]
    out = []
    for m in msgs:
        role = m.get("role", "user")
        text = (m.get("content") or "").strip().replace("\n", " ")
        out.append(f"{role}: {text}")
    return "\n".join(out)

def _last_assistant_question(n: int = 5) -> str:
    """Find the most recent assistant question in the last n messages."""
    for m in reversed(st.session_state.get("messages", [])[-n:]):
        if m.get("role") == "assistant":
            t = (m.get("content") or "").strip()
            if "?" in t:
                return t
    return ""

def _set_offer(task: str, payload: dict | None = None):
    """Remember the last offer so a bare 'yes' has meaning."""
    st.session_state["last_offer"] = {"task": task, "payload": payload or {}, "ts": time.time()}

def _consume_offer():
    st.session_state.pop("last_offer", None)

def _answer_with_context(user_intent_fallback: str = "continue"):
    """
    Build a context-rich prompt that includes:
      • last 5 exchanges
      • last transcript (if any)
      • the latest assistant question (if any)
    """
    ss = st.session_state
    dialog = _recent_dialog(5)
    last_q = _last_assistant_question(5)
    transcript = (ss.get("ctx_text") or "").strip()
    source = ss.get("ctx_source") or "recent"
    # Prefer transcript when available
    if transcript:
        q = _build_augmented_query(
            transcript,
            source_label=source,
            task_prompt=f"Using the transcript and the last 5 messages below, {user_intent_fallback}.",
            user_q=f"Recent dialog:\n{dialog}\n\nLatest assistant question (if any): {last_q or '—'}"
        )
    else:
        q = (
            "Use the following recent dialog as authoritative context before answering. "
            "If the user said 'yes', do what was last offered.\n\n"
            f"{dialog}\n\n"
            f"User intent: {user_intent_fallback}."
        )
    # Stick to the reply language currently selected
    return chat_once(ss.agent, q, reply_lang=ss.reply_lang)

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
import difflib

def _extract_dishes(text: str) -> list[str]:
    """Heuristically pull dish names from a user/assistant turn."""
    if not text:
        return []
    t = text.strip()
    # Prefer “title line” if assistant just wrote a recipe card.
    first = t.splitlines()[0].strip(": ").lower()
    # Common trigger patterns
    pats = [
        r"(?:recipe for|make|cook|how to make)\s+([A-Za-z][\w\s\-']{2,})",
        r"\b(?:pasta|adobo|biryani|ramen|pad thai|fried rice|carbonara|bolognese|aglio e olio|pesto|alfredo|noodles)\b",
    ]
    found: list[str] = []
    for p in pats:
        m = re.search(p, t, flags=re.I)
        if m:
            g = m.group(1) if m.lastindex else m.group(0)
            if g: found.append(g.strip().title())
    # Title-ish first line (e.g., "Spaghetti Aglio e Olio — Serves 4")
    if len(first) <= 60 and any(k in first for k in ["pasta","adobo","biry","ramen","rice","noodle","spag","penne","pad thai","pesto","carbon","bolo","aglio","olio","alfredo"]):
        found.append(first.title())
    # Unique, preserve order
    out = []
    for x in found:
        if x not in out:
            out.append(x)
    return out[:3]

def _update_recent_dishes(new_items: list[str]):
    ss = st.session_state
    if not new_items: return
    lst = ss.get("recent_dishes", [])
    for dish in new_items:
        # move-to-front if exists, else insert
        if dish in lst:
            lst.remove(dish)
        lst.insert(0, dish)
    ss["recent_dishes"] = lst[:3]

def _best_hit_for_title(title: str) -> dict:
    """Pick the closest 'hit' (recipe struct) for a title from session hits."""
    hits = ss.session.get_hits() if ss.session else []
    if not hits:
        return {}
    # direct match
    for h in hits:
        if (h.get("title") or "").strip().lower() == title.strip().lower():
            return h
    # fuzzy title match
    titles = [(h, h.get("title","")) for h in hits]
    best = max(titles, key=lambda tup: difflib.SequenceMatcher(None, title.lower(), (tup[1] or "").lower()).ratio())
    return best[0] if best and best[1] else {}


def _ensure_hit_for_dish(title: str) -> dict:
    """Ensure we have a minimal recipe 'hit' (title + ingredients) for a dish name."""
    hit = _best_hit_for_title(title)
    if hit and (hit.get("ingredients") or hit.get("ingredients_display")):
        return hit

    # Ask the model for a minimal ingredient list, then cache it as a session hit
    prompt = ChatPromptTemplate.from_messages([
        ("system", "List 8–12 core ingredients for the dish. Plain list, one per line. No steps."),
        ("human", f"Dish: {title}")
    ])
    out = (prompt | llm_zero(temperature=0)).invoke({})
    ings = [ln.strip("-• ").strip() for ln in out.content.splitlines() if ln.strip()][:12]
    minimal = {
        "id": f"user:{title}",
        "title": title,
        "url": "",
        "source": "user",
        "ingredients": ings,
        "ingredients_display": ings,
        "steps": [],
    }
    if ss.session:
        ss.session.remember_hits([minimal])
    return minimal

def _format_nutrition(answer_json: str) -> str:
    """Format the JSON-ish nutrition text into a single clean line."""
    try:
        data = json.loads(answer_json)
        parts = []
        def g(k, unit): 
            v = data.get(k)
            if v is None: return None
            try: v = float(v)
            except: pass
            return f"{k.capitalize()}: {v}{unit}"
        parts.append(g("calories"," kcal"))
        parts.append(g("protein"," g"))
        parts.append(g("carbohydrates"," g"))
        parts.append(g("fat"," g"))
        if data.get("fiber") is not None: parts.append(g("fiber"," g"))
        if data.get("sugar") is not None: parts.append(g("sugar"," g"))
        if data.get("sodium") is not None: parts.append(g("sodium"," mg"))
        parts = [p for p in parts if p]
        return " · ".join(parts)
    except Exception:
        # fallback: strip code fences/braces if model returned text
        s = re.sub(r"```[\s\S]*?```", "", answer_json)
        s = re.sub(r"^\s*\{|\}\s*$", "", s).strip()
        return s
# --- Intent detection (mic / uploads / chat) ---
INTENT_PATTERNS = {
    "calories": [
        r"\bcalories?\b", r"\bcalorie count\b", r"\bmacros?\b",
        r"\bnutrition(?:al)?\b", r"\bper[-\s]?serving\b"
    ],
    "shopping_list": [
        r"\bshopping list\b", r"\bgrocery list\b",
        r"\blista de compra(s)?\b", r"\bmarket list\b"
    ],
    "recipe": [
        r"\brecipe\b", r"\breceta\b",
        r"\bhow (?:to|do i) (?:make|cook)\b",
        r"\bsteps?\b"
    ],
}

def _detect_intent(text: str) -> str:
    t = (text or "").lower()
    for name, pats in INTENT_PATTERNS.items():
        if any(re.search(p, t) for p in pats):
            return name
    return ""

def _build_augmented_query(transcript: str, source_label: str, task_prompt: str, user_q: str | None = None) -> str:
    base = (
        "Prefer the transcript facts below. If minor details are missing, fill sensible defaults; "
        "if the key fact truly isn’t present, say so briefly. "
        "Respond in plain text only (no JSON, no code blocks).\n\n"
        f"--- transcript ({source_label}) ---\n{transcript}\n--- end transcript ---\n\n"
    )
    return base + (user_q or task_prompt)

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
    ss.setdefault("ctx_text", "") 
    ss.setdefault("ctx_source", "") 
    ss.setdefault("recent_dishes", [])  # list of recently discussed dishes

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
            _update_recent_dishes(_extract_dishes(txt))
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
                _update_recent_dishes(_extract_dishes(conf_msg))
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


            txt_norm = (user_text or "").strip().lower()
            offer = st.session_state.get("last_offer")
            pending = st.session_state.get("pending_action")

            # YES / PLEASE = accept last offer (or continue context)
            if (txt_norm in YES_WORDS) or (txt_norm in PLEASE_WORDS):
                handled = False

                # 1) Honor your existing pending_action first
                if pending == "offer_recipe":
                    dish = st.session_state.get("pending_payload", {}).get("dish", "")
                    q = f"Please give me the full recipe for {dish}. Include brief tips."
                    ans = chat_once(st.session_state.agent, _build_augmented_query(st.session_state.get("ctx_text",""), st.session_state.get("ctx_source",""), q), reply_lang=st.session_state.reply_lang)
                    st.session_state.messages.append({"role": "assistant", "content": ans})
                    _consume_pending(); _consume_offer()
                    st.rerun(); st.stop()

                # 2) If we have a remembered 'offer' (e.g., after a transcript), continue from it
                if offer:
                    ans = _answer_with_context(user_intent_fallback="please carry on with your last offer")
                    st.session_state.messages.append({"role": "assistant", "content": ans})
                    _consume_offer()
                    st.rerun(); st.stop()

                # 3) Fallback: continue based on recent dialog
                ans = _answer_with_context(user_intent_fallback="continue helpfully")
                st.session_state.messages.append({"role": "assistant", "content": ans})
                st.rerun(); st.stop()

            # NO = cancel the offer and ask what next
            if txt_norm in NO_WORDS:
                _consume_pending(); _consume_offer()
                st.session_state.messages.append({"role": "assistant", "content": "No problem — what would you like to do next?"})
                st.rerun(); st.stop()

            # Call agent for answer
            ans = chat_once(ss.agent, txt, reply_lang=turn_lang)
            if ss.translate_on and turn_lang != ss.reply_lang:
                ans = ensure_reply_language(ans, ss.reply_lang)
            # Clean any ```json ...```, naked { ... }, stray `json` string, etc.
            # ans = remove_json_block(ans)
            # Append assistant reply
            ss.messages.append({"role": "assistant", "content": ans})
            _update_recent_dishes(_extract_dishes(ans))
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
    
    # # Speaker toggle (shows ON/OFF; pre‑generates TTS when turning on)
    # speaker_label = "🔊 Speaker: ON" if ss.speaker_on else "🔈 Speaker: OFF"
    # if st.button(speaker_label, use_container_width=True):
    #     ss.speaker_on = not ss.speaker_on
    #     if ss.speaker_on:
    #         # Pre-generate audio for the most recent bot reply if available
    #         if ss.last_bot_text:
    #             b64 = _gtts_to_b64(ss.last_bot_text)
    #             if b64:
    #                 ss.last_tts_b64 = b64
    #     else:
    #         ss.last_tts_b64 = None
    #     st.rerun()
        # Speaker as a toggle with a stable key
    new_speaker = st.toggle("🔈 Auto-speak last reply", value=ss.speaker_on, key="speaker_toggle")
    if new_speaker != ss.speaker_on:
        ss.speaker_on = new_speaker
        if ss.speaker_on and ss.last_bot_text:
            b64 = _gtts_to_b64(ss.last_bot_text)
            ss.last_tts_b64 = b64 if b64 else None
        else:
            ss.last_tts_b64 = None
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
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
                ss.messages.append({"role": "user", "content": "(Uploaded media)"})
                ss.messages.append({"role": "assistant", "content": summary_text})
                _update_recent_dishes(_extract_dishes(summary_text))
                # Offer a next step + remember it
                offer = "Would you like the full recipe for any dish mentioned, or a shopping list, or a calorie estimate?"
                ss.messages.append({"role": "assistant", "content": offer})
                _set_offer(task="followup_from_transcript", payload={"source": ss.get("ctx_source"), "hint": _extract_dishes(summary_text)})
                ss.last_bot_text = summary_text
                if ss.speaker_on:
                    b64tts = _gtts_to_b64(summary_text)
                    if b64tts:
                        ss.last_tts_b64 = b64tts
                st.rerun()

                # Determine language for this turn
                det = detect_language(summary_text)
                turn_lang = ss.reply_lang if ss.translate_on else (
                    ss.session.reply_lang if (det == ss.session.reply_lang or len(clean_transcript) < 60) else det
                )

                # Call agent for answer
                ans = chat_once(ss.agent, summary_text, reply_lang=turn_lang)
                if ss.translate_on and turn_lang != ss.reply_lang:
                    ans = ensure_reply_language(ans, ss.reply_lang)
                # Append assistant reply
                ss.messages.append({"role": "assistant", "content": ans})
                _update_recent_dishes(_extract_dishes(ans))
                ss.last_bot_text = ans
                # Pre‑generate TTS if speaker is on
                if ss.speaker_on:
                    b64 = _gtts_to_b64(ans)
                    if b64:
                        ss.last_tts_b64 = b64
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
            # pcols = st.columns(2)
            # if pcols[0].button("▶️ Play Recording", use_container_width=True):
            #     _play_audio_controls(ss.rec_b64, ss.rec_mime)
            # if pcols[1].button("⏹ Stop Playback", use_container_width=True):
            #     st.rerun()
            # # Transcribe button
            # # if st.button("📝 Transcribe", use_container_width=True):
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
                        s.get("text", "").strip()
                        for s in transcript_obj.get("segments", [])
                        if s.get("text")
                    )
                    clean_transcript = remove_json_block(transcript_raw)
                except Exception:
                    clean_transcript = ""

                # (Optional but useful) short summary for the chat UI (cleaned)
                try:
                    summ = summarize_transcript_file.invoke({
                        "transcript_path": str(transcript_path),
                        "target_lang": ss.reply_lang
                    })
                    summary_text = summ if isinstance(summ, str) else str(summ)
                    summary_text = remove_json_block(summary_text)
                except Exception:
                    summary_text = ""

                # Update context and a stable signature (prevents double-handling on rerun)
                ss.ctx_text = (clean_transcript or "")[:4000]
                ctx_id = hashlib.sha1((ss.ctx_text or "").encode("utf-8")).hexdigest()[:10]
                ss.ctx_source = f"mic:{ctx_id}"
                if ss.get("handled_ctx_sig") == ctx_id:
                    st.stop()
                ss.handled_ctx_sig = ctx_id

                # Try to guess a dish name from transcript or summary → feeds “recent dishes” box
                dish_candidates = _extract_dishes(clean_transcript) or _extract_dishes(summary_text)
                dish_hint = dish_candidates[0] if dish_candidates else ""
                _update_recent_dishes(dish_candidates)

                # Detect intent from transcript first, then summary as fallback
                intent = _detect_intent(clean_transcript) or _detect_intent(summary_text)

                # If no obvious intent → show summary and invite follow-up (normal chat over transcript)
                if not intent:
                    if summary_text:
                        ss.messages.append({"role": "user", "content": "🎙️ (Mic upload) Please summarize."})
                        ss.messages.append({"role": "assistant", "content": clean_transcript})
                        ss.last_bot_text = clean_transcript
                        if ss.speaker_on:
                            b64tts = _gtts_to_b64(clean_transcript)
                            if b64tts:
                                ss.last_tts_b64 = b64tts
                    ss.messages.append({
                        "role": "assistant",
                        "content": "Got it. Ask me anything about this recording (recipe, calorie count, or a shopping list)."
                    })
                    ss.handled_ctx_sig = ctx_id
                    st.rerun()
                # Determine language for this turn
                det = detect_language(clean_transcript)
                turn_lang = ss.reply_lang if ss.translate_on else (
                    ss.session.reply_lang if (det == ss.session.reply_lang or len(clean_transcript) < 60) else det
                )
                # Call agent for answer
                ans = chat_once(ss.agent, clean_transcript, reply_lang=turn_lang)
                if ss.translate_on and turn_lang != ss.reply_lang:
                    ans = ensure_reply_language(ans, ss.reply_lang)
                # Append assistant reply
                ss.messages.append({"role": "assistant", "content": ans})
                _update_recent_dishes(_extract_dishes(ans))
                ss.last_bot_text = ans
                # Pre‑generate TTS if speaker is on
                if ss.speaker_on:
                    b64 = _gtts_to_b64(ans)
                    if b64:
                        ss.last_tts_b64 = b64
                st.rerun()

                # Build a “user bubble” preview for the mic request (nice chat UX)
                preview = " ".join(ss.ctx_text.split())[:160]
                label = "calorie count" if intent == "calories" else ("shopping list" if intent == "shopping_list" else "recipe")
                ss.messages.append({"role": "user", "content": f"🎙️ Mic → {label} request\n> {preview}…"})

            except Exception as e:
                ss.messages.append({"role": "assistant", "content": ensure_reply_language(f"Mic processing error: {e}", ss.reply_lang)})
                st.rerun()
    else:
        st.caption("Install 'audio-recorder-streamlit' to enable mic recording.")





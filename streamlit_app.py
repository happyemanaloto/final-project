# streamlit_app.py
# Quick Streamlit wrapper for Kusina bot
# ─────────────────────────────────────────────────────────────────────────────
# Prereqs:
#   pip install streamlit langchain langchain-openai langchain-community chromadb
#   pip install rapidfuzz youtube-transcript-api yt-dlp openai-whisper  # optional (for whisper fallback)
#   pip install python-dotenv
#
# Run:
#   streamlit run streamlit_app.py
#
# Environment (.env or Streamlit secrets):
#   OPENAI_API_KEY=sk-...
#   CHEF_TRANSCRIBE_BACKEND=internal   # recommended for speed while testing
#   CHEF_TRANSCRIBE=api_only           # set to "whisper" to allow audio download fallback

from __future__ import annotations
import os, json, time, tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv, find_dotenv

# Load .env (works both locally and on Streamlit Cloud if you add secrets)
_ = load_dotenv(find_dotenv(filename=".env", usecwd=True))

# ── Optional: read from Streamlit secrets (overrides .env if present)
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ── Your project imports (assumes your refactor layout)
from bot.data import load_all_docs, build_or_load_vectorstore
from bot.tools import (
    bind_vectorstore, bind_session_hooks,
    transcribe_youtube_best_effort, summarize_video, qa_video,
    make_shopping_list, estimate_nutrition,
)
from bot.session import SessionMemory
from bot.agent import build_agent, chat_once, SYSTEM
from bot.nlp import llm_zero, ensure_reply_language

# ─────────────────────────────────────────────────────────────────────────────
# Caching: build data & VS once per process
@st.cache_resource(show_spinner=True)
def _load_assets(rebuild_vs: bool = False):
    docs = load_all_docs()
    vs = build_or_load_vectorstore(docs, rebuild=rebuild_vs)
    return docs, vs

def _init_session():
    """Create SessionMemory and bind hooks for tools once per Streamlit session."""
    if "session" not in st.session_state:
        user_id = f"web-{int(time.time())}"
        st.session_state.session = SessionMemory(user_id=user_id)
        # tools access this session's hits via hooks
        bind_session_hooks(
            get_hits=st.session_state.session.get_hits,
            set_hits=st.session_state.session.remember_hits
        )

    if "reply_lang" not in st.session_state:
        st.session_state.reply_lang = os.getenv("CHEF_DEFAULT_LANG", "en")

    if "agent" not in st.session_state:
        # Build LLM + agent
        llm = llm_zero()
        st.session_state.agent = build_agent(llm, st.session_state.session)

    if "chat" not in st.session_state:
        st.session_state.chat = []  # [{role: "user"/"assistant", "text": "..."}]

# ─────────────────────────────────────────────────────────────────────────────
def _render_sidebar():
    st.sidebar.header("⚙️ Settings")

    # LangSmith toggle
    use_langsmith = st.sidebar.checkbox("Enable LangSmith tracing", value=False, help="Requires LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if use_langsmith else "false"
    os.environ["LANGCHAIN_PROJECT"] = st.sidebar.text_input("LangSmith project", value="kusina-bot")

    # Reply language
    st.sidebar.subheader("Reply language")
    st.session_state.reply_lang = st.sidebar.text_input(
        "ISO / name (e.g., en, tl, es)", value=st.session_state.reply_lang
    )

    # Transcription backend (just a quick control for testing)
    st.sidebar.subheader("Transcription")
    backend = st.sidebar.selectbox("Backend", ["internal", "auto", "external"], index=0)
    os.environ["CHEF_TRANSCRIBE_BACKEND"] = backend
    mode = st.sidebar.selectbox("Mode", ["api_only", "whisper"], index=0)
    os.environ["CHEF_TRANSCRIBE"] = mode

    st.sidebar.subheader("Vectorstore")
    if st.sidebar.button("Rebuild vectorstore"):
        # clear cache + reload
        _load_assets.clear()
        st.cache_resource.clear()
        st.experimental_rerun()

# ─────────────────────────────────────────────────────────────────────────────
def _handle_user_turn(user_text: str):
    if not user_text.strip():
        return
    st.session_state.chat.append({"role": "user", "text": user_text})

    try:
        answer = chat_once(
            st.session_state.agent,
            user_text=user_text,
            reply_lang=st.session_state.reply_lang
        )
    except Exception as e:
        answer = f"Sorry, something went wrong: {e}"

    st.session_state.chat.append({"role": "assistant", "text": answer})

# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Kusina Bot", page_icon="🍳", layout="centered")

    st.title("🍳 Kusina Bot — Streamlit Demo")
    st.caption("Kitchen buddy & nutrition coach. Paste a link or just chat.")

    _init_session()
    _render_sidebar()

    # Load assets & bind vectorstore into tools
    with st.spinner("Loading recipes & vector store…"):
        docs, vs = _load_assets(rebuild_vs=False)
        bind_vectorstore(vs)
    st.success(f"Recipes loaded: {len(docs)}")

    # ── Quick actions row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Shopping list 🛒"):
            out = make_shopping_list.invoke({"recipes": None, "servings_multiplier": 1.0, "target_lang": st.session_state.reply_lang})
            txt = out if isinstance(out, str) else str(out)
            st.session_state.chat.append({"role": "assistant", "text": ensure_reply_language(txt, st.session_state.reply_lang)})
    with col2:
        if st.button("Calories 🔢"):
            # calories short-circuit happens in agent, but we can nudge by asking
            _handle_user_turn("calorie count for these?")
    with col3:
        if st.button("Tagalog 🇵🇭"):
            st.session_state.reply_lang = "tl"
            st.toast("Okay! I’ll reply in Tagalog.")
    with col4:
        if st.button("English 🇬🇧"):
            st.session_state.reply_lang = "en"
            st.toast("Okay! I’ll reply in English.")

    # ── Link helpers
    st.subheader("Link tools")
    link = st.text_input("YouTube / recipe link")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Summarize link"):
            if not link.strip():
                st.warning("Paste a link first.")
            else:
                with st.spinner("Summarizing…"):
                    out = summarize_video.invoke({"url": link, "target_lang": st.session_state.reply_lang})
                    txt = out if isinstance(out, str) else str(out)
                    st.session_state.chat.append({"role": "assistant", "text": ensure_reply_language(txt, st.session_state.reply_lang)})
    with c2:
        ask = st.text_input("Ask about this link (Q&A)")
        if st.button("Ask"):
            if not link.strip():
                st.warning("Paste a link first.")
            elif not ask.strip():
                st.warning("Type a question for the video.")
            else:
                with st.spinner("Answering from transcript…"):
                    out = qa_video.invoke({"url": link, "question": ask, "target_lang": st.session_state.reply_lang})
                    txt = out if isinstance(out, str) else str(out)
                    st.session_state.chat.append({"role": "assistant", "text": ensure_reply_language(txt, st.session_state.reply_lang)})

    # ── Optional: local media upload (audio/video) for Whisper
    st.subheader("Upload audio/video (optional, Whisper)")
    up = st.file_uploader("MP3 / WAV / MP4", type=["mp3", "wav", "m4a", "mp4", "mov", "webm"])
    if up is not None and st.button("Transcribe & summarize upload"):
        tmp = Path(tempfile.gettempdir()) / f"kusina_upload_{int(time.time())}_{up.name}"
        tmp.write_bytes(up.read())
        # Reuse the same summarize path: our transcriber accepts local paths
        with st.spinner("Transcribing…"):
            # treat like a URL to local file
            out = summarize_video.invoke({"url": str(tmp), "target_lang": st.session_state.reply_lang})
            txt = out if isinstance(out, str) else str(out)
            st.session_state.chat.append({"role": "assistant", "text": ensure_reply_language(txt, st.session_state.reply_lang)})

    # ── Chat box
    st.subheader("Chat")
    user_text = st.text_area("Type your message", height=90, placeholder="e.g., '30-minute Filipino dinners', 'calorie count of that', 'shopping list please', 'translate to Tagalog'")
    if st.button("Send", type="primary"):
        _handle_user_turn(user_text)

    # ── Render history
    for turn in st.session_state.chat:
        if turn["role"] == "user":
            st.markdown(f"**You:** {turn['text']}")
        else:
            st.markdown(turn["text"])

    st.divider()
    st.caption("Tip: switch transcription to `whisper` in the sidebar if a video has no official transcript.")

if __name__ == "__main__":
    main()

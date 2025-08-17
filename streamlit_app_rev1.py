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

# streamlit_app.py (snippet)
import streamlit as st
from bot.yt_transcriber import transcribe_youtube_streamlit, transcribe_youtube_best_effort

import streamlit as st
from dotenv import load_dotenv, find_dotenv

import re
from collections import Counter


# Load .env (works both locally and on Streamlit Cloud if you add secrets)
_ = load_dotenv(find_dotenv(filename=".env", usecwd=True))

# ── Optional: read from Streamlit secrets (overrides .env if present)
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ── Your project imports (assumes your refactor layout)
from bot.data import load_all_docs, build_or_load_vectorstore
from bot.tools import (
    bind_vectorstore, bind_session_hooks,
    summarize_video, qa_video,
    make_shopping_list, estimate_nutrition,
)
from bot.session import SessionMemory
from bot.agent import build_agent, chat_once, SYSTEM
from bot.nlp import llm_zero, ensure_reply_language

from bot.voice_sanity import record_to_wav, transcribe_whisper
from gtts import gTTS
import io
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

def ui_transcribe_block():
    st.subheader("Paste a YouTube Link")
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

    if st.button("Transcribe"):
        if not url.strip():
            st.warning("Please paste a valid YouTube link.")
            return

        # Option A: Streamlit-friendly (with spinners/status)
        payload = transcribe_youtube_streamlit(
            st,
            url_or_id=url,
            cache_dir="data/recipes",
            max_minutes=20,
            download_timeout_s=60,
            whisper_model="base",   # 'tiny' for faster CPU demos
        )

        if payload:
            # You now have payload["transcript"]["text"]
            st.write(f"**Title:** {payload['title']}")
            st.write(f"**Duration:** {payload['duration_hms']}")
            with st.expander("Show raw transcript"):
                st.write(payload["transcript"]["text"])
            # TODO: call your summarizer / QA / gTTS here using payload

# ─────────────────────────────────────────────────────────────────────────────
def ui_mic_transcribe_block():
    st.subheader("🎙️ Mic → Transcript → Summary → Q&A")

    mic_index = st.number_input("Mic index (-1 = auto)", value=-1, step=1)
    seconds = st.slider("Record seconds", 2, 15, 8)
    model = st.selectbox("Whisper model", ["tiny","base","small","small.en","medium","medium.en"], index=3)
    lang_hint = st.text_input("Language hint (optional, e.g., en, tl, es)", "")

    if st.button("Record & Transcribe"):
        with st.spinner("Recording…"):
            wav_path = record_to_wav(
                seconds=seconds,
                samplerate=16000,
                mic_index=(None if mic_index < 0 else int(mic_index)),
            )
            st.success(f"Saved to: {wav_path}")

        with st.spinner("Transcribing (accuracy mode)…"):
            text = transcribe_whisper(
                wav_path,
                model_name=model,
                language=(lang_hint or None),
            )

        if not text:
            st.warning("No speech recognized. Try a different mic index or speak closer to the mic.")
            return

        st.markdown("**Transcript:**")
        st.write(text)
        st.session_state["last_mic_text"] = text

        # --- Summarize (Agent if available, else offline) ---
        with st.spinner("Summarizing…"):
            summary = None
            try:
                agent = st.session_state.get("agent")
                if agent and hasattr(agent, "summarize"):
                    summary = agent.summarize(text)  # your agent method
                elif agent and hasattr(agent, "summarize_text"):
                    summary = agent.summarize_text(text)
            except Exception as e:
                st.info(f"Agent summarization unavailable; using offline summary. Details: {e}")

            if not summary:
                summary = summarize_text_offline(text, max_sentences=5)

        st.markdown("**Summary:**")
        st.write(summary)

        # --- Q&A (Agent if available, else offline) ---
        st.markdown("**Ask about your recording:**")
        user_q = st.text_input("Your question", key="mic_qa_input")
        if user_q:
            with st.spinner("Answering…"):
                answer = None
                try:
                    agent = st.session_state.get("agent")
                    if agent and hasattr(agent, "qa_over_text"):
                        answer = agent.qa_over_text(text, user_q)
                    elif agent and hasattr(agent, "ask"):
                        # generic ask with context
                        answer = agent.ask(f"Context:\n{text}\n\nQuestion: {user_q}\nAnswer briefly:")
                except Exception as e:
                    st.info(f"Agent Q&A unavailable; using offline lookup. Details: {e}")

                if not answer:
                    answer = qa_text_offline(text, user_q)

            st.markdown("**Answer:**")
            st.write(answer)

        # Optional: speak the transcript back via gTTS (browser-friendly)
        try:
            buf = io.BytesIO()
            gTTS(text=text, lang=(lang_hint or "en")).write_to_fp(buf)
            buf.seek(0)
            st.audio(buf, format="audio/mp3")
        except Exception as e:
            st.info(f"TTS playback skipped: {e}")

# ─────────────────────────────────────────────────────────────────────────────
def summarize_text_offline(text: str, max_sentences: int = 5) -> str:
    """
    Lightweight extractive summary (no LLM):
    - Scores sentences by word frequency (minus stopwords)
    - Returns top N sentences in original order
    """
    if not text:
        return ""
    # crude sentence split
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= max_sentences:
        return text.strip()

    # very small stoplist
    stop = set("""the a an and or to of in on for from with at by as is are was were be been being it this that those these you your i we they them he she his her their our not no yes do does did have has had into about over under up down out more most much many very just can could should would there here then than so such if else when while where who whom whose which what""".split())
    words = re.findall(r"[a-zA-Z']+", text.lower())
    freqs = Counter(w for w in words if w not in stop and len(w) > 2)
    if not freqs:
        return " ".join(sentences[:max_sentences])

    scores = []
    for i, s in enumerate(sentences):
        toks = re.findall(r"[a-zA-Z']+", s.lower())
        score = sum(freqs.get(t, 0) for t in toks) / (len(toks) + 1e-6)
        scores.append((score, i, s))

    # pick top-N by score, then restore original order
    top = sorted(sorted(scores, key=lambda x: x[1])[:0], key=lambda x: x[0], reverse=True)  # keep linter happy
    top = sorted(sorted(scores, key=lambda x: x[0], reverse=True)[:max_sentences], key=lambda x: x[1])
    return " ".join(s for _, _, s in top)

# ─────────────────────────────────────────────────────────────────────────────
def qa_text_offline(text: str, question: str, max_sentences: int = 6) -> str:
    """
    Very simple keyword Q&A:
    - Pulls sentences with overlapping keywords
    - Returns the best-matching snippets as an “answer”
    """
    if not text or not question:
        return ""
    qs = set(re.findall(r"[a-zA-Z']+", question.lower()))
    qs = {q for q in qs if len(q) > 2}
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    scored = []
    for i, s in enumerate(sentences):
        toks = set(re.findall(r"[a-zA-Z']+", s.lower()))
        overlap = len(qs & toks)
        if overlap > 0:
            scored.append((overlap, i, s))
    if not scored:
        return "I couldn’t find a direct answer in the transcript. Try rephrasing."
    top = sorted(scored, key=lambda x: (-x[0], x[1]))[:max_sentences]
    return " ".join(s for _, _, s in top)

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

    # Tabs for input modes
    tab_y, tab_m = st.tabs(["YouTube Link", "Mic Input"])

    with tab_y:
        ui_transcribe_block()

    with tab_m:
        ui_mic_transcribe_block()

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

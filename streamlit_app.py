# streamlit_app.py — single-tab, mobile-style chat + single YouTube box + (optional) mic
from __future__ import annotations
import os, json, time, tempfile, io, re
from pathlib import Path
from collections import Counter

import streamlit as st
from dotenv import load_dotenv, find_dotenv
from gtts import gTTS

# ── Load .env / secrets
_ = load_dotenv(find_dotenv(filename=".env", usecwd=True))
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ── Your project imports
from bot.data import load_all_docs, build_or_load_vectorstore
from bot.tools import (
    bind_vectorstore, bind_session_hooks,
    summarize_video, qa_video,
    make_shopping_list, estimate_nutrition,
)
from bot.session import SessionMemory
from bot.agent import build_agent, chat_once
from bot.nlp import llm_zero, ensure_reply_language

# Optional mic (commented by default). Install to enable:
#   pip install streamlit-webrtc
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    HAS_WEBRTC = True
except Exception:
    HAS_WEBRTC = False

# ─────────────────────────────────────────────────────────────────────────────
# Cache assets (recipes + vectorstore)
@st.cache_resource(show_spinner=True)
def _load_assets(rebuild_vs: bool = False):
    docs = load_all_docs()
    vs = build_or_load_vectorstore(docs, rebuild=rebuild_vs)
    return docs, vs

def _init_session():
    if "session" not in st.session_state:
        user_id = f"web-{int(time.time())}"
        st.session_state.session = SessionMemory(user_id=user_id)
        bind_session_hooks(
            get_hits=st.session_state.session.get_hits,
            set_hits=st.session_state.session.remember_hits
        )

    if "reply_lang" not in st.session_state:
        st.session_state.reply_lang = os.getenv("CHEF_DEFAULT_LANG", "en")

    if "agent" not in st.session_state:
        llm = llm_zero()
        st.session_state.agent = build_agent(llm, st.session_state.session)

    if "messages" not in st.session_state:
        st.session_state.messages = []   # [{role: "user"/"assistant", "content": "..."}]

# ─────────────────────────────────────────────────────────────────────────────
# Minimal styles to mimic a phone chat window
def _inject_css():
    st.markdown("""
    <style>
      .app-wrap {max-width: 420px; margin: 0 auto;}
      .phone-frame {
        border: 1px solid #ddd; border-radius: 18px; padding: 10px 12px 80px;
        min-height: 65vh; background: #fafafa; position: relative;
      }
      .footer-bar {
        position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
        max-width: 420px; width: calc(100% - 24px);
        background: white; border: 1px solid #ddd; border-radius: 12px;
        padding: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.06);
      }
      /* tighten chat bubbles a bit */
      .stChatMessage {padding-top: 6px; padding-bottom: 6px;}
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
def _handle_user_turn(user_text: str):
    if not user_text.strip():
        return
    st.session_state.messages.append({"role": "user", "content": user_text})
    try:
        answer = chat_once(
            st.session_state.agent,
            user_text=user_text,
            reply_lang=st.session_state.reply_lang
        )
    except Exception as e:
        answer = f"Sorry, something went wrong: {e}"
    st.session_state.messages.append({"role": "assistant", "content": ensure_reply_language(answer, st.session_state.reply_lang)})

# ─────────────────────────────────────────────────────────────────────────────
# Lightweight local summary / QA fallbacks (used rarely)
def summarize_text_offline(text: str, max_sentences: int = 5) -> str:
    if not text:
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= max_sentences:
        return text.strip()

    stop = set("""
        the a an and or to of in on for from with at by as is are was were
        be been being it this that those these you your i we they them he she
        his her their our not no yes do does did have has had into about over
        under up down out more most much many very just can could should would
        there here then than so such if else when while where who whom whose
        which what
    """.split())

    words = re.findall(r"[a-zA-Z']+", text.lower())
    freqs = Counter(w for w in words if w not in stop and len(w) > 2)
    if not freqs:
        return " ".join(sentences[:max_sentences])

    scores = []
    for i, s in enumerate(sentences):
        toks = re.findall(r"[a-zA-Z']+", s.lower())
        score = sum(freqs.get(t, 0) for t in toks) / (len(toks) + 1e-6)
        scores.append((score, i, s))

    top = sorted(sorted(scores, key=lambda x: x[0], reverse=True)[:max_sentences],
                 key=lambda x: x[1])
    return " ".join(s for _, _, s in top)

def qa_text_offline(text: str, question: str, max_sentences: int = 6) -> str:
    if not text or not question:
        return ""
    qs = {q for q in re.findall(r"[a-zA-Z']+", question.lower()) if len(q) > 2}
    sentences = re.split(r'(?<=[.!?])\\s+', text.strip())
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
    _inject_css()
    _init_session()

    with st.sidebar:
        st.header("⚙️ Settings")
        # LangSmith toggle
        use_langsmith = st.checkbox("Enable LangSmith tracing", value=False, help="Requires LANGCHAIN_API_KEY")
        os.environ["LANGCHAIN_TRACING_V2"] = "true" if use_langsmith else "false"
        os.environ["LANGCHAIN_PROJECT"] = st.text_input("LangSmith project", value="kusina-bot")
        # Reply language
        st.subheader("Reply language")
        st.session_state.reply_lang = st.text_input("ISO / name (e.g., en, tl, es)", value=st.session_state.reply_lang)
        # Vectorstore
        if st.button("Rebuild vectorstore"):
            _load_assets.clear()
            st.cache_resource.clear()
            st.experimental_rerun()

        st.caption("gTTS is enabled (we’ll render MP3 for the last answer).")

    # Load data & VS once
    with st.spinner("Loading recipes & vector store…"):
        docs, vs = _load_assets(rebuild_vs=False)
        bind_vectorstore(vs)
    st.success(f"Recipes loaded: {len(docs)}")

    st.markdown('<div class="app-wrap">', unsafe_allow_html=True)
    st.title("🍳 Kusina Bot")
    st.caption("Kitchen buddy & nutrition coach — single chat window.")

    # ── SINGLE place for YouTube
    st.subheader("YouTube link")
    url = st.text_input("Paste a YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("Summarize video"):
            if not url.strip():
                st.warning("Paste a link first.")
            else:
                with st.spinner("Summarizing from transcript…"):
                    out = summarize_video.invoke({"url": url, "target_lang": st.session_state.reply_lang})
                    txt = out if isinstance(out, str) else str(out)
                    st.session_state.messages.append({"role": "assistant", "content": ensure_reply_language(txt, st.session_state.reply_lang)})

    with col_b:
        ask = st.text_input("Ask about the video", key="yt_qa")
        if st.button("Ask"):
            if not url.strip():
                st.warning("Paste a link first.")
            elif not ask.strip():
                st.warning("Type your question.")
            else:
                with st.spinner("Answering from transcript…"):
                    out = qa_video.invoke({"url": url, "question": ask, "target_lang": st.session_state.reply_lang})
                    txt = out if isinstance(out, str) else str(out)
                    st.session_state.messages.append({"role": "assistant", "content": ensure_reply_language(txt, st.session_state.reply_lang)})

    # ── Chat window
    st.markdown('<div class="phone-frame">', unsafe_allow_html=True)
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # (Optional) speak last assistant answer via gTTS
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        last = st.session_state.messages[-1]["content"]
        with st.expander("🔊 Speak to me, please. ", expanded=False):
            try:
                buf = io.BytesIO()
                gTTS(text=last, lang=("en" if st.session_state.reply_lang == "en" else "en")).write_to_fp(buf)
                buf.seek(0)
                st.audio(buf, format="audio/mp3")
            except Exception as e:
                st.info(f"TTS skipped: {e}")

    st.markdown('</div>', unsafe_allow_html=True)  # end phone-frame

    # ── Footer (chat input + optional mic)
    with st.container():
        # Chat input
        prompt = st.chat_input("Type a message (e.g., '30-minute dinners', 'calorie count of that', 'shopping list please')")
        if prompt:
            _handle_user_turn(prompt)

        # Optional mic (click-to-speak). For true hold-to-talk, enable streamlit-webrtc block below.
        if HAS_WEBRTC:
            with st.expander("🎙️ Voice input (click to record)", expanded=False):
                rtc_cfg = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
                ctx = webrtc_streamer(
                    key="speech",
                    mode=WebRtcMode.SENDONLY,
                    audio_receiver_size=1024,
                    rtc_configuration=rtc_cfg,
                    media_stream_constraints={"audio": True, "video": False},
                )
                # NOTE: You can pipe audio frames to a Whisper server or local recognizer here.
                st.caption("Tip: For press-and-hold UX, bind start/stop to mouse down/up events in a custom component.")

    st.markdown('</div>', unsafe_allow_html=True)  # end app-wrap

if __name__ == "__main__":
    main()

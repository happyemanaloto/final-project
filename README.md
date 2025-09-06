# 🍳 Happy Kusina Bot

Kusina Bot is a conversational cooking assistant built with **Streamlit**.  
It combines recipe retrieval, transcription, summarization, and multilingual chat into a single interactive web app.

---

## ✨ Features

- **Conversational agent** with memory, powered by LangChain.
- **Recipe vector store** for fast retrieval (Chroma by default).
- **Language support** with a dropdown (English, Tagalog, Spanish, French, German, Italian, Dutch, Portuguese, Japanese, Chinese).
- **Translation toggle** – translate answers automatically into your chosen language.
- **Speech I/O**
  - 🎙️ Mic input (via `audio-recorder-streamlit`) with transcription & summarization.
  - 🔈 Speaker toggle for auto-playback of assistant replies using gTTS.
- **Media uploads** – upload audio/video files, transcribe them, and get summaries.
- **YouTube integration** – ingest and summarize videos using `yt-dlp`.
- **RAG Debugger** – probe the vector store with custom queries and inspect top-k results.
- **LangSmith tracing** – optional logging of conversations for debugging.

---

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt

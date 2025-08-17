# bot/yt_transcriber.py
# Reliable YouTube transcription with:
# - API-first transcript (YouTubeTranscriptApi)
# - Whisper fallback (with download timeout)
# - 20-minute guard
# - Streamlit-friendly status messages
# - Caching to data/recipes/<video_id>.json

from __future__ import annotations
import json, os, re, tempfile, threading
from functools import lru_cache
from datetime import timedelta
from typing import Dict, Optional, Tuple

# 3rd party
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import yt_dlp

# Whisper (openai-whisper)
try:
    import whisper
except ImportError:
    whisper = None  # handled at runtime


# ---------- Paths & Helpers ----------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _cache_path(cache_dir: str, video_id: str) -> str:
    _ensure_dir(cache_dir)
    return os.path.join(cache_dir, f"{video_id}.json")

def _safe_video_id(url_or_id: str) -> Optional[str]:
    # Accept raw ID or any common YouTube URL
    # Matches 11-char video IDs
    m = re.search(
        r"(?:v=|youtu\.be/|shorts/|embed/)?([0-9A-Za-z_-]{11})", url_or_id
    )
    return m.group(1) if m else None


# ---------- Metadata (no download) ----------

def get_video_metadata(url: str, timeout_s: int = 15) -> Dict:
    """
    Uses yt_dlp to fetch metadata without downloading media.
    """
    result_container = {"data": None, "error": None}

    def _worker():
        try:
            ydl_opts = {
                "skip_download": True,
                "quiet": True,
                "nocheckcertificate": True,
                "noplaylist": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            result_container["data"] = {
                "id": info.get("id"),
                "title": info.get("title"),
                "duration": info.get("duration") or 0,  # seconds
                "webpage_url": info.get("webpage_url") or url,
                "uploader": info.get("uploader"),
            }
        except Exception as e:
            result_container["error"] = str(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        raise TimeoutError("Timed out reading video metadata")
    if result_container["error"]:
        raise RuntimeError(result_container["error"])
    return result_container["data"]


# ---------- API-first transcript ----------

PREFERRED_LANGS = ["en", "en-US", "en-GB", "tl", "fil", "es"]

def try_transcript_api(video_id: str) -> Optional[Dict]:
    try:
        # Find best available transcript
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        # Prefer our languages, otherwise allow translation to English if available
        for lang in PREFERRED_LANGS:
            if transcripts.find_manually_created_transcript([lang]):
                tr = transcripts.find_manually_created_transcript([lang])
                data = tr.fetch()
                return {
                    "source": "api",
                    "lang": tr.language_code,
                    "chunks": data,
                    "text": " ".join([c["text"] for c in data]).strip(),
                }
        for lang in PREFERRED_LANGS:
            if transcripts.find_generated_transcript([lang]):
                tr = transcripts.find_generated_transcript([lang])
                data = tr.fetch()
                return {
                    "source": "api",
                    "lang": tr.language_code,
                    "chunks": data,
                    "text": " ".join([c["text"] for c in data]).strip(),
                }

        # As a last resort, translate a transcript to English if possible
        for tr in transcripts:
            if tr.is_translatable:
                data = tr.translate("en").fetch()
                return {
                    "source": "api_translated",
                    "lang": "en",
                    "chunks": data,
                    "text": " ".join([c["text"] for c in data]).strip(),
                }
        return None
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception:
        return None


# ---------- Whisper fallback ----------

def _download_audio_with_timeout(url: str, out_dir: str, basename: str, timeout_s: int = 60) -> Optional[str]:
    """
    Download best audio track to M4A (or MP4) with yt_dlp.
    Returns the file path or None on timeout/error.
    """
    result = {"path": None, "error": None}

    def _worker():
        try:
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": os.path.join(out_dir, basename + ".%(ext)s"),
                "quiet": True,
                "noplaylist": True,
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "m4a",
                        "preferredquality": "192",
                    }
                ],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
            # Pick produced file
            base = os.path.join(out_dir, basename)
            for ext in (".m4a", ".mp3", ".webm", ".mp4"):
                cand = base + ext
                if os.path.exists(cand):
                    result["path"] = cand
                    return
            result["error"] = "Downloaded file not found."
        except Exception as e:
            result["error"] = str(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return None  # timeout
    if result["error"]:
        return None
    return result["path"]


@lru_cache(maxsize=1)
def _whisper_model(name: str = "base"):
    if whisper is None:
        raise ImportError("whisper package not installed. `pip install -U openai-whisper`")
    # Load once & cache
    return whisper.load_model(name)


def whisper_transcribe_file(audio_path: str, model_name: str = "base") -> Dict:
    """
    Accuracy-oriented transcription:
    - Try Faster-Whisper (CPU-friendly, robust) if installed.
    - Else fall back to openai-whisper with beam search + temperature fallback.
    """
    # 1) Try Faster-Whisper first (optional dependency)
    try:
        from faster_whisper import WhisperModel  # type: ignore
        print(f"[yt_transcriber] Using Faster-Whisper model: {model_name}")
        # CPU friendly; if you have CUDA/GPU change device and compute_type accordingly.
        fw = WhisperModel(model_name, device="cpu", compute_type="int8")
        segments, info = fw.transcribe(
            audio_path,
            vad_filter=True,       # helps ignore silence/noise
            beam_size=5,           # beam search reduces hallucinations
            language=None,         # auto-detect; FW handles it well
        )
        text = " ".join(s.text for s in segments).strip()
        lang = (info.language or "unknown")
        return {"source": "faster_whisper", "lang": lang, "chunks": [], "text": text}
    except Exception:
        pass  # fall back

    # 2) Fallback: openai-whisper
    model = _whisper_model(model_name)
    # decoding knobs: temperature fallback + beam search when temp hits 0.0
    params = dict(
        fp16=False,
        temperature=[0.0, 0.2, 0.4],
        beam_size=5,
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    out = model.transcribe(audio_path, **params)
    text = out.get("text", "").strip()
    return {
        "source": "whisper",
        "lang": out.get("language", "unknown"),
        "chunks": out.get("segments", []),
        "text": text,
    }



# ---------- Main entry (pure Python) ----------

def transcribe_youtube_best_effort(
    url_or_id: str,
    cache_dir: str = "data/recipes",
    max_minutes: int = 20,
    download_timeout_s: int = 60,
    whisper_model: str = "base",
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Returns (payload, error_message).
    payload = {
      'video_id','title','duration_sec','url',
      'transcript': {... as returned above ...}
    }
    On guard/timeout/error, returns (None, 'reason message').
    """
    vid = _safe_video_id(url_or_id)
    if not vid:
        return None, "Could not parse a valid YouTube video ID from the input."

    # Serve from cache if exists
    cache_path = _cache_path(cache_dir, vid)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            return cached, None
        except Exception:
            pass  # fallthrough if cache corrupted

    # Metadata (also used for duration guard)
    try:
        meta = get_video_metadata(url_or_id)
    except TimeoutError:
        return None, "Timeout getting video metadata. Please try again."
    except Exception as e:
        return None, f"Failed to read video metadata: {e}"

    duration = int(meta.get("duration") or 0)
    minutes = duration / 60.0
    if duration > 0 and minutes > max_minutes:
        return None, f"This video is too long for demo mode (limit: {max_minutes} minutes)."

    # API-first transcript
    tx = try_transcript_api(vid)
    if tx is None:
        # Whisper fallback (with audio download timeout)
        with tempfile.TemporaryDirectory() as td:
            audio = _download_audio_with_timeout(meta["webpage_url"], td, vid, timeout_s=download_timeout_s)
            if audio is None:
                return None, "This video is too long/heavy to transcribe right now. Please try a shorter video."
            try:
                tx = whisper_transcribe_file(audio, model_name=whisper_model)
            except Exception as e:
                return None, f"Whisper transcription failed: {e}"

    payload = {
        "video_id": vid,
        "title": meta.get("title"),
        "duration_sec": duration,
        "duration_hms": str(timedelta(seconds=duration)) if duration else None,
        "url": meta.get("webpage_url") or url_or_id,
        "uploader": meta.get("uploader"),
        "transcript": tx,
    }

    # Write cache
    try:
        _ensure_dir(cache_dir)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # non-fatal

    return payload, None


# ---------- Streamlit-friendly helper ----------

def transcribe_youtube_streamlit(
    st,
    url_or_id: str,
    cache_dir: str = "data/recipes",
    max_minutes: int = 20,
    download_timeout_s: int = 60,
    whisper_model: str = "base",
) -> Optional[Dict]:
    """
    Same as transcribe_youtube_best_effort, but with nice Streamlit messages.
    """
    with st.spinner("Reading video metadata…"):
        try:
            meta = get_video_metadata(url_or_id)
        except TimeoutError:
            st.error("Timeout getting video metadata. Please try again.")
            return None
        except Exception as e:
            st.error(f"Failed to read video metadata: {e}")
            return None

    duration = int(meta.get("duration") or 0)
    minutes = duration / 60.0
    if duration > 0 and minutes > max_minutes:
        st.warning(f"This video is too long for demo mode (limit: {max_minutes} minutes). Please try a shorter one.")
        return None

    # Serve cache fast
    vid = _safe_video_id(url_or_id)
    cache_path = _cache_path(cache_dir, vid)
    if os.path.exists(cache_path):
        with st.spinner("Loading cached transcript…"):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                st.success("Transcript ready (from cache)!")
                return cached
            except Exception:
                pass

    with st.spinner("Fetching transcript (API)…"):
        tx = try_transcript_api(vid)

    if tx is None:
        with st.spinner("No transcript available; downloading audio for Whisper…"):
            with tempfile.TemporaryDirectory() as td:
                audio = _download_audio_with_timeout(meta["webpage_url"], td, vid, timeout_s=download_timeout_s)
                if audio is None:
                    st.error("This video is too long/heavy to transcribe right now. Please try a shorter video.")
                    return None
                with st.spinner("Transcribing with Whisper…"):
                    try:
                        tx = whisper_transcribe_file(audio, model_name=whisper_model)
                    except Exception as e:
                        st.error(f"Whisper transcription failed: {e}")
                        return None

    payload = {
        "video_id": vid,
        "title": meta.get("title"),
        "duration_sec": duration,
        "duration_hms": str(timedelta(seconds=duration)) if duration else None,
        "url": meta.get("webpage_url"),
        "uploader": meta.get("uploader"),
        "transcript": tx,
    }

    # Cache & finish
    try:
        _ensure_dir(cache_dir)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    st.success("Transcript ready!")
    return payload

"""
YouTube → Transcript → Structured Recipe (JSON)

Features
- Input: channel /videos URL (e.g., https://www.youtube.com/@MarionsKitchen/videos)
         OR a file with one YouTube video URL per line
- Lists videos reliably with yt-dlp
- Transcript strategy:
    1) Try YouTubeTranscriptApi (fast, no download)
    2) Fallback to Whisper local transcription (downloads audio)
- Extracts structured recipe fields via LangChain + OpenAI (Pydantic schema)
- Outputs:
    data/recipes/<video_id>.json     # per-video record (url, title, transcript, recipe{...})
    data/recipes/recipes.jsonl       # one record per line for your chatbot

Usage
  # Install (inside your venv):
  pip install yt-dlp openai-whisper youtube-transcript-api langchain langchain-openai pydantic bs4 requests

  # Set your OpenAI key (PowerShell example):
  $env:OPENAI_API_KEY="sk-..."

  # From a channel (first 10 videos):
  python youtube_recipe_pipeline.py --channel https://www.youtube.com/@MarionsKitchen/videos --max 10

  # From a urls.txt:
  python youtube_recipe_pipeline.py --urls-file urls.txt --max 5

Notes
- Requires ffmpeg in PATH if Whisper fallback is used.
- Safe to re-run: existing per-video JSONs are skipped unless --no-skip is passed.

Invocation:
 python youtube_recipe_pipeline.py --channel https://www.youtube.com/@MarionsKitchen/videos --max 5       
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv, find_dotenv
import os

# Load .env starting from your current working directory and walking up
_ = load_dotenv(find_dotenv(filename=".env", usecwd=True))

# Optional: fail fast if missing
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found. Ensure it’s set in your .env")

import requests
from bs4 import BeautifulSoup  # not strictly needed now, but handy for debugging pages
from yt_dlp import YoutubeDL

# Optional imports: we handle absence gracefully
try:
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
    HAS_YT_TRANSCRIPT = True
except Exception:
    HAS_YT_TRANSCRIPT = False

try:
    import whisper
    HAS_WHISPER = True
except Exception:
    HAS_WHISPER = False

# LangChain / OpenAI for structured extraction
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# ------------------ Config / Paths ------------------
DATA_DIR = Path("data/recipes")
AUDIO_DIR = DATA_DIR / "audio"
DATA_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Reuse Whisper model between videos
_WHISPER_MODEL = None


# ------------------ Helpers ------------------
def safe_slug(text: str, max_len: int = 120) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", (text or "")).strip("_")[:max_len]


def ensure_videos_url(url: str) -> str:
    """Normalize a channel URL to its /videos page."""
    u = url.rstrip("/")
    if not re.search(r"/videos/?$", u, flags=re.IGNORECASE):
        u = u + "/videos"
    return u


def extract_video_id(url: str) -> Optional[str]:
    """Extract the video ID from any YouTube watch/embed/shorts URL."""
    try:
        # Common patterns
        m = re.search(r"(?:v=|\.be/|/shorts/|/embed/)([\w-]{11})", url)
        return m.group(1) if m else None
    except Exception:
        return None


# ------------------ Transcript (Strategy) ------------------
def get_transcript_via_api(video_id: str) -> Optional[str]:
    """Try YouTubeTranscriptApi (fast). Returns text or None if unavailable."""
    if not HAS_YT_TRANSCRIPT:
        return None
    try:
        segments = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        return " ".join(seg["text"] for seg in segments).strip()
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception:
        return None


def load_whisper(model_size: str = "base"):
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        if not HAS_WHISPER:
            raise RuntimeError(
                "Whisper is not installed. Install with `pip install openai-whisper` "
                "and ensure ffmpeg is available in PATH."
            )
        _WHISPER_MODEL = whisper.load_model(model_size)
    return _WHISPER_MODEL


def download_audio(url: str) -> dict:
    """Download best audio and convert to WAV with ffmpeg (via yt-dlp)."""
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(AUDIO_DIR / "%(id)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}],
        "quiet": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    video_id = info.get("id")
    wav_path = AUDIO_DIR / f"{video_id}.wav"
    return {
        "video_id": video_id,
        "video_title": info.get("title"),
        "channel": info.get("uploader"),
        "audio_path": str(wav_path),
        "webpage_url": info.get("webpage_url") or url,
    }


def transcribe_via_whisper(audio_path: str, model_size: str = "base") -> str:
    model = load_whisper(model_size)
    result = model.transcribe(audio_path, fp16=False)  # CPU-friendly
    return (result.get("text") or "").strip()


def get_transcript(url: str, prefer_api: bool = True, whisper_size: str = "base") -> tuple[str, dict]:
    """
    Returns (transcript_text, meta_info)
      - meta_info: may contain 'video_id', 'video_title', 'channel', 'webpage_url'
    Strategy: Try API first (if enabled) then Whisper fallback.
    """
    vid = extract_video_id(url)
    meta = {"video_id": vid, "video_title": None, "channel": None, "webpage_url": url}

    if prefer_api and vid:
        text = get_transcript_via_api(vid)
        if text:
            return text, meta

    # Fallback to whisper (downloads audio)
    dl = download_audio(url)
    meta.update(dl)
    text = transcribe_via_whisper(dl["audio_path"], model_size=whisper_size)
    return text, meta


# ------------------ Video listing ------------------
def fetch_video_urls(channel_videos_url: str) -> List[str]:
    """
    Robustly list video URLs using yt-dlp. Works with /@handle/videos, /channel/<id>/videos, etc.
    """
    url = ensure_videos_url(channel_videos_url)
    ydl_opts = {"quiet": True, "extract_flat": "in_playlist", "dump_single_json": True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    urls = []
    for e in (info or {}).get("entries", []) or []:
        u = e.get("url") or e.get("webpage_url")
        if u and u.startswith("http"):
            urls.append(u)
        elif e.get("id"):
            urls.append(f"https://www.youtube.com/watch?v={e['id']}")
    # de-dup while preserving order
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


# ------------------ LLM: Structured Recipe Extraction ------------------
class RecipeFields(BaseModel):
    title: str = Field(..., description="Clear recipe title")
    ingredients: List[str] = Field(default_factory=list, description="One ingredient per item")
    steps: List[str] = Field(default_factory=list, description="Numbered, concise steps")
    servings: Optional[str] = None
    cook_time_minutes: Optional[int] = None
    cuisine: Optional[str] = None
    tools: Optional[List[str]] = None


parser = PydanticOutputParser(pydantic_object=RecipeFields)

EXTRACT_PROMPT = PromptTemplate(
    input_variables=["transcript"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    template=(
        "You are a precise recipe extractor. Use ONLY the transcript to infer details.\n"
        "If the information isn't present, leave that field null or empty.\n\n"
        "{format_instructions}\n\n"
        "Transcript:\n{transcript}"
    ),
)


def extract_recipe(transcript: str, model_name: str = "gpt-4o-mini") -> RecipeFields:
    llm = ChatOpenAI(model=model_name, temperature=0)
    chain = EXTRACT_PROMPT | llm | parser
    return chain.invoke({"transcript": transcript})


# ------------------ Save / IO ------------------
def save_record(record: dict, out_dir: Path = DATA_DIR):
    video_id = record.get("video_id") or safe_slug(record.get("video_title", "video"))
    per_video = out_dir / f"{video_id}.json"
    with per_video.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    master = out_dir / "recipes.jsonl"
    with master.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ------------------ Orchestrate one video ------------------
def process_one(
    url: str,
    whisper_size: str = "base",
    llm_model: str = "gpt-4o-mini",
    prefer_api: bool = True,
    out_dir: Path = DATA_DIR,
    skip_existing: bool = True,
) -> dict:
    # Try to detect video_id early for skip logic
    vid_guess = extract_video_id(url)
    if skip_existing and vid_guess:
        per_video = out_dir / f"{vid_guess}.json"
        if per_video.exists():
            with per_video.open("r", encoding="utf-8") as f:
                return json.load(f)

    transcript, meta = get_transcript(url, prefer_api=prefer_api, whisper_size=whisper_size)
    if not transcript:
        raise RuntimeError("No transcript text could be obtained.")

    # Extract structured recipe
    recipe = extract_recipe(transcript, model_name=llm_model)

    video_id = meta.get("video_id") or extract_video_id(meta.get("webpage_url", "")) or safe_slug(meta.get("video_title"))
    record = {
        "url": meta.get("webpage_url") or url,
        "video_id": video_id,
        "video_title": meta.get("video_title"),
        "channel": meta.get("channel"),
        "transcript": transcript,
        "recipe": recipe.model_dump(),
    }
    save_record(record, out_dir=out_dir)
    return record


# ------------------ Batch runner ------------------
def run_pipeline(
    urls: List[str],
    max_videos: Optional[int] = None,
    whisper_size: str = "base",
    llm_model: str = "gpt-4o-mini",
    prefer_api: bool = True,
    out_dir: Path = DATA_DIR,
    skip_existing: bool = True,
) -> List[dict]:
    if max_videos:
        urls = urls[:max_videos]
    results = []
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {url}")
        try:
            rec = process_one(
                url,
                whisper_size=whisper_size,
                llm_model=llm_model,
                prefer_api=prefer_api,
                out_dir=out_dir,
                skip_existing=skip_existing,
            )
            results.append(rec)
        except Exception as e:
            print(f"  !! Failed: {e}")
    print(f"Done. Saved to {out_dir}/<video_id>.json and {out_dir}/recipes.jsonl")
    return results


# ------------------ CLI ------------------
def main():
    ap = argparse.ArgumentParser(description="YouTube → Transcript → Structured Recipe JSON")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--channel", type=str, help="YouTube channel URL (any form); will use its /videos page")
    src.add_argument("--urls-file", type=str, help="Text file with one YouTube video URL per line")

    ap.add_argument("--max", type=int, default=None, help="Process at most N videos")
    ap.add_argument("--whisper-size", type=str, default="base", help="Whisper model size: tiny, base, small, medium, large")
    ap.add_argument("--llm-model", type=str, default="gpt-4o-mini", help="OpenAI chat model for extraction")
    ap.add_argument("--prefer-api", action="store_true", help="Prefer YouTubeTranscriptApi first (default: True)")
    ap.add_argument("--no-prefer-api", dest="prefer_api", action="store_false", help="Skip API; force Whisper transcription")
    ap.add_argument("--out-dir", type=str, default=str(DATA_DIR), help="Output directory for JSON/JSONL")
    ap.add_argument("--no-skip", dest="skip_existing", action="store_false", help="Reprocess even if per-video JSON exists")
    ap.set_defaults(prefer_api=True, skip_existing=True)

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.urls_file:
        urls = [ln.strip() for ln in Path(args.urls_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        urls = fetch_video_urls(args.channel)

    run_pipeline(
        urls=urls,
        max_videos=args.max,
        whisper_size=args.whisper_size,
        llm_model=args.llm_model,
        prefer_api=args.prefer_api,
        out_dir=out_dir,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()

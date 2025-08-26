#!/usr/bin/env python3
"""
transcribe_videos.py
- Input: list of YouTube URLs and/or local media files
- Output:
  raw/{video_id}.transcript.json      -> full segments w/ timestamps
  chunks/{video_id}.jsonl             -> chunked text for RAG
  index/recipes_transcripts.jsonl     -> catalog for all processed assets

Notes
- Uses faster-whisper (CPU ok, set compute_type='int8' for speed)
- Light NLP: ingredient keyword pass + “step-like” sentence hints
- No external API calls here; embeddings can be run later as a separate step
"""

import os, json, re, uuid, subprocess, sys, math, time
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple, Optional

# ---------- CONFIG ----------
OUTPUT_DIR = Path("data_kusina")
RAW_DIR   = OUTPUT_DIR / "raw"
CHUNK_DIR = OUTPUT_DIR / "chunks"
INDEX     = OUTPUT_DIR / "index" / "recipes_transcripts.jsonl"

AUDIO_FMT = "mp3"  # yt-dlp audio format
WHISPER_MODEL = "small"  # "base" for very fast, "small" is a good balance
COMPUTE_TYPE = "int8"    # "int8" on CPU is fast; use "float16" for GPU

# chunking
TARGET_CHARS = 1200     # ~200-300 tokens
CHUNK_OVERLAP_CHARS = 200

# simple ingredient vocabulary (expand later)
INGR_HINTS = [
    "rice","garlic","onion","egg","tuna","tomato","soy sauce","vinegar","sugar","salt",
    "pepper","chicken","pork","beef","fish","oil","calamansi","lime","lemon","ginger",
    "milk","evaporated milk","coconut milk","butter","flour","yeast","scallion","spring onion",
    "moringa","malunggay","bell pepper","chili","sardines","ketchup","banana ketchup","oyster sauce",
]

STEP_HINT_WORDS = [
    "mix","stir","saute","sauté","boil","simmer","marinate","preheat","bake","airfry",
    "fry","pan-fry","grill","toast","combine","whisk","add","pour","slice","chop","mince",
    "season","knead","rest","serve","garnish"
]

# ---------- UTIL ----------
def sh(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def ensure_dirs():
    for d in [OUTPUT_DIR, RAW_DIR, CHUNK_DIR, INDEX.parent]:
        d.mkdir(parents=True, exist_ok=True)

def safe_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", s)[:150]

def guess_id_from_url(url: str) -> str:
    # crude; yt-dlp will also return id in metadata
    m = re.search(r"v=([A-Za-z0-9_\-]{6,})", url)
    return m.group(1) if m else uuid.uuid4().hex[:10]

def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")

def load_faster_whisper():
    from faster_whisper import WhisperModel
    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type=COMPUTE_TYPE)

def tokenize_len(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # fallback rough estimate
        return max(1, math.ceil(len(text) / 4))

def chunk_segments(segments: List[Dict[str, Any]],
                   target_chars: int = TARGET_CHARS,
                   overlap: int = CHUNK_OVERLAP_CHARS) -> List[Dict[str, Any]]:
    """
    Greedy chunk by character length, preserving timestamps.
    """
    chunks = []
    cur_text, cur_start, cur_end = [], None, None
    cur_chars = 0

    def flush():
        nonlocal cur_text, cur_start, cur_end, cur_chars
        if not cur_text:
            return
        text = " ".join(cur_text).strip()
        chunks.append({
            "text": text,
            "start": cur_start,
            "end": cur_end,
            "n_tokens": tokenize_len(text),
            "n_chars": len(text),
        })
        # create overlap basis
        if overlap > 0 and len(text) > overlap:
            keep = text[-overlap:]
            cur_text = [keep]
            cur_chars = len(keep)
            cur_start = max(cur_start, cur_end - 10.0) if cur_start is not None and cur_end is not None else None
        else:
            cur_text, cur_chars, cur_start = [], 0, None
        cur_end = None

    for seg in segments:
        seg_text = seg["text"].strip()
        if not seg_text:
            continue
        if cur_start is None:
            cur_start = seg["start"]
        cur_end = seg["end"]
        cur_text.append(seg_text)
        cur_chars += len(seg_text) + 1
        if cur_chars >= target_chars:
            flush()
    flush()
    return chunks

def extract_ingredient_hints(text: str) -> List[str]:
    text_l = text.lower()
    found = [ing for ing in INGR_HINTS if ing in text_l]
    # de-dup while preserving order
    seen = set()
    out = []
    for x in found:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def score_step_likeness(text: str) -> float:
    tl = text.lower()
    score = sum(1 for w in STEP_HINT_WORDS if re.search(rf"\b{re.escape(w)}\b", tl))
    # reward numbers → “Step 1”, times “10 minutes”
    score += len(re.findall(r"\b\d+\b", tl)) * 0.3
    score += len(re.findall(r"\b(min|mins|minutes|sec|seconds)\b", tl)) * 0.5
    return float(score)

# ---------- DOWNLOAD & METADATA ----------
def ytdlp_metadata(url: str) -> Dict[str, Any]:
    # Get JSON metadata without downloading the media
    out = sh(["yt-dlp", "-J", url]).stdout
    meta = json.loads(out)
    # If playlist, choose first entry
    if "entries" in meta and meta["entries"]:
        meta = meta["entries"][0]
    return {
        "id": meta.get("id") or guess_id_from_url(url),
        "title": meta.get("title"),
        "uploader": meta.get("uploader"),
        "channel": meta.get("channel"),
        "duration": meta.get("duration"),
        "webpage_url": meta.get("webpage_url") or url,
        "upload_date": meta.get("upload_date"),
        "description": meta.get("description"),
    }

def ytdlp_audio(url: str, out_dir: Path) -> Path:
    """
    Download audio-only track to out_dir and return the file path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tpl = str(out_dir / f"%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", AUDIO_FMT,
        "-o", tpl,
        url
    ]
    sh(cmd)
    # find the downloaded file (id.ext)
    meta = ytdlp_metadata(url)
    path = out_dir / f"{meta['id']}.{AUDIO_FMT}"
    if not path.exists():
        # fallback: find any new file
        files = sorted(out_dir.glob(f"*.{AUDIO_FMT}"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            raise FileNotFoundError("Audio not found after yt-dlp")
        path = files[0]
    return path

# ---------- TRANSCRIBE ----------
def transcribe_audio(audio_path: Path) -> Dict[str, Any]:
    model = load_faster_whisper()
    segments_iter, info = model.transcribe(
        str(audio_path),
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=1, best_of=1
    )
    segments = []
    for s in segments_iter:
        segments.append({
            "id": len(segments),
            "start": float(s.start),
            "end": float(s.end),
            "text": s.text.strip()
        })
    return {
        "language": info.language,
        "duration": sum(seg["end"] - seg["start"] for seg in segments) if segments else 0.0,
        "segments": segments
    }

# ---------- MAIN PIPE ----------
def process_source(src: str) -> Dict[str, Any]:
    ensure_dirs()
    started = time.time()

    if is_url(src):
        meta = ytdlp_metadata(src)
        vid = meta["id"] or guess_id_from_url(src)
        audio = ytdlp_audio(src, OUTPUT_DIR / "audio")
    else:
        # local file: fabricate a meta
        vid = safe_filename(Path(src).stem)[:24]
        meta = {
            "id": vid, "title": Path(src).name, "uploader": None, "channel": None,
            "duration": None, "webpage_url": None, "upload_date": None, "description": None
        }
        audio = Path(src)

    # transcribe
    tr = transcribe_audio(audio)

    # write raw transcript
    raw_path = RAW_DIR / f"{vid}.transcript.json"
    raw = {**meta, **tr}
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    # chunk
    chunks = chunk_segments(tr["segments"])
    # add light hints
    out_lines = []
    for i, ch in enumerate(chunks):
        hints_ing = extract_ingredient_hints(ch["text"])
        step_score = score_step_likeness(ch["text"])
        out = {
            "video_id": meta["id"],
            "title": meta["title"],
            "uploader": meta["uploader"],
            "channel": meta["channel"],
            "webpage_url": meta["webpage_url"],
            "upload_date": meta["upload_date"],
            "chunk_id": f"{meta['id']}_{i}",
            "start": ch["start"],
            "end": ch["end"],
            "text": ch["text"],
            "n_tokens_est": ch["n_tokens"],
            "ingredient_hints": hints_ing,
            "step_likeness": step_score,
            "source": "youtube" if is_url(src) else "local",
        }
        out_lines.append(out)

    # write chunks
    chunk_path = CHUNK_DIR / f"{vid}.jsonl"
    with chunk_path.open("w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    # append to global index
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX, "a", encoding="utf-8") as f:
        idx_entry = {
            "type": "video_transcript",
            "id": meta["id"],
            "title": meta["title"],
            "channel": meta["channel"],
            "webpage_url": meta["webpage_url"],
            "path_transcript": str(raw_path),
            "path_chunks": str(chunk_path),
            "n_chunks": len(out_lines),
            "lang": tr["language"],
            "created_at": int(time.time())
        }
        f.write(json.dumps(idx_entry, ensure_ascii=False) + "\n")

    elapsed = time.time() - started
    return {
        "id": meta["id"], "title": meta["title"],
        "chunks": len(out_lines), "language": tr["language"],
        "elapsed_sec": round(elapsed, 2),
        "paths": {"raw": str(raw_path), "chunks": str(chunk_path)}
    }

def main(args: List[str]):
    if not args:
        print("Usage: python transcribe_videos.py <youtube_url_or_file> [more ...]")
        sys.exit(1)
    ensure_dirs()
    results = []
    for src in args:
        try:
            print(f"Processing: {src}")
            res = process_source(src)
            print(f"  -> {res}")
            results.append(res)
        except Exception as e:
            print(f"[ERROR] {src}: {e}")
    print("\nDone. Summary:")
    for r in results:
        print(r)

if __name__ == "__main__":
    main(sys.argv[1:])

# bot/tools.py
import os, json, re, datetime as dt, tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from pydantic import BaseModel
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript
from yt_dlp import YoutubeDL
import tempfile
import trafilatura  # pip install trafilatura
# Local helpers
from .nlp import ensure_reply_language, localize_ingredients, llm_zero

# Import your full pipeline
try:
    from .transcribe_videos import process_source as _process_local_media
except Exception:
    _process_local_media = None

# ========= Session + VectorStore bindings =========
VS = None
_get_hits: Optional[Callable[[], list]] = None
_set_hits: Optional[Callable[[list], None]] = None
# Cache for estimate_nutrition calls to avoid recomputing identical requests
_nutrition_cache: Dict[tuple, str] = {}

def bind_vectorstore(vs):
    """Call once at startup to give tools access to your Chroma vector store."""
    global VS
    VS = vs

def bind_session_hooks(get_hits: Callable[[], list], set_hits: Callable[[list], None]):
    """Call once per session to connect tools to this session's memory."""
    global _get_hits, _set_hits
    _get_hits, _set_hits = get_hits, set_hits

def _session_get_hits() -> list:
    return _get_hits() if _get_hits else []

def _session_set_hits(hits: list) -> None:
    if _set_hits:
        _set_hits(hits)

def _upsert_transcript_into_vs(url: str, title: str, transcript: str):
    """Best-effort: add transcript as a retrievable doc so future queries see this video."""
    global VS
    if VS is None or not transcript:
        return
    # If it’s a YouTube URL, attach a thumbnail
    vid = _extract_video_id(url or "")
    meta = {
        "id": f"yt:{vid or url}",
        "title": title or "YouTube Recipe",
        "url": url,
        "source": "youtube" if ("youtube.com" in (url or "") or "youtu.be" in (url or "")) else "web",
        "image_url": f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None,
        "cuisine": None,
        "cook_time": None,
        "servings": None,
        # light text filter helpers (optional):
        "ingredients_text": "",
    }
    try:
        VS.add_texts(texts=[transcript], metadatas=[meta])
        VS.persist()
    except Exception as e:
        print("[vector] upsert transcript failed:", e)


def _as_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            j = json.loads(v)
            if isinstance(j, list):
                return j
        except Exception:
            return [s.strip() for s in re.split(r"[\n;,•·]", v) if s.strip()]
    return []

# ========= Media helpers (YouTube + Whisper fallback) =========
_YT_ID_RE = re.compile(r"(?:v=|\.be/|/shorts/|/embed/)([\w-]{11})", re.I)

# --- YouTube transcription helpers (robust) ---

def _extract_video_id(url: str) -> Optional[str]:
    m = re.search(r"(?:v=|\.be/|/shorts/|/embed/)([\w-]{11})", url)
    return m.group(1) if m else None

def _transcript_via_api(vid: str) -> Optional[str]:
    """Try official + auto-generated transcripts in multiple languages."""
    if not vid:
        return None
    try:
        # 1) try direct (fast path)
        segs = YouTubeTranscriptApi.get_transcript(
            vid,
            languages=["en", "tl", "es", "de", "fr", "ja"]
        )
        if segs:
            return " ".join(s["text"] for s in segs if s.get("text")).strip()
    except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript):
        pass
    except Exception:
        pass

    # 2) enumerate transcripts and pick whatever is available (including generated)
    try:
        listing = YouTubeTranscriptApi.list_transcripts(vid)
        # prefer English manual > English generated > anything manual > anything generated
        order = []
        try:
            order.append(listing.find_manually_created_transcript(["en"]))
        except Exception:
            pass
        try:
            order.append(listing.find_generated_transcript(["en"]))
        except Exception:
            pass
        for tr in listing.transcripts:  # fallback: any language
            order.append(tr)
        for tr in order:
            if not tr:
                continue
            try:
                segs = tr.fetch()
                txt = " ".join(s["text"] for s in segs if s.get("text")).strip()
                if txt:
                    return txt
            except Exception:
                continue
    except Exception:
        pass
    return None

# --- Whisper fallback (if enabled) ---
# minimal internal Whisper fallback (only used if pipeline disabled/missing)
_WHISPER = None
def _whisper_model():
    global _WHISPER
    if _WHISPER is None:
        import whisper
        _WHISPER = whisper.load_model(CHEF_WHISPER_MODEL)
    return _WHISPER

# def _download_audio(url: str) -> Path:
#     """Download best audio and convert to .wav via yt-dlp+ffmpeg."""
#     outdir = Path(tempfile.gettempdir()) / "kusina_audio"
#     outdir.mkdir(parents=True, exist_ok=True)
#     ydl_opts = {
#         "format": "bestaudio/best",
#         "outtmpl": str(outdir / "%(id)s.%(ext)s"),
#         "postprocessors": [
#             {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}
#         ],
#         "quiet": True,
#         "noprogress": True,
#     }
#     with YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(url, download=True)
#     return outdir / f"{info.get('id')}.wav"


def _download_audio(url: str) -> Path:
    tmp_root = Path(tempfile.gettempdir()) / "kusina_audio"
    tmp_root.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "quiet": True,
        "skip_download": False,
        "format": "bestaudio/best",
        "outtmpl": str(tmp_root / "%(id)s.%(ext)s"),
        "concurrent_fragment_downloads": 1,  # avoid multiple writers on Windows
        "nopart": True,                      # don’t create .part-Frag### files
        "windowsfilenames": True,            # Windows-safe renames
        "noprogress": True,
    }

    from yt_dlp import YoutubeDL
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        fname = ydl.prepare_filename(info)
    return Path(fname)


def _transcribe_whisper_local(url_or_path: str) -> str:
    """Download audio (if URL) and transcribe with Whisper. Returns '' on failure."""
    try:
        model_size = (os.getenv("CHEF_WHISPER_MODEL", "small") or "small").lower()
        import whisper
        model = whisper.load_model(model_size)

        # If it's a local file, use it; else download audio with yt-dlp
        p = Path(url_or_path)
        if p.exists():
            wav_path = str(p)
        else:
            outdir = Path(tempfile.gettempdir()) / "kusina_audio"
            outdir.mkdir(parents=True, exist_ok=True)
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": str(outdir / "%(id)s.%(ext)s"),
                "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}],
                "quiet": True,
            }
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url_or_path, download=True)
            wav_path = str(outdir / f"{info.get('id')}.wav")

        res = model.transcribe(wav_path, fp16=False)  # CPU-friendly
        return (res.get("text") or "").strip()
    except Exception as e:
        print("[transcribe] whisper failed:", e)
        return ""

def _get_transcript_external(url: str) -> Optional[str]:
    """
    Use your pipeline’s get_transcript(url, prefer_api=True/False, whisper_size='small').
    Returns transcript string or None.
    """
    if not _HAS_PIPELINE or _pipeline_get_transcript is None:
        return None
    try:
        # prefer_api=True lets your pipeline try YT API first, then Whisper
        text, meta = _pipeline_get_transcript(url, prefer_api=True, whisper_size=CHEF_WHISPER_MODEL)
        return (text or "").strip() if text else None
    except Exception as e:
        print(f"[transcribe] external pipeline failed: {e}")
        return None

import os

def transcribe_youtube_best_effort(url: str, max_videos: int = 1) -> str:
    """
    Master switch:
      - if CHEF_TRANSCRIBE_BACKEND=external → pipeline only
      - if =internal → internal only
      - if =auto (default) → pipeline first, fallback internal (API→Whisper)
    """
    backend = os.getenv("CHEF_TRANSCRIBE_BACKEND", "auto").lower()

    # external only
    if backend == "external":
        txt = _get_transcript_external_subprocess(url, max_videos=max_videos)
        return txt or ""

    # internal only
    if backend == "internal":
        vid = _extract_video_id(url)
        txt = _transcript_via_api(vid) if vid else None
        if txt:
            return txt
        return _transcribe_whisper_local(url)

    # auto (pipeline → API → Whisper)
    txt = _get_transcript_external_subprocess(url, max_videos=max_videos)
    if txt:
        return txt
    vid = _extract_video_id(url)
    txt = _transcript_via_api(vid) if vid else None
    if txt:
        return txt
    return _transcribe_whisper_local(url)

def _transcribe_any(url_or_path: str) -> str:
    """
    Fallback path: if YouTube transcript is absent, download audio and transcribe with Whisper.
    Requires: yt-dlp + ffmpeg + openai-whisper installed & ffmpeg on PATH.
    """
    # If it's a YouTube URL and we can extract vid, try API first (already done by caller normally)
    if "youtube.com" in url_or_path or "youtu.be" in url_or_path:
        vid = _extract_video_id(url_or_path)

    p = Path(url_or_path)
    if not p.exists():
        # download to wav
        try:
            p = _download_audio(url_or_path)
        except Exception as e:
            print(f"[transcribe] yt-dlp failed: {e}")
            return ""

    try:
        res = _whisper_model().transcribe(str(p), fp16=False)
        return (res.get("text") or "").strip()
    except Exception as e:
        print(f"[transcribe] whisper failed: {e}")
        return ""

def _get_transcript_external_subprocess(url: str, max_videos: int = 1) -> Optional[str]:
    """
    Runs: python youtube_recipe_pipeline.py --urls-file <tmp> --max N --prefer-api
    Then reads data/recipes/<video_id>.json for transcript.
    """
    try:
        import sys, json, subprocess, time, tempfile
        tmp = Path(tempfile.gettempdir()) / f"one_url_{int(time.time())}.txt"
        tmp.write_text(url + "\n", encoding="utf-8")

        script = "youtube_recipe_pipeline.py"
        cmd = [
            sys.executable,
            script,
            "--urls-file", str(tmp),
            "--max", str(max_videos),
            "--prefer-api",
        ]
        subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]), check=True)

        vid = _extract_video_id(url)
        if not vid:
            return None
        candidate = Path("data/recipes") / f"{vid}.json"
        if not candidate.exists():
            return None
        obj = json.loads(candidate.read_text(encoding="utf-8"))
        return (obj.get("transcript") or "").strip()
    except Exception as e:
        print("[transcribe] external subprocess failed:", e)
        return None

# ========= Vector Search =========
class VSearchArgs(BaseModel):
    query: str
    top_k: int = 3
    time_limit: Optional[int] = None
    cuisine: Optional[str] = None
    country: Optional[str] = None
    continent: Optional[str] = None
    must_include: Optional[List[str]] = None
    exclude_ingredients: Optional[List[str]] = None
    avoid_allergens: Optional[List[str]] = None
    display_lang: Optional[str] = None

@tool("vector_search", args_schema=VSearchArgs)
def vector_search(**kwargs) -> str:
    """Semantic recipe search against the Chroma vector store.
    Returns JSON: {"hits": [ {id,title,url,source,image_url,cuisine,cook_time,ingredients,ingredients_display,steps,content} ]}.
    """
    global VS
    if VS is None:
        return json.dumps({"hits": [], "note": "vector store not ready"})

    query = kwargs.get("query", "")
    top_k = int(kwargs.get("top_k", 3))
    display_lang = kwargs.get("display_lang")
    country = kwargs.get("country")
    continent = kwargs.get("continent")

    # docs_scores = VS.similarity_search_with_score(query, k=max(12, top_k * 4))
    docs_scores = VS.similarity_search_with_score(query, k=max(8, top_k * 2))

    ranked: List[tuple] = []
    for doc, dist in docs_scores:
        m = doc.metadata
        ings = _as_list(m.get("ingredients_json") or m.get("ingredients") or m.get("ingredients_text"))
        steps = _as_list(m.get("steps_json") or m.get("steps"))
        if len(ings) < 2 and len(steps) < 2:
            continue

        # Base semantic score (0..1)
        base = 1.0 - float(dist if dist is not None else 1.0)
        base = max(0.0, min(1.0, base))

        # Optional additional similarity (0..1) if you store it in metadata
        t_sim = float(m.get("text_similarity", 0.0))

        # Geographic boost
        boost = 0.0
        if country and (m.get("country") or "").lower() == str(country).lower():
            boost += 0.10
        elif continent and (m.get("continent") or "").lower() == str(continent).lower():
            boost += 0.05

        score = 0.7 * base + 0.3 * t_sim + boost
        ings_local = localize_ingredients(ings, display_lang)
        ranked.append((score, doc, ings, steps, ings_local))

    ranked.sort(key=lambda x: x[0], reverse=True)

    hits: List[Dict[str, Any]] = []
    for score, doc, ings, steps, ings_local in ranked[:top_k]:
        m = doc.metadata
        hits.append({
            "id": m.get("id"),
            "title": m.get("title"),
            "url": m.get("url"),
            "source": m.get("source"),
            "image_url": m.get("image_url"),
            "cuisine": m.get("cuisine"),
            "cook_time": m.get("cook_time"),
            "ingredients": ings,
            "ingredients_display": ings_local,
            "steps": steps[:4],
            "content": doc.page_content[:1000],
        })

    _session_set_hits(hits)
    return json.dumps({"hits": hits})

# ========= Keyword Search (fallback) =========
from rapidfuzz import process, fuzz

class KeywordIndex:
    """Simple fuzzy keyword fallback using RapidFuzz."""
    def __init__(self, docs):
        self.docs = docs
        self.corpus = [d.search_text for d in docs]

    def search(self, q, k=3):
        scored = process.extract((q or "").lower(), self.corpus, scorer=fuzz.token_set_ratio, score_cutoff=0)
        return [self.docs[idx] for _, _, idx in scored[:k]]

KIDX = None

def bind_keyword_index(kidx):
    """Optionally bind a keyword index built in your app layer."""
    global KIDX
    KIDX = kidx

class KSearchArgs(BaseModel):
    preferences_json: str
    top_k: int = 3

@tool("keyword_search", args_schema=KSearchArgs)
def keyword_search(preferences_json: str, top_k: int = 3) -> str:
    """Fuzzy keyword fallback search using RapidFuzz.
    Input is a preferences JSON; returns JSON {"hits":[...]}.
    """
    if not KIDX:
        return json.dumps({"hits": [], "note": "keyword index not ready"})
    try:
        prefs = json.loads(preferences_json)
    except Exception:
        prefs = {}
    q = prefs.get("free_text", "")
    hits = KIDX.search(q, k=top_k)
    out = [{"id": d.id, "title": d.title, "url": d.url, "image_url": d.image_url, "source": d.source} for d in hits]
    return json.dumps({"hits": out})

# ========= Transcription Tool =========
class TranscribeArgs(BaseModel):
    url_or_path: str

@tool("transcribe_media", args_schema=TranscribeArgs)
def transcribe_media(url_or_path: str) -> str:
    """Get a transcript for a YouTube URL (API first) or local/remote audio (Whisper if enabled).
    Returns JSON: {"transcript": "<text>"}.
    """
    if "youtube.com" in url_or_path or "youtu.be" in url_or_path:
        return json.dumps({"transcript": transcribe_youtube_best_effort(url_or_path) or ""})

    mode = (os.getenv("CHEF_TRANSCRIBE", "api_only") or "").lower()
    if mode != "api_only":
        return json.dumps({"transcript": _transcribe_whisper_local(url_or_path)})
    return json.dumps({"transcript": ""})
# ========= Local media (full pipeline) =========
class TranscribeLocalArgs(BaseModel):
    path: str  # local audio/video file path

@tool("transcribe_local_media", args_schema=TranscribeLocalArgs)
def transcribe_local_media(path: str) -> str:
    """
    Run the full faster-whisper pipeline on a local media file.
    Returns JSON: {"transcript_path": "<...raw/...json>", "chunks_path": "<...chunks/...jsonl>", "meta": {...}}
    """
    if _process_local_media is None:
        return json.dumps({"error": "pipeline not importable (bot/transcribe_videos.py missing)"} )
    p = Path(path)
    if not p.exists():
        return json.dumps({"error": f"file not found: {path}"})
    try:
        res = _process_local_media(str(p))
        # res: {"id","title","chunks","language","elapsed_sec","paths":{"raw","chunks"}}
        return json.dumps({"transcript_path": res["paths"]["raw"], "chunks_path": res["paths"]["chunks"], "meta": res})
    except Exception as e:
        return json.dumps({"error": str(e)})

class SummarizeTranscriptArgs(BaseModel):
    transcript_path: str
    target_lang: str = "en"

@tool("summarize_transcript_file", args_schema=SummarizeTranscriptArgs)
def summarize_transcript_file(transcript_path: str, target_lang: str = "en") -> str:
    """
    Summarize a saved transcript JSON ({segments:[{start,end,text},...]}) into:
    Title; ≤6 key ingredients; 3–5 steps; 1 tip (localized). Also caches a minimal session hit.
    """
    p = Path(transcript_path)
    if not p.exists():
        return ensure_reply_language("Transcript file not found.", target_lang or "en")

    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        segs = obj.get("segments") or []
        title = obj.get("title") or (obj.get("webpage_url") or obj.get("id") or "Local media")
        raw_txt = " ".join([s.get("text","").strip() for s in segs if s.get("text")]).strip()
    except Exception:
        return ensure_reply_language("Could not read transcript JSON.", target_lang or "en")

    if not raw_txt:
        return ensure_reply_language("Transcript is empty.", target_lang or "en")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "From transcript, write: Title; ≤6 key ingredients; 3–5 steps; 1 tip."),
        ("human", "Target language: {lang}\n\nTranscript:\n{tx}")
    ])
    # out = (prompt | ChatOpenAI(model=os.getenv("CHEF_BOT_MODEL", "gpt-4o-mini"), temperature=0)).invoke(
    # out = (prompt | ChatOpenAI(model=os.getenv("CHEF_BOT_MODEL", "gpt‑3.5‑turbo"), temperature=0)).invoke(
    #     {"lang": target_lang or "en", "tx": raw_txt[:12000]}
    # )
    out = (prompt | llm_zero(temperature=0)).invoke(
        {"lang": target_lang or "en", "tx": raw_txt[:5000]}
    )

    summary = out.content.strip()

    # Try to collect a few ingredients from the summary for follow-ups
    ings = []
    collect = False
    for ln in summary.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.lower().startswith("key ingredients"):
            collect = True
            continue
        if collect:
            if s[:1] in "-•":
                ings.append(s.lstrip("-• ").strip())
            else:
                break
    ings = ings[:10]

    # Cache a minimal hit so calories/shopping list flows work
    _session_set_hits([{
        "id": f"local:{p.stem}",
        "title": (summary.splitlines()[0] or title).strip(),
        "url": "",  # local file has no URL
        "source": "local",
        "ingredients": ings,
        "ingredients_display": ings,
        "steps": [],
    }])

    # Also upsert raw transcript into VS for future semantic search
    try:
        _upsert_transcript_into_vs(url="", title=title, transcript=raw_txt)
    except Exception:
        pass

    return ensure_reply_language(summary, target_lang or "en")

# ========= Nutrition =========
class NutriArgs(BaseModel):
    ingredients: List[str]
    servings: Optional[int] = 2
    locale: Optional[str] = "EU"

@tool("estimate_nutrition", args_schema=NutriArgs)
def estimate_nutrition(ingredients: List[str], servings: Optional[int] = 2, locale: Optional[str] = "EU") -> str:
    """Estimate calories/macros per serving from an ingredient list.
    Returns compact JSON (as a string).
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Estimate nutrition per serving from ingredient list. Return compact JSON numbers."),
        ("human", "Ingredients:\n{ings}\nServings: {serv}\nLocale: {loc}")
    ])
    # Build a cache key (convert ingredients list to a tuple so it’s hashable)
    cache_key = (tuple(ingredients), servings or 2, locale or "EU")
    if cache_key in _nutrition_cache:
        return _nutrition_cache[cache_key]

    # out = (prompt | ChatOpenAI(model=os.getenv("CHEF_BOT_MODEL", "gpt-4o-mini"), temperature=0)).invoke(
    # out = (prompt | ChatOpenAI(model=os.getenv("CHEF_BOT_MODEL", "gpt‑3.5‑turbo"), temperature=0)).invoke(
    #     {"ings": "\n".join(ingredients), "serv": servings, "loc": locale}
    # )
    # return out.content
    out = (prompt | llm_zero(temperature=0)).invoke(
        {"ings": "\n".join(ingredients), "serv": servings, "loc": locale}
    )
    result = out.content
    _nutrition_cache[cache_key] = result  # save to cache
    return result

# ========= Shopping List =========
class ShopArgs(BaseModel):
    recipes: Optional[List[Dict[str, Any]]] = None
    servings_multiplier: Optional[float] = 1.0
    target_lang: Optional[str] = None

@tool("make_shopping_list", args_schema=ShopArgs)
def make_shopping_list(recipes: Optional[List[Dict[str, Any]]] = None,
                       servings_multiplier: Optional[float] = 1.0,
                       target_lang: Optional[str] = None) -> str:
    """Aggregate ingredients across recipes into a grouped shopping list (text).
    If recipes is None, uses the session's last hits.
    """
    if not recipes:
        recipes = _session_get_hits()
    slim = []
    for r in recipes:
        ings = r.get("ingredients") or r.get("ingredients_display") or []
        if isinstance(ings, str):
            ings = [ings]
        slim.append({"title": r.get("title"), "ingredients": ings})

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Aggregate a concise shopping list grouped by aisle; merge duplicates; add brief cheaper substitutions."),
        ("human", "Target language: {lang}\nServings x: {mult}\n\nRecipes:\n{payload}")
    ])
    # out = (prompt | ChatOpenAI(model=os.getenv("CHEF_BOT_MODEL", "gpt-4o-mini"), temperature=0)).invoke(
    # out = (prompt | ChatOpenAI(model=os.getenv("CHEF_BOT_MODEL", "gpt‑3.5‑turbo"), temperature=0)).invoke(
    #     {"lang": target_lang or "en", "mult": servings_multiplier or 1.0, "payload": json.dumps({"recipes": slim}, ensure_ascii=False)}
    # )
    # return out.content.strip()
    out = (prompt | llm_zero(temperature=0)).invoke(
        {"lang": target_lang or "en", "mult": servings_multiplier or 1.0,
        "payload": json.dumps({"recipes": slim}, ensure_ascii=False)}
    )
    return out.content.strip()

# ========= Cookbook / Feedback / Translate =========
class CookbookArgs(BaseModel):
    recipe_ids: List[str]
    language: Optional[str] = "en"
    title: Optional[str] = "My Personal Cookbook"

@tool("create_cookbook", args_schema=CookbookArgs)
def create_cookbook(recipe_ids: List[str], language: Optional[str] = "en", title: Optional[str] = "My Personal Cookbook") -> str:
    """Create a Markdown cookbook from selected recipe IDs (looked up in session hits).
    Returns JSON: {"path":"<output.md>"}.
    """
    out_dir = Path(__file__).resolve().parents[1] / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    idmap = {r.get("id"): r for r in _session_get_hits()}
    lines = [f"# {title}\n"]
    for rid in recipe_ids:
        r = idmap.get(rid)
        if not r:
            continue
        lines.append(f"## {r.get('title','Recipe')}\n")
        lines.append(f"[Link]({r.get('url','')})  \nSource: {r.get('source','')}\n")
        ings = r.get("ingredients") or []
        steps = r.get("steps") or []
        if ings:
            lines += ["### Ingredients", *[f"- {i}" for i in ings]]
        if steps:
            lines += ["### Steps", *[f"{i}. {s}" for i, s in enumerate(steps, 1)]]
        lines.append("")

    out_path = out_dir / f"cookbook_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return json.dumps({"path": str(out_path)})

class FeedbackArgs(BaseModel):
    recipe_id: str
    feedback_text: str
    user_lang: Optional[str] = None

@tool("add_feedback", args_schema=FeedbackArgs)
def add_feedback(recipe_id: str, feedback_text: str, user_lang: Optional[str] = None) -> str:
    """Append user feedback to data/feedback.jsonl (UTC timestamped)."""
    fb = Path(__file__).resolve().parents[1] / "data" / "feedback.jsonl"
    fb.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "recipe_id": recipe_id,
        "feedback": feedback_text,
        "lang": user_lang or "unknown",
    }
    with fb.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return json.dumps({"status": "ok"})

class TranslateArgs(BaseModel):
    text: str
    target_lang: str

@tool("translate_text", args_schema=TranslateArgs)
def translate_text(text: str, target_lang: str) -> str:
    """Translate arbitrary text to the target language; preserve bullets/formatting.
    Returns ONLY the translated text.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Translate; preserve bullets and formatting; return only translated text."),
        ("human", "Target language: {lang}\n\nText:\n{txt}")
    ])
    # out = (prompt | ChatOpenAI(model=os.getenv("CHEF_BOT_MODEL", "gpt-4o-mini"), temperature=0)).invoke(
    # out = (prompt | ChatOpenAI(model=os.getenv("CHEF_BOT_MODEL", "gpt‑3.5‑turbo"), temperature=0)).invoke(
    #     {"lang": target_lang, "txt": text}
    # )
    out = (prompt | llm_zero(temperature=0)).invoke(
        {"lang": target_lang, "txt": text}
    )

    return out.content.strip()

# ========= YouTube: summarize & QA (now cache a hit) =========
class SummarizeArgs(BaseModel):
    url: str
    target_lang: Optional[str] = None

@tool("summarize_video", args_schema=SummarizeArgs)
def summarize_video(url: str, target_lang: Optional[str] = None) -> str:
    """Summarize a YouTube recipe transcript into: Title; ≤6 key ingredients; 3–5 steps; 1 tip (in target_lang).
    Also caches a minimal hit in session memory.
    """
    tx = transcribe_youtube_best_effort(url)

    # Always attempt Whisper fallback if API failed AND CHEF_TRANSCRIBE != 'off'
    # (Your .env already has CHEF_TRANSCRIBE=whisper, so this runs.)
    # mode = (os.getenv("CHEF_TRANSCRIBE", "api_only") or "").lower()
    # if not tx and mode != "off":
    #     tx = _transcribe_any(url)

    if not tx:
        return ensure_reply_language("Transcript unavailable right now.", target_lang or "en")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "From transcript, write: Title; ≤6 key ingredients; 3–5 steps; 1 tip."),
        ("human", "Target language: {lang}\n\nTranscript:\n{tx}")
    ])
    # out = (prompt | ChatOpenAI(model=os.getenv("CHEF_BOT_MODEL", "gpt-4o-mini"), temperature=0)).invoke(
    # out = (prompt | ChatOpenAI(model=os.getenv("CHEF_BOT_MODEL", "gpt‑3.5‑turbo"), temperature=0)).invoke(
    #     {"lang": target_lang or "en", "tx": tx}
    # )
    out = (prompt | llm_zero(temperature=0)).invoke(
        {"lang": target_lang or "en", "tx": tx[:5000]}
    )

    # Parse a reasonable title
    title_line = (out.content.splitlines()[0] or "From YouTube").strip()

    # 1) Upsert the transcript so the video is retrievable later
    _upsert_transcript_into_vs(url, title_line, tx)

    # 2) Cache a proper session hit so calories/shopping list can work
    vid = _extract_video_id(url or "")
    thumb = f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

    # Optional: try to extract the “Key ingredients” bullets from the summary:
    ings = []
    collect = False
    for ln in out.content.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.lower().startswith("key ingredients"):
            collect = True
            continue
        if collect:
            if s[:1] in "-•":
                ings.append(s.lstrip("-• ").strip())
            else:
                # stop once bullets end
                break

    _session_set_hits([{
        "id": f"yt:{vid or url}",
        "title": title_line,
        "url": url,
        "source": "youtube",
        "image_url": thumb,
        "ingredients": ings,             # helps calorie tool
        "ingredients_display": ings,     # localized later by UI if needed
        "steps": [],
    }])


    return ensure_reply_language(out.content.strip(), target_lang or "en")

class QAVideoArgs(BaseModel):
    url: str
    question: str
    target_lang: Optional[str] = None

@tool("qa_video", args_schema=QAVideoArgs)
def qa_video(url: str, question: str, target_lang: Optional[str] = None) -> str:
    """Answer a specific question strictly from a YouTube transcript (in target_lang).
    If not stated, reply 'Not stated in the video.' Also caches a minimal hit.
    """
    tx = transcribe_youtube_best_effort(url)

    if not tx:
        return ensure_reply_language("I can’t read that video’s transcript right now.", target_lang or "en")
    
    # ✅ vector store insert
    _upsert_transcript_into_vs(url, "QA video", tx)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer strictly from transcript; if missing say 'Not stated in the video.' Be concise."),
        ("human", "Target language: {lang}\nQuestion: {q}\n\nTranscript:\n{tx}")
    ])
    # out = (prompt | ChatOpenAI(model=os.getenv("CHEF_BOT_MODEL", "gpt-4o-mini"), temperature=0)).invoke(
    # out = (prompt | ChatOpenAI(model=os.getenv("CHEF_BOT_MODEL", "gpt‑3.5‑turbo"), temperature=0)).invoke(
    #     {"lang": target_lang or "en", "q": question, "tx": tx}
    # )
    out = (prompt | llm_zero(temperature=0)).invoke(
        {"lang": target_lang or "en", "q": question, "tx": tx[:5000]}
    )

    _session_set_hits([{
        "id": f"yt:{url}",
        "title": "From YouTube",
        "url": url,
        "source": "youtube",
        "ingredients": [], "steps": []
    }])

    return ensure_reply_language(out.content.strip(), target_lang or "en")

# ========= Unified link ingestor =========
class IngestLinkArgs(BaseModel):
    url: str
    target_lang: Optional[str] = None

@tool("ingest_link", args_schema=IngestLinkArgs)
def ingest_link(url: str, target_lang: Optional[str] = None) -> str:
    """Read a recipe link (YouTube or article), produce a mini-recipe card, and cache it in session hits.
    Returns the formatted card text (in target_lang).
    """
    # YouTube path
    if "youtube.com" in url or "youtu.be" in url:
        card = summarize_video.invoke({"url": url, "target_lang": target_lang or "en"})
        _session_set_hits([{
            "id": f"yt:{url}",
            "title": "From YouTube",
            "url": url,
            "source": "youtube",
            "ingredients": [], "steps": []
        }])
        return card if isinstance(card, str) else str(card)

    # Article path
    try:
        downloaded = trafilatura.fetch_url(url)  # keep compatible (no timeout kw)
        text = trafilatura.extract(downloaded) or ""
    except Exception:
        text = ""

    if not text:
        return ensure_reply_language(
            "I couldn't read that page, but I can still suggest a solid version. Tell me the dish name. 🙂",
            target_lang or "en"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract a concise recipe from the article. Return:\nTitle\n\nKey ingredients (≤8 bullets)\n\nQuick steps (3–6 bullets)\n\nTip (1 line)."),
        ("human", "Target language: {lang}\n\nArticle:\n{body}")
    ])
    # out = (prompt | ChatOpenAI(model=os.getenv("CHEF_BOT_MODEL", "gpt-4o-mini"), temperature=0)).invoke(
    # out = (prompt | ChatOpenAI(model=os.getenv("CHEF_BOT_MODEL", "gpt‑3.5‑turbo"), temperature=0)).invoke(
    #     {"lang": target_lang or "en", "body": text[:8000]}
    # )
    out = (prompt | llm_zero(temperature=0)).invoke(
        {"lang": target_lang or "en", "body": text[:5000]}
    )
    card_text = out.content.strip()

    # try to grab ingredients for follow-ups
    ings = [ln.strip("-• ").strip() for ln in card_text.splitlines() if ln.strip().startswith(("-", "•"))][:12]
    _session_set_hits([{
        "id": f"web:{url}",
        "title": (card_text.splitlines()[0] or "Recipe").strip(),
        "url": url,
        "source": "web",
        "ingredients": ings,
        "steps": []
    }])

    return ensure_reply_language(card_text, target_lang or "en")

# ========= Deterministic calories from link =========
class CaloriesFromUrlArgs(BaseModel):
    url: str
    servings: Optional[int] = 1
    locale: Optional[str] = "EU"
    target_lang: Optional[str] = None

@tool("calories_from_url", args_schema=CaloriesFromUrlArgs)
def calories_from_url(url: str, servings: Optional[int] = 1, locale: Optional[str] = "EU",
                      target_lang: Optional[str] = None) -> str:
    """Estimate calories/macros per serving for ONE recipe from a YouTube or article link.
    Extract a single ingredient list from the transcript/article; cache it as the current session hit; then call estimate_nutrition.
    Returns a compact answer in target_lang.
    """
    # 1) get raw text from URL
    txt = ""
    if "youtube.com" in url or "youtu.be" in url:
        vid = _extract_video_id(url)
        txt = _transcript_via_api(vid) if vid else ""
        if (not txt) and os.getenv("CHEF_TRANSCRIBE", "api_only").lower() != "api_only":
            try:
                txt = _transcribe_any(url)
            except Exception:
                txt = ""
    else:
        try:
            downloaded = trafilatura.fetch_url(url)
            txt = trafilatura.extract(downloaded) or ""
        except Exception:
            txt = ""

    if not txt:
        return ensure_reply_language("Transcript or article not readable right now.", target_lang or "en")

    # 2) extract ONE ingredient list for a single cooked dish
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "From the text, extract ONE concrete cooked dish and its core ingredients only. "
         "Return JSON with keys: title, ingredients (list of 6–14 items). No steps, no chatter."),
        ("human", "Text:\n{body}")
    ])
    # out = (prompt | ChatOpenAI(model=os.getenv("CHEF_BOT_MODEL", "gpt-4o-mini"), temperature=0)).invoke(
    # out = (prompt | ChatOpenAI(model=os.getenv("CHEF_BOT_MODEL", "gpt‑3.5‑turbo"), temperature=0)).invoke(
    #     {"body": txt[:8000]}
    # )
    out = (prompt | llm_zero(temperature=0)).invoke(
        {"body": txt[:5000]}
    )
    title = "Recipe"
    ings: List[str] = []
    try:
        data = json.loads(out.content)
        title = (data.get("title") or "Recipe").strip()
        ings = [s.strip() for s in (data.get("ingredients") or []) if s and s.strip()]
    except Exception:
        pass

    if not ings:
        # crude fallback: pick bullet-looking lines
        ings = [ln.strip("-• ").strip() for ln in txt.splitlines() if ln.strip().startswith(("-", "•"))][:12]
    if not ings:
        return ensure_reply_language("I couldn’t extract ingredients reliably from that link.", target_lang or "en")

    # 3) cache as session hit so follow-ups work
    _session_set_hits([{
        "id": f"link:{url}",
        "title": title,
        "url": url,
        "source": "youtube" if "youtu" in url else "web",
        "ingredients": ings,
        "steps": []
    }])

    # 4) estimate nutrition
    try:
        est = estimate_nutrition.invoke({"ingredients": ings, "servings": servings or 1, "locale": locale})
        est_txt = est if isinstance(est, str) else str(est)
    except Exception:
        return ensure_reply_language("Couldn’t estimate calories right now.", target_lang or "en")

    reply = f"{title} — per serving:\n{est_txt}"
    return ensure_reply_language(reply, target_lang or "en")

r"""
kusina-bot.py — Rev0: Kitchen Chatbot (Agents + Tools + Embeddings)

data layout:

YouTube (from youtube_recipe_pipeline.py)
- JSONL: C:\Users\happy\Documents\ironhack\kusina-bot\final-project\src\backend\scrapers\data\recipes\recipes.jsonl
- Per-video JSON files in the same folder

Wikibooks (from cookbook_toc_scraper.py)
- JSONL: C:\Users\happy\Documents\ironhack\kusina-bot\final-project\src\backend\scrapers\data\open_wikibooks_toc\recipes.jsonl
- Per-recipe JSON files in the same folder

The bot:
- Loads both JSONL and per-file JSON (deduplicated) for YT + Wikibooks
- Builds/loads a FAISS vector store (OpenAI embeddings) for semantic search
- Exposes tools (vector_search, keyword_search, transcribe_media, add_feedback, create_cookbook)
- Multilingual: translates to English for search, replies in user's language
- Media-aware: paste a YouTube/audio/video URL to transcribe first

Setup (once in env):
    pip install -U python-dotenv langchain langchain-openai langchain-community \
                   faiss-cpu pydantic rapidfuzz langdetect \
                   youtube-transcript-api yt-dlp openai-whisper

    .env (project root):
    OPENAI_API_KEY=sk-...

Run (defaults already point to paths; ):
    python kusina-bot.py --rebuild-vs    # first time to build embeddings
    python kusina-bot.py                 # later runs reuse the index

"""
from __future__ import annotations
import os, json, re, argparse, datetime as dt
from pathlib import Path
from typing import List, Dict, Any, Optional

# ---------- env ----------
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv(filename=".env", usecwd=True))

# ---------- llm / langchain ----------
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import AgentType, initialize_agent
from langchain.tools import tool
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
# ---------- nlp utils ----------
from rapidfuzz import process, fuzz
# from langdetect import detect as lang_detect
from langdetect import DetectorFactory, detect_langs
DetectorFactory.seed = 0

# ---------- media transcription ----------
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from yt_dlp import YoutubeDL
# import whisper  # requires ffmpeg installed/available
import re
from typing import Optional
# =========================
# Default paths (YOUR folders)
# =========================
DEFAULT_YT_DIR   = r"C:\Users\happy\Documents\ironhack\kusina-bot\final-project\src\backend\scrapers\data\recipes"
DEFAULT_YT_JSONL = r"C:\Users\happy\Documents\ironhack\kusina-bot\final-project\src\backend\scrapers\data\recipes\recipes.jsonl"

DEFAULT_WB_DIR   = r"C:\Users\happy\Documents\ironhack\kusina-bot\final-project\src\backend\scrapers\data\open_wikibooks_toc"
DEFAULT_WB_JSONL = r"C:\Users\happy\Documents\ironhack\kusina-bot\final-project\src\backend\scrapers\data\open_wikibooks_toc\recipes.jsonl"

DEFAULT_VS_DIR   = r"C:\Users\happy\Documents\ironhack\kusina-bot\final-project\src\backend\scrapers\data\vs"

FEEDBACK_PATH = Path(r"C:\Users\happy\Documents\ironhack\kusina-bot\final-project\src\backend\scrapers\data\feedback.jsonl")
EXPORTS_DIR   = Path(r"C:\Users\happy\Documents\ironhack\kusina-bot\final-project\exports")

# Models (override via env as an option)
LLM_MODEL  = os.getenv("CHEF_BOT_MODEL", "gpt-4o-mini")
EMB_MODEL  = os.getenv("CHEF_EMBED_MODEL", "text-embedding-3-small")

# FORCE_REPLY_LANG = os.getenv("CHEF_FORCE_REPLY_LANG")  or "en"
FORCE_REPLY_LANG = os.getenv("CHEF_FORCE_REPLY_LANG")  

# Session default (English unless changed later)
SESSION_REPLY_LANG = os.getenv("CHEF_DEFAULT_LANG", "en")

# Build a generous alias map
LANG_ALIASES = {}
def _add(code, *names):
    for n in (code, *names):
        LANG_ALIASES[n.lower()] = code

_add("en","english")
_add("tl","tagalog","fil","filipino")
_add("ko","korean")
_add("es","spanish","español")
_add("fr","french","français")
_add("de","german","deutsch")
_add("nl","dutch")
_add("it","italian","italiano")
_add("pt","portuguese","português","pt-br","brazilian portuguese")
_add("zh","chinese","zh-cn","simplified chinese","zh-tw","traditional chinese")
_add("ja","japanese","nihongo")
_add("vi","vietnamese","tiếng việt")
_add("id","indonesian","bahasa indonesia")
_add("th","thai")
_add("hi","hindi")
_add("ar","arabic","عربي")
_add("ru","russian","русский")
_add("tr","turkish","türkçe")
_add("el","greek","ελληνικά")
_add("he","hebrew","עברית")
_add("pl","polish","polski")
_add("sv","swedish","svenska")
_add("no","norwegian","norsk")
_add("da","danish","dansk")
_add("fi","finnish","suomi")
_add("cs","czech","čeština")
_add("hu","hungarian","magyar")
_add("ro","romanian","română")
_add("bg","bulgarian","български")
_add("uk","ukrainian","українська")
_add("ms","malay","bahasa melayu")
_add("ta","tamil")
_add("bn","bengali","বাংলা")
_add("ur","urdu","اردو")
_add("fa","farsi","persian","فارسی")
_add("sw","swahili","kiswahili")


def parse_language_switch(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    t = re.sub(r"[^\w\s\-]", "", t)  # strip punctuation

    # A) Explicit commands: "/lang ko", "switch to korean", "reply in pt-br", etc.
    m = re.search(
        r"(?:^|[\s:/])(?:/lang|lang(?:uage)?|switch|reply|answer|speak|use|set|respond)\s*"
        r"(?:to|in|:)?\s*([a-z][a-z0-9\- ]+)\s*$",
        t
    )
    if m:
        key = re.sub(r"\s+", " ", m.group(1).strip())
        return LANG_ALIASES.get(key)

    # B) Bare full language name ONLY if the whole message is just that name
    if t in LANG_ALIASES and len(t) > 2:
        return LANG_ALIASES[t]

    # C) Bare 2-letter or hyphen code ONLY if the whole message is just the code
    if (re.fullmatch(r"[a-z]{2}", t) or re.fullmatch(r"[a-z]{2}-[a-z]{2}", t)) and t in LANG_ALIASES:
        return LANG_ALIASES[t]

    # Otherwise, don't switch (prevents "ako" triggering 'ko')
    return None

# =========================
# Unified recipe model
# =========================
class RecipeDoc(BaseModel):
    id: str
    title: str
    url: str
    source: str  # "youtube" | "wikibooks"
    image_url: Optional[str] = None
    ingredients: List[str] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list)
    cuisine: Optional[str] = None
    cook_time_minutes: Optional[int] = None
    extras: Dict[str, Any] = Field(default_factory=dict)

    @property
    def search_text(self) -> str:
        # compact text for keyword/fuzzy
        return " ".join([
            self.title or "",
            self.cuisine or "",
            " ".join(self.ingredients or []),
        ]).lower()

# =========================
# Loaders (YouTube + Wikibooks)
# - Read both JSONL and per-file JSON
# - Deduplicate by id
# =========================
def _yt_obj_to_doc(obj: Dict[str, Any]) -> Optional[RecipeDoc]:
    rec = obj.get("recipe", {}) or {}
    vid = obj.get("video_id") or obj.get("id") or None
    if not vid and "url" in obj:
        m = re.search(r"(?:v=|\.be/|/shorts/|/embed/)([\w-]{11})", obj["url"])
        vid = m.group(1) if m else None
    url = obj.get("url") or (f"https://www.youtube.com/watch?v={vid}" if vid else "")
    title = rec.get("title") or obj.get("video_title") or "Untitled"
    img = f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None
    return RecipeDoc(
        id=f"yt:{vid or title}",
        title=title,
        url=url,
        source="youtube",
        image_url=img,
        ingredients=rec.get("ingredients") or [],
        steps=rec.get("steps") or [],
        cuisine=rec.get("cuisine"),
        cook_time_minutes=rec.get("cook_time_minutes"),
        extras={"channel": obj.get("channel")}
    )

def _wb_obj_to_doc(obj: Dict[str, Any]) -> Optional[RecipeDoc]:
    rec = obj.get("recipe", {}) or {}
    title = rec.get("title") or obj.get("title") or "Untitled"
    url = obj.get("source_url") or obj.get("url") or ""
    return RecipeDoc(
        id=f"wb:{title}",
        title=title,
        url=url,
        source="wikibooks",
        image_url=None,  # keep legal + simple
        ingredients=rec.get("ingredients") or [],
        steps=rec.get("steps") or [],
        cuisine=rec.get("cuisine"),
        cook_time_minutes=rec.get("cook_time_minutes"),
        extras={"license": obj.get("license"), "attribution": obj.get("attribution")}
    )

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    if path and path.exists():
        for ln in path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
    return out

def load_youtube(yt_dir: Path, yt_jsonl: Optional[Path]) -> List[RecipeDoc]:
    docs, seen = [], set()
    # 1) JSONL
    for obj in _load_jsonl(yt_jsonl) if yt_jsonl else []:
        d = _yt_obj_to_doc(obj)
        if d and d.id not in seen:
            docs.append(d); seen.add(d.id)
    # 2) Per-file JSONs (skip the JSONL file itself)
    if yt_dir and yt_dir.exists():
        for p in yt_dir.glob("*.json"):
            if p.name.lower().endswith(".jsonl"):
                continue
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            d = _yt_obj_to_doc(obj)
            if d and d.id not in seen:
                docs.append(d); seen.add(d.id)
    return docs

def load_wikibooks(wb_dir: Path, wb_jsonl: Optional[Path]) -> List[RecipeDoc]:
    docs, seen = [], set()
    # 1) JSONL
    for obj in _load_jsonl(wb_jsonl) if wb_jsonl else []:
        d = _wb_obj_to_doc(obj)
        if d and d.id not in seen:
            docs.append(d); seen.add(d.id)
    # 2) Per-file JSONs
    if wb_dir and wb_dir.exists():
        for p in wb_dir.glob("*.json"):
            if p.name.lower().endswith(".jsonl"):
                continue
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            d = _wb_obj_to_doc(obj)
            if d and d.id not in seen:
                docs.append(d); seen.add(d.id)
    return docs

# =========================
# Keyword/Fuzzy (fallback)
# =========================
class KeywordIndex:
    def __init__(self, docs: List[RecipeDoc]):
        self.docs = docs
        self.corpus = [d.search_text for d in docs]

    def search(self, query: str, top_k: int = 3, time_limit: Optional[int] = None) -> List[RecipeDoc]:
        if not self.docs:
            return []
        scored = process.extract((query or "").lower(), self.corpus, scorer=fuzz.token_set_ratio, score_cutoff=0)
        ranked = []
        for _, score, idx in scored:
            d = self.docs[idx]
            penalty = 0
            if time_limit and d.cook_time_minutes and d.cook_time_minutes > time_limit:
                overflow = d.cook_time_minutes - time_limit
                penalty = min(30, overflow // 5)
            ranked.append((score - penalty, d))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in ranked[:top_k]]

# =========================
# Embeddings + FAISS
# =========================
def _doc_to_embed_text(d: RecipeDoc) -> str:
    return "\n".join([
        d.title or "",
        f"Cuisine: {d.cuisine}" if d.cuisine else "",
        "Ingredients:\n" + "\n".join(d.ingredients or []),
        "Steps:\n" + "\n".join(d.steps or []),
    ]).strip()

# def build_or_load_vectorstore(docs: List[RecipeDoc], persist_dir: Path, rebuild: bool=False):
#     embed = OpenAIEmbeddings(model=EMB_MODEL)
#     persist_dir.mkdir(parents=True, exist_ok=True)
#     if (persist_dir / "index.faiss").exists() and (persist_dir / "index.pkl").exists() and not rebuild:
#         return FAISS.load_local(str(persist_dir), embed, allow_dangerous_deserialization=True)
#     texts, metas = [], []
#     for d in docs:
#         texts.append(_doc_to_embed_text(d))
#         metas.append({
#             "id": d.id, "title": d.title, "url": d.url, "source": d.source,
#             "image_url": d.image_url, "cuisine": d.cuisine, "cook_time": d.cook_time_minutes
#         })
#     vs = FAISS.from_texts(texts=texts, embedding=embed, metadatas=metas)
#     vs.save_local(str(persist_dir))
#     return vs

def build_or_load_vectorstore(docs, persist_dir: Path, rebuild: bool=False):
    embed = OpenAIEmbeddings(model=EMB_MODEL)
    persist_dir.mkdir(parents=True, exist_ok=True)
    texts, metas = [], []
    for d in docs:
        texts.append("\n".join([
            d.title or "",
            f"Cuisine: {d.cuisine}" if d.cuisine else "",
            "Ingredients:\n" + "\n".join(d.ingredients or []),
            "Steps:\n" + "\n".join(d.steps or []),
        ]).strip())
        metas.append({
            "id": d.id, "title": d.title, "url": d.url, "source": d.source,
            "image_url": d.image_url, "cuisine": d.cuisine, "cook_time": d.cook_time_minutes
        })
    # rebuild or first run
    if rebuild or not (persist_dir / "chroma.sqlite").exists():
        vs = Chroma.from_texts(texts=texts, embedding=embed, metadatas=metas,
                               persist_directory=str(persist_dir))
    else:
        vs = Chroma(persist_directory=str(persist_dir), embedding_function=embed)
    return vs

# =========================
# LLM helpers (translate + prefs)
# =========================
def llm_zero():
    return ChatOpenAI(model=LLM_MODEL, temperature=0)

TRANS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Translate to English. Return translation if it is not English."),
    ("human", "{text}")
])

def translate_to_english(text: str) -> str:
    if not text:
        return text
    try:
        if lang_detect(text) == "en":
            return text
    except Exception:
        pass
    out = (TRANS_PROMPT | llm_zero()).invoke({"text": text})
    return out.content.strip()

class Prefs(BaseModel):
    language: Optional[str] = None
    cuisine: Optional[str] = None
    part_of_meal: Optional[str] = None
    part_of_day: Optional[str] = None
    heavy_or_light: Optional[str] = None
    time_minutes: Optional[int] = None
    difficulty: Optional[str] = None
    budget: Optional[str] = None
    available_ingredients: Optional[str] = None
    servings: Optional[int] = None
    allergens: Optional[List[str]] = None
    goals: Optional[List[str]] = None
    free_text: Optional[str] = None

PREFS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Extract user cooking preferences as JSON for fields: "
     "language,cuisine,part_of_meal,part_of_day,heavy_or_light,time_minutes,"
     "difficulty,budget,available_ingredients,servings,allergens,goals,free_text. "
     "If not present, set null. Respond with JSON only."),
    ("human", "{text}")
])

def extract_prefs(text_en: str) -> Prefs:
    out = (PREFS_PROMPT | llm_zero()).invoke({"text": text_en})
    try:
        return Prefs(**json.loads(out.content))
    except Exception:
        return Prefs(free_text=text_en)

# =========================
# Transcription
# =========================
def _extract_video_id(url: str) -> Optional[str]:
    m = re.search(r"(?:v=|\.be/|/shorts/|/embed/)([\w-]{11})", url)
    return m.group(1) if m else None

def _transcript_via_api(vid: str) -> Optional[str]:
    try:
        segs = YouTubeTranscriptApi.get_transcript(vid, languages=["en"])
        return " ".join(s["text"] for s in segs).strip()
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception:
        return None

_WHISPER = None
def _whisper_model(size: str = "base"):
    global _WHISPER
    if _WHISPER is None:
        import whisper  # lazy import
        _WHISPER = whisper.load_model(size)
    return _WHISPER

def _download_audio(url: str) -> Path:
    outdir = Path(DEFAULT_YT_DIR).parent / "tmp_audio"
    outdir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(outdir / "%(id)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}],
        "quiet": True
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return outdir / f"{info.get('id')}.wav"

def transcribe_media(url_or_path: str) -> str:
    if "youtube.com" in url_or_path or "youtu.be" in url_or_path:
        vid = _extract_video_id(url_or_path)
        if vid:
            t = _transcript_via_api(vid)
            if t:
                return t
    p = Path(url_or_path)
    wav = p if p.exists() else _download_audio(url_or_path)
    res = _whisper_model().transcribe(str(wav), fp16=False)
    return (res.get("text") or "").strip()

# =========================
# Globals (wired in main)
# =========================
DOCS: List[RecipeDoc] = []
KIDX: Optional[KeywordIndex] = None
VS = None

# =========================
# Tools (for the Agent)
# =========================
class VSearchArgs(BaseModel):
    query: str
    top_k: int = 3
    time_limit: Optional[int] = None

@tool("vector_search", args_schema=VSearchArgs)
def tool_vector_search(query: str, top_k: int = 3, time_limit: Optional[int] = None) -> str:
    """Semantic search over title+ingredients+steps+cuisine (FAISS/OpenAI embeddings)."""
    global VS
    if VS is None:
        return json.dumps({"hits": [], "note": "vector store not ready"})
    docs_scores = VS.similarity_search_with_score(query, k=max(6, top_k * 3))
    ranked = []
    for doc, score in docs_scores:
        m = doc.metadata
        penalty = 0.0
        ct = m.get("cook_time")
        if time_limit and isinstance(ct, int) and ct > time_limit:
            overflow = ct - time_limit
            penalty = min(5.0, overflow / 20.0)
        ranked.append((score + penalty, m))
    ranked.sort(key=lambda x: x[0])
    hits = [{
        "id": m["id"], "title": m["title"], "url": m["url"],
        "image_url": m.get("image_url"), "source": m["source"]
    } for _, m in ranked[:top_k]]
    return json.dumps({"hits": hits})

class KSearchArgs(BaseModel):
    preferences_json: str
    top_k: int = 3

@tool("keyword_search", args_schema=KSearchArgs)
def tool_keyword_search(preferences_json: str, top_k: int = 3) -> str:
    """Fuzzy keyword fallback using RapidFuzz."""
    global KIDX
    if not KIDX:
        return json.dumps({"hits": [], "note": "keyword index not ready"})
    try:
        prefs = json.loads(preferences_json)
    except Exception:
        prefs = {}
    tokens = []
    for k in ["cuisine","part_of_meal","part_of_day","heavy_or_light","difficulty",
              "budget","available_ingredients","free_text"]:
        v = prefs.get(k)
        if isinstance(v, list): tokens += v
        elif isinstance(v, str) and v: tokens.append(v)
    if prefs.get("servings"): tokens.append(f"servings:{prefs['servings']}")
    query = " ".join(tokens)
    time_limit = prefs.get("time_minutes")
    hits = KIDX.search(query=query, top_k=top_k, time_limit=time_limit)
    out = [{
        "id": d.id, "title": d.title, "url": d.url,
        "image_url": d.image_url, "source": d.source
    } for d in hits]
    return json.dumps({"hits": out})

class TranscribeArgs(BaseModel):
    url_or_path: str

@tool("transcribe_media", args_schema=TranscribeArgs)
def tool_transcribe_media(url_or_path: str) -> str:
    """Transcribe a YouTube/audio/video URL or path. Returns {'transcript': ...}"""
    text = transcribe_media(url_or_path)
    return json.dumps({"transcript": text})

class FeedbackArgs(BaseModel):
    recipe_id: str
    feedback_text: str
    user_lang: Optional[str] = None

@tool("add_feedback", args_schema=FeedbackArgs)
def tool_add_feedback(recipe_id: str, feedback_text: str, user_lang: Optional[str] = None) -> str:
    """Append user feedback to feedback.jsonl (UTC timestamped)."""
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "recipe_id": recipe_id,
        "feedback": feedback_text,
        "lang": user_lang or "unknown",
    }
    with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return json.dumps({"status": "ok"})

class CookbookArgs(BaseModel):
    recipe_ids: List[str]
    language: Optional[str] = "en"
    title: Optional[str] = "My Personal Cookbook"

@tool("create_cookbook", args_schema=CookbookArgs)
def tool_create_cookbook(recipe_ids: List[str], language: Optional[str] = "en", title: Optional[str] = "My Personal Cookbook") -> str:
    """Build a markdown cookbook from selected recipes; returns {'path': ...}."""
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    idmap = {d.id: d for d in DOCS}
    lines = [f"# {title}\n"]
    for rid in recipe_ids:
        d = idmap.get(rid)
        if not d:
            continue
        lines.append(f"## {d.title}\n")
        lines.append(f"[Link]({d.url})  \nSource: {d.source}\n")
        if d.ingredients:
            lines.append("### Ingredients")
            lines += [f"- {ing}" for ing in d.ingredients]
        if d.steps:
            lines.append("### Steps")
            lines += [f"{i}. {step}" for i, step in enumerate(d.steps, 1)]
        lines.append("")
    out_path = EXPORTS_DIR / f"cookbook_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return json.dumps({"path": str(out_path)})

# =========================
# Agent system prompt
# =========================
SYSTEM = SystemMessage(content=(
"You are a chef and nutritionist chatbot who assists people worldwide, adapting to cravings, time, skill, "
"price sensitivity, health goals, and budget.\n"
"Reply in the language specified by 'reply_language' inside USER_REQUEST_JSON. "
"If it is not provided, mirror the user's message language. Do not switch languages on your own.\n"
"1) If the user provided audio/video, call transcribe_media first.\n"
"2) Translate the request to English for search, but reply in 'reply_language'.\n"
"3) Extract preferences (language, cuisine, part_of_meal, part_of_day, heavy_or_light, time_minutes, "
"difficulty, budget, available_ingredients, servings, allergens, goals, free_text).\n"
"4) Use vector_search to get top 3 matches; if sparse, still search and prefer healthy, practical options. "
"Fallback to keyword_search if needed.\n"
"5) Present results with links and images (YouTube thumbnails if available). Ask if they want alternatives.\n"
"6) Provide tips (efficiency, health, prices). Offer step-by-step sourcing/help.\n"
"7) Ask for feedback; if provided, call add_feedback.\n"
"8) If asked, call create_cookbook with selected recipes.\n"
"Keep answers concise, friendly, and actionable."
))


def build_agent():
    tools = [tool_vector_search, tool_keyword_search, tool_transcribe_media, tool_add_feedback, tool_create_cookbook]
    return initialize_agent(
        tools=tools,
        llm=ChatOpenAI(model=LLM_MODEL, temperature=0),
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=False,
        handle_parsing_errors=True,
        agent_kwargs={"system_message": SYSTEM},
    )

# =========================
# Chat orchestration
# =========================
URL_RE = re.compile(r"(https?://\S+)", re.I)

def detect_language(text: str) -> str:
    try:
        cands = detect_langs(text)  # e.g. [en:0.55, id:0.44]
        ascii_ok = text.isascii()
        short = len(text) < 40
        # Prefer English for short ASCII messages if it's a candidate at all
        if ascii_ok and short:
            for c in cands:
                if c.lang == "en" and c.prob >= 0.15:
                    return "en"
        # Otherwise, pick the highest-probability candidate
        return cands[0].lang if cands else "en"
    except Exception:
        return "en"


def maybe_media_url(text: str) -> Optional[str]:
    m = URL_RE.search(text or "")
    if not m:
        return None
    u = m.group(1)
    if any(x in u for x in ["youtube.com","youtu.be",".mp3",".wav",".m4a",".mp4",".mov",".webm"]):
        return u
    return None

from typing import Optional  # (you already have this at the top)

def chat_once(agent, user_text: str, session_reply_lang: Optional[str] = None) -> str:
    # 1) If there's media, transcribe to enrich SEARCH (not language choice)
    media = maybe_media_url(user_text)
    transcript = None
    if media:
        try:
            transcript = transcribe_media(media)
        except Exception:
            transcript = None

    # 2) Choose reply language (env override -> session -> detection)
    user_lang_guess = detect_language(user_text)
    if (not user_lang_guess or user_lang_guess == "unknown") and len(user_text) < 20 and user_text.isascii():
        user_lang_guess = "en"
    reply_lang = (FORCE_REPLY_LANG or session_reply_lang or user_lang_guess or "en")

    # 3) Translate FOR SEARCH (ok to include transcript)
    text_for_search = (transcript + "\n\n" + user_text) if transcript else user_text
    text_en = translate_to_english(text_for_search)

    # 4) Extract prefs; keep tools informed of reply language
    prefs = extract_prefs(text_en)
    prefs.language = reply_lang

    # 5) Agent call
    directive = {
        "translated_text_en": text_en,
        "preferences": prefs.model_dump(),
        "media_transcribed": bool(transcript),
        "reply_language": reply_lang,
    }
    result = agent.invoke({"input": f"USER_REQUEST_JSON: {json.dumps(directive, ensure_ascii=False)}"})
    return result.get("output", str(result))


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(description="Kusina Bot — Chef & Nutritionist (Agents + Tools + Embeddings)")
    ap.add_argument("--yt-dir",   type=str, default=DEFAULT_YT_DIR,   help="Folder with per-video JSON files")
    ap.add_argument("--yt-jsonl", type=str, default=DEFAULT_YT_JSONL, help="YouTube recipes.jsonl")
    ap.add_argument("--wb-dir",   type=str, default=DEFAULT_WB_DIR,   help="Folder with per-recipe JSON files")
    ap.add_argument("--wb-jsonl", type=str, default=DEFAULT_WB_JSONL, help="Wikibooks recipes.jsonl")
    ap.add_argument("--vs-dir",   type=str, default=DEFAULT_VS_DIR,   help="Folder to persist FAISS index")
    ap.add_argument("--rebuild-vs", action="store_true", help="Force rebuild of FAISS index")
    ap.add_argument("--force-reply-lang", type=str, default=None, help="Force assistant reply language (e.g., en, nl, es).")
    args = ap.parse_args()
    
    global FORCE_REPLY_LANG
    if args.force_reply_lang:
        FORCE_REPLY_LANG = args.force_reply_lang

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing. Put it in your .env")

    yt_dir   = Path(args.yt_dir)
    yt_jsonl = Path(args.yt_jsonl)
    wb_dir   = Path(args.wb_dir)
    wb_jsonl = Path(args.wb_jsonl)
    vs_dir   = Path(args.vs_dir)

    # Load both sources (JSONL + JSON files), deduped
    yt_docs = load_youtube(yt_dir, yt_jsonl if yt_jsonl.exists() else None)
    wb_docs = load_wikibooks(wb_dir, wb_jsonl if wb_jsonl.exists() else None)

    global DOCS, KIDX, VS
    DOCS = yt_docs + wb_docs
    print(f"Loaded {len(DOCS)} recipes ({len(yt_docs)} YouTube, {len(wb_docs)} Wikibooks).")
    if not DOCS:
        print("Warning: no recipes found. Check paths or run your pipelines first.")

    # Keyword index + FAISS vector index
    KIDX = KeywordIndex(DOCS)
    VS = build_or_load_vectorstore(DOCS, persist_dir=vs_dir, rebuild=args.rebuild_vs)
    print("Vector store ready.")

    agent = build_agent()
    global SESSION_REPLY_LANG
    print("\nKusina Bot ready. I am your kitchen assistant. How can I help you? Ctrl+C to exit.\n")
    while True:
        try:
            user = input("You: ").strip()
            if not user:
                continue

            # Language switch?
            maybe_lang = parse_language_switch(user)
            if maybe_lang:
                SESSION_REPLY_LANG = maybe_lang
                pretty = "Tagalog" if maybe_lang == "tl" else maybe_lang
                print(f"\nAssistant:\nOkay! I’ll reply in {pretty} from now on.\n")
                continue  # if the user only sent the switch command

            # 2) Ephemeral auto-switch for THIS TURN if confident + long message
            ephemeral_lang = SESSION_REPLY_LANG
            det = detect_language(user)
            if det in LANG_ALIASES.values() and det != SESSION_REPLY_LANG and len(user) >= 60:
                ephemeral_lang = det
                # (Optional) let them know how to lock it
                # print(f"[note] Detected {det}; replying in that language for this turn. Use '/lang {det}' to switch permanently.")

            answer = chat_once(agent, user, session_reply_lang=ephemeral_lang)
            print("\nAssistant:\n" + answer + "\n")
        except KeyboardInterrupt:
            print("\nBye!")
            break

if __name__ == "__main__":
    main()

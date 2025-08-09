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

CHEF_TEMP = float(os.getenv("CHEF_TEMP", "0.5"))

def llm_zero():
    return ChatOpenAI(model=LLM_MODEL, temperature=CHEF_TEMP)

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
# Philippines languages
_add("pam", "kapampangan", "pampangan", "pampango", "capampangan", "pampangueño", "pampangueo")
_add("ilo", "ilocano", "ilokano", "iloko")
_add("ceb", "cebuano", "bisaya", "binisaya", "sugbuanon", "visayan")
# Arabic dialects
_add("ary", "moroccan arabic", "moroccan", "darija", "derija", "ddarija", "العربية المغربية", "الدارجة")
_add("arz", "egyptian arabic", "egyptian", "masri", "masry", "العربية المصرية", "لهجة مصرية")

# def parse_language_switch(text: str) -> Optional[str]:
#     t = (text or "").strip().lower()
#     t = re.sub(r"[^\w\s\-]", "", t)  # strip punctuation

#     # A) Explicit commands: "/lang ko", "switch to korean", "reply in pt-br", etc.
#     m = re.search(
#         r"(?:^|[\s:/])(?:/lang|lang(?:uage)?|switch|reply|answer|speak|use|set|respond)\s*"
#         r"(?:to|in|:)?\s*([a-z][a-z0-9\- ]+)\s*$",
#         t
#     )
#     if m:
#         key = re.sub(r"\s+", " ", m.group(1).strip())
#         return LANG_ALIASES.get(key)

#     # B) Bare full language name ONLY if the whole message is just that name
#     if t in LANG_ALIASES and len(t) > 2:
#         return LANG_ALIASES[t]

#     # C) Bare 2-letter or hyphen code ONLY if the whole message is just the code
#     if (re.fullmatch(r"[a-z]{2}", t) or re.fullmatch(r"[a-z]{2}-[a-z]{2}", t)) and t in LANG_ALIASES:
#         return LANG_ALIASES[t]

#     # Otherwise, don't switch (prevents "ako" triggering 'ko')
#     return None
def parse_language_switch(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    t = re.sub(r"[^\w\s\-]", "", t)  # strip punctuation

    # A) Explicit commands you already have...
    m = re.search(
        r"(?:^|[\s:/])(?:/lang|lang(?:uage)?|switch|reply|answer|speak|use|set|respond)\s*"
        r"(?:to|in|:)?\s*([a-z][a-z0-9\- ]+)\s*$",
        t
    )
    if m:
        key = re.sub(r"\s+", " ", m.group(1).strip())
        return LANG_ALIASES.get(key)

    # NEW: catch "<language> please" or "in <language> please"
    m2 = re.search(r"(?:^|[\s])translate\s+.+?\s(?:to|into|in)\s+([a-z][a-z0-9\- ]+)\s*$", t)
    # m2 = re.search(r"^(?:in\s+)?([a-z][a-z0-9\- ]+)\s+(?:please|pls)\s*$", t)
    if m2:
        key = re.sub(r"\s+", " ", m2.group(1).strip())
        return LANG_ALIASES.get(key)

    # "<language> please" or "in <language> please"
    m3 = re.search(r"^(?:in\s+)?([a-z][a-z0-9\- ]+)\s+(?:please|pls)\s*$", t)
    if m3:
        key = re.sub(r"\s+", " ", m3.group(1).strip())
        return LANG_ALIASES.get(key)

    # add this after your other regex checks
    m4 = re.search(r"^\s*(?:in|en)\s+([a-z][a-z0-9\- ]+)\s*$", t)
    if m4:
        key = re.sub(r"\s+", " ", m4.group(1).strip())
        return LANG_ALIASES.get(key)

    # B) Bare full language name ONLY if whole message is just the name
    if t in LANG_ALIASES and len(t) > 2:
        return LANG_ALIASES[t]

    # C) Bare 2-letter or hyphen code ONLY if the whole message is just the code
    if (re.fullmatch(r"[a-z]{2}", t) or re.fullmatch(r"[a-z]{2}-[a-z]{2}", t)) and t in LANG_ALIASES:
        return LANG_ALIASES[t]

    return None

# =========================
# Unified recipe model
# =========================
class RecipeDoc(BaseModel):
    id: str
    title: str
    url: str
    source: str  # "youtube" | "wikibooks" | …
    image_url: Optional[str] = None
    ingredients: List[str] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list)
    cuisine: Optional[str] = None
    cook_time_minutes: Optional[int] = None
    servings: Optional[int] = None
    dietary_tags: List[str] = Field(default_factory=list)  # e.g., ["vegan", "gluten-free"]
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
        # rich page content for retrieval/summaries
        texts.append("\n".join([
            d.title or "",
            f"Cuisine: {d.cuisine}" if d.cuisine else "",
            "Ingredients:\n" + "\n".join(d.ingredients or []),
            "Steps:\n" + "\n".join(d.steps or []),
        ]).strip())

        metas.append({
            "id": d.id,
            "title": d.title,
            "url": d.url,
            "source": d.source,
            "image_url": d.image_url,
            "cuisine": d.cuisine,
            "cook_time": d.cook_time_minutes,
            "servings": d.servings,
            # store lists as JSON strings (OK for Chroma)
            "ingredients_json": json.dumps(d.ingredients or [], ensure_ascii=False),
            "steps_json": json.dumps(d.steps or [], ensure_ascii=False),
            "dietary_tags_json": json.dumps(d.dietary_tags or [], ensure_ascii=False),
            # optional: quick text for simple filtering/search
            "ingredients_text": "; ".join(d.ingredients or []),
        })

    if rebuild or not (persist_dir / "chroma.sqlite").exists():
        vs = Chroma.from_texts(texts=texts, embedding=embed, metadatas=metas,
                               persist_directory=str(persist_dir))
    else:
        vs = Chroma(persist_directory=str(persist_dir), embedding_function=embed)
    return vs

# =========================
# LLM helpers (translate + prefs)
# =========================
TRANS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Translate to English. Return translation if it is not English."),
    ("human", "{text}")
])

def translate_to_english(text: str) -> str:
    if not text:
        return text
    try:
        cands = detect_langs(text)  # e.g., [en:0.76, tl:0.23]
        if cands and cands[0].lang == "en" and cands[0].prob >= 0.6:
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
    include_ingredients: Optional[List[str]] = None   # NEW
    exclude_ingredients: Optional[List[str]] = None   # NEW
    free_text: Optional[str] = None

PREFS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Extract user cooking preferences as compact JSON with keys: "
     "language,cuisine,part_of_meal,part_of_day,heavy_or_light,time_minutes,"
     "difficulty,budget,available_ingredients,servings,allergens,goals,"
     "include_ingredients,exclude_ingredients,free_text. "
     "Infer include_ingredients from any explicit wants; infer exclude_ingredients from phrases like "
     "'no X', 'without Y', 'avoid Z', 'allergic to W'. "
     "Return JSON only."),
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
LAST_HITS: List[Dict[str, Any]] = []  # <-- NEW: cache last recipe candidates


# =========================
# Tools (for the Agent)
# =========================
def _as_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            j = json.loads(v)
            if isinstance(j, list):
                return j
        except Exception:
            # fallback: split on common separators
            return [s.strip() for s in re.split(r"[\n;,•·]", v) if s.strip()]
    return []

# helper to skip category/landing pages and empty stubs
def is_real_recipe(ings, steps, meta):
    title = (meta.get("title") or "").lower()
    url = (meta.get("url") or "").lower()
    # reject obvious Wikibooks category/landing pages
    if "cookbook%3a" in url and ("category" in url or "appetizers" in url or "breakfast" in url):
        return False
    # must have at least some real content
    if len(ings) < 2 and len(steps) < 2:
        return False
    return True

# simple ingredient translations (extend anytime)
ING_TRANSLATIONS = {
    "tl": {
        "lime juice": "katas ng dayap",
        "lime": "dayap",
        "red onion": "pulang sibuyas",
        "onion": "sibuyas",
        "garlic": "bawang",
        "cilantro": "wansoy",
        "tomato": "kamatis",
        "jalapeño": "siling jalapeño",
        "scallion": "dahong sibuyas",
        "soy sauce": "toyo",
        "sesame oil": "mantikang linga",
        "lettuce leaves": "dahon ng litsugas",
        "tortilla chips": "tortilla chips",
        "shrimp": "hipon",
        "scallops": "scallops",
        "salt": "asin",
        "pepper": "paminta",
    }
}

def localize_ingredients(ings: List[str], lang: Optional[str]) -> List[str]:
    mapping = ING_TRANSLATIONS.get((lang or "").lower())
    if not mapping:
        return ings
    # longest-first to avoid partial overlaps
    keys = sorted(mapping.keys(), key=len, reverse=True)
    def repl_line(s: str) -> str:
        t = s
        for k in keys:
            t = re.sub(rf"(?i)\b{re.escape(k)}\b", mapping[k], t)
        return t
    return [repl_line(i) for i in ings]

class VSearchArgs(BaseModel):
    query: str
    top_k: int = 3
    time_limit: Optional[int] = None
    cuisine: Optional[str] = None
    must_include: Optional[List[str]] = None
    exclude_ingredients: Optional[List[str]] = None
    avoid_allergens: Optional[List[str]] = None
    display_lang: Optional[str] = None   # <— NEW

@tool("vector_search", args_schema=VSearchArgs)
def tool_vector_search(query: str, top_k: int = 3, time_limit: Optional[int] = None,
                       cuisine: Optional[str] = None, must_include: Optional[List[str]] = None,
                       exclude_ingredients: Optional[List[str]] = None,
                       avoid_allergens: Optional[List[str]] = None,
                       display_lang: Optional[str] = None) -> str:
    """Semantic search with filters; returns content ready to summarize."""
    global VS
    if VS is None:
        return json.dumps({"hits": [], "note": "vector store not ready"})

    meta_filter = {}
    if cuisine:
        meta_filter["cuisine"] = cuisine
    filter_arg = meta_filter or None

    docs_scores = VS.similarity_search_with_score(
        query, k=max(8, top_k * 3), filter=filter_arg
    )

    def contains_any(items, needles):
        text = " ".join(items or []).lower()
        return any(n in text for n in needles)

    inc = [x.lower() for x in (must_include or [])]
    exc = [x.lower() for x in (exclude_ingredients or [])]
    alr = [x.lower() for x in (avoid_allergens or [])]

    ranked = []
    for doc, dist in docs_scores:
        m = doc.metadata
        ings = _as_list(m.get("ingredients_json") or m.get("ingredients") or m.get("ingredients_text"))
        steps = _as_list(m.get("steps_json") or m.get("steps"))
        tags  = _as_list(m.get("dietary_tags_json") or m.get("dietary_tags"))

        if not is_real_recipe(ings, steps, m):        continue
        if inc and not contains_any(ings, inc):       continue
        if exc and contains_any(ings, exc):           continue
        if alr and contains_any(ings, alr):           continue

        # distance -> relevance in [0,1]
        base = 1.0 - float(dist if dist is not None else 1.0)
        base = max(0.0, min(1.0, base))

        # time penalty
        score = base
        ct = m.get("cook_time")
        if time_limit and isinstance(ct, int) and ct > time_limit:
            overflow = ct - time_limit
            score -= min(0.5, overflow / 60.0)

        # ✅ localize per-hit, then append
        ings_local = localize_ingredients(ings, display_lang)
        ranked.append((score, doc, ings, steps, tags, ings_local))

    ranked.sort(key=lambda x: x[0], reverse=True)

    hits = []
    for score, doc, ings_en, steps, tags, ings_local in ranked[:top_k]:
        m = doc.metadata
        hits.append({
            "id": m.get("id"),
            "title": m.get("title"),
            "url": m.get("url"),
            "source": m.get("source"),
            "image_url": m.get("image_url"),
            "cuisine": m.get("cuisine"),
            "cook_time": m.get("cook_time"),
            "servings": m.get("servings"),
            "dietary_tags": tags,
            "ingredients": ings_en,                # original (English) for tools
            "ingredients_display": ings_local,     # localized for UI
            "steps": steps[:4],
            "content": doc.page_content[:1000],
        })
    # cache for follow-up commands like "shopping list for these"
    global LAST_HITS
    LAST_HITS = hits
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
    """Fast transcript: YouTube API only; no audio download. Returns {'transcript': ''} if unavailable."""
    if "youtube.com" in url_or_path or "youtu.be" in url_or_path:
        vid = _extract_video_id(url_or_path)
        if vid:
            t = _transcript_via_api(vid)
            return json.dumps({"transcript": t or ""})
    if os.getenv("CHEF_TRANSCRIBE", "api_only").lower() != "api_only":
        text = transcribe_media(url_or_path)
        return json.dumps({"transcript": text})
    return json.dumps({"transcript": ""})


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
# ---- Optional: Nutrition estimation ----
class NutriArgs(BaseModel):
    ingredients: List[str]
    servings: Optional[int] = 2
    locale: Optional[str] = "EU"

@tool("estimate_nutrition", args_schema=NutriArgs)
def tool_estimate_nutrition(ingredients: List[str], servings: Optional[int] = 2, locale: Optional[str] = "EU") -> str:
    """Estimate calories and macros per serving from an ingredient list.
    Returns a JSON string with numeric fields (per serving), e.g.:
    {"calories_kcal": 520, "protein_g": 28, "carbs_g": 45, "fat_g": 24}.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Estimate nutrition per serving from ingredient list. Return compact JSON numbers."),
        ("human", "Ingredients:\n{ings}\nServings: {serv}\nLocale: {loc}")
    ])
    out = (prompt | llm_zero()).invoke({"ings":"\n".join(ingredients), "serv": servings, "loc": locale})
    return out.content

# ---- Optional: Shopping list aggregator ----
class ShopArgs(BaseModel):
    recipes: Optional[List[Dict[str, Any]]] = None  
    servings_multiplier: Optional[float] = 1.0
    target_lang: Optional[str] = None 

@tool("make_shopping_list", args_schema=ShopArgs)
def tool_make_shopping_list(recipes: Optional[List[Dict[str, Any]]] = None,
                            servings_multiplier: Optional[float] = 1.0,
                            target_lang: Optional[str] = None) -> str:
    """Aggregate ingredients across recipes into a normalized shopping list.
    Input: list of recipes (each with title and ingredients) and an optional servings_multiplier.
    Returns a grouped list (by aisle) with merged quantities and simple substitutions (as text/JSON).
    """
        # fallback to cached hits if recipes not supplied
    if not recipes:
        from typing import cast
        global LAST_HITS
        recipes = cast(List[Dict[str, Any]], LAST_HITS) or []

    # keep only what's needed (title + ingredients)
    slim = []
    for r in recipes:
        ings = r.get("ingredients") or r.get("ingredients_display") or []
        if isinstance(ings, str):
            ings = [ings]
        slim.append({"title": r.get("title"), "ingredients": ings})

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Aggregate a concise shopping list from the given recipes. "
         "Group by aisle/category (Produce, Pantry, Dairy, Meat/Seafood, Bakery, Frozen, Other). "
         "Merge duplicates with summed quantities when obvious. "
         "Add brief cheaper substitutions where relevant. "
         "Keep it tidy with bullets. Write in the target language if provided."),
        ("human", "Target language: {lang}\nServings x: {mult}\n\nRecipes:\n{payload}")
    ])
    out = (prompt | llm_zero()).invoke({
        "lang": target_lang or "en",
        "mult": servings_multiplier or 1.0,
        "payload": json.dumps({"recipes": slim}, ensure_ascii=False)
    })
    return out.content.strip()

class TranslateArgs(BaseModel):
    text: str
    target_lang: str  # e.g., "pam", "ilo", "ceb", "tl", "ary", "arz"

@tool("translate_text", args_schema=TranslateArgs)
def tool_translate_text(text: str, target_lang: str) -> str:
    """Translate the given text into the target language. Preserve formatting (lists, numbers),
    keep culinary terms natural for the locale, and return ONLY the translated text."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a careful translator. Translate into the target language. "
         "Preserve bullets, numbers, and line breaks. Use natural culinary terms. "
         "Return ONLY the translated text, no preface or notes."),
        ("human", "Target language: {lang}\n\nText:\n{txt}")
    ])
    out = (prompt | llm_zero()).invoke({"lang": target_lang, "txt": text})
    return out.content.strip()


# =========================
# Agent system prompt
# =========================
# SYSTEM = SystemMessage(content=(
# "You are a cheerful kitchen buddy and nutrition coach.\n"
# "Voice & style:\n"
# "- Sound human, warm, and encouraging. Use contractions and 1–2 friendly emojis max (e.g., 🍳🥗), never every line.\n"
# "- Prefer short sentences and compact bullets. Avoid big headings/tables unless the user asks.\n"
# "- Keep it actionable: 2–3 specific suggestions, each with 2–4 quick steps.\n"
# "- Weave links inline with the title; don’t dump raw URLs or long source blocks.\n"
# "- If results look generic (category pages with no real ingredients/steps), skip them.\n"
# "\nWorkflow:\n"
# "1) If input has media, call transcribe_media first.\n"
# "2) Translate to English for search; reply in 'reply_language'.\n"
# "3) Extract preferences JSON.\n"
# "4) Call vector_search first (pass time_limit, cuisine, must_include, exclude_ingredients, avoid_allergens from vector_search_plan). Fallback to keyword_search if needed.\n"
# "5) For each selected result, output a friendly mini-card:\n"
# "   • Title (linked) — 1-line why it fits (time, diet, cravings).\n"
# "   • Key ingredients (≤6).\n"
# "   • 2–4 quick steps (imperative, one line each).\n"
# "   • Optional: tiny tip/substitution.\n"
# "6) Close with one casual question or offer (e.g., 'Want more like this or a quick shopping list?').\n"
# "7) If the user asks for calories/macros/nutrition (e.g., ‘how many calories?’, ‘calorie count of X’), call estimate_nutrition with the ingredients of the most relevant recipe (use recent results if available). Always answer in reply_language.\n"
# "8) If the user asks for a shopping/grocery list and recipes were just shown, call make_shopping_list (recipes may be omitted; use cached hits). Write the list in reply_language.\n"
# "9) If asked, call create_cookbook with selected recipe_ids.\n"
# "10) When replying in a non-English language, prefer ingredients_display if present; otherwise use ingredients.\n"
# "11) If the user asks to translate text, call translate_text with raw_user_text (or the recipe text shown) and the requested language, then reply with ONLY that translated text.\n"
# "12) Always reply in reply_language. If no strong recipe matches, still give 2–3 helpful ideas or substitutions in that same language; do not switch languages or apologize.\n"
# "Keep it concise, friendly, and helpful."
# "Examples (behavioral):\n"
# "- If the user says: "in Spanish" and there is a previous answer, translate your previous answer into Spanish and continue in Spanish next turns.\n"
# "- If the user asks: "calorie count of ceviche", estimate nutrition from a sensible ingredient list and answer in reply_language, e.g. in Spanish:\n"
#   "Aproximado por porción: ~180 kcal (proteína 20–25 g, carbs 6–10 g, grasa 4–6 g). ¿Ajusto porciones o ingredientes?\n"
# " -If the user asks: "translate ceviche recipe in kapampangan", translate the ceviche recipe into Kapampangan and reply with ONLY that translated text.\n"

# ))

SYSTEM = SystemMessage(content="""You are a cheerful kitchen buddy and nutrition coach.

Voice & style
- Sound human, warm, and encouraging. Use contractions and at most 1–2 emojis total (🍳🥗), not every line.
- Prefer short sentences and compact bullets. Avoid big headings/tables unless asked.
- Keep it actionable: give 2–3 concrete suggestions, each with 2–4 quick steps.
- Weave links inline with the title; don’t dump raw URLs or long source blocks.
- If results look generic (category pages with no real ingredients/steps), skip them.

Language rules
- Always reply in reply_language. Detect media/text language for search, but never change reply_language on your own.
- When replying in a non-English language, prefer ingredients_display if present; otherwise use ingredients.
- If the user says “in <language>” or “translate … to <language>”, translate the relevant text and continue in that language next turns. If they provide only a language (no text), translate the previous assistant message.

Workflow
1) If input has media, call transcribe_media first.
2) Translate the user request (and any transcript) to English for retrieval; keep reply_language for the final answer.
3) Extract preferences as JSON with keys:
   language, cuisine, part_of_meal, part_of_day, heavy_or_light, time_minutes, difficulty,
   budget, available_ingredients, servings, allergens, goals, include_ingredients, exclude_ingredients, free_text.
4) If seed_recipe_id is provided, prioritize that recipe; you may summarize it directly without calling transcription.                      
5) Call vector_search first using vector_search_plan (time_limit, cuisine, must_include, exclude_ingredients, avoid_allergens, display_lang). Fallback to keyword_search if needed.
6) If request info is sparse, still suggest 2–3 practical, healthy recipes using common/easy-to-source ingredients in reply_language (no apologies).
7) For each selected result, output a friendly mini-card:
   • [Title](link) — 1-line why it fits (time, diet, cravings).  
   • Key ingredients (≤6).  
   • 2–4 quick steps (imperative, one line each).  
   • Optional: tiny tip/substitution.  
   Include image/thumbnail if available.
8) Close with one casual offer/question (e.g., “Want more like this or a quick shopping list?”).

Tools & follow-ups
- Calories/macros: if the user asks (e.g., “how many calories?”, “calorie count of X”), call estimate_nutrition with the ingredients of the most relevant recipe (use recent results if available). Answer in reply_language with compact numbers per serving.
- Shopping list: if the user asks for a shopping/grocery list and recipes were just shown, call make_shopping_list (recipes may be omitted; use cached hits). Respond in reply_language.
- Cookbook: if asked, call create_cookbook with selected recipe_ids.
- Feedback: if the user gives feedback on a recipe, call add_feedback.
- Translation: if the user asks to translate text, call translate_text with raw_user_text (or the shown recipe text) and reply with ONLY the translated text.
- If you mention nearby stores or restaurants, first ask for their location or use an appropriate lookup tool if available.

Behavioral examples
- “in Spanish” (with a previous answer): translate your previous answer into Spanish and keep Spanish for future turns.
- “calorie count of ceviche”: estimate from a sensible ingredient list and answer in reply_language; e.g., in Spanish: “Aproximado por porción: ~180 kcal (proteína 20–25 g, carbs 6–10 g, grasa 4–6 g). ¿Ajusto porciones o ingredientes?”
- “translate ceviche recipe in kapampangan”: translate that recipe into Kapampangan and reply with ONLY the translated text.

Keep it concise, friendly, and helpful.
""")

def build_agent():
    tools = [
        tool_vector_search,
        tool_keyword_search,
        tool_transcribe_media,
        tool_add_feedback,
        tool_create_cookbook,
        tool_estimate_nutrition,
        tool_make_shopping_list,
        tool_translate_text,
    ]
    return initialize_agent(
        tools=tools,
        llm=ChatOpenAI(model=LLM_MODEL, temperature=CHEF_TEMP),
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=False,
        handle_parsing_errors=True,
        agent_kwargs={"system_message": SYSTEM},
    )

# =========================
# Chat orchestration
# =========================
URL_RE = re.compile(r"(https?://\S+)", re.I)

def get_doc_by_youtube_id(vid: str):
    prefix = f"yt:{vid}"
    for d in DOCS:
        if d.id == prefix:
            return d
    return None


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

def extract_translation_intent(text: str):
    """Return {'lang': 'pam', 'text': '...'} or None. Uses LANG_ALIASES."""
    if not text:
        return None
    t = text.strip()

    # e.g., "translate ceviche recipe in kapampangan" / "... to kapampangan"
    m = re.search(r'^\s*translate\s+(.+?)\s+(?:to|into|in)\s+([a-zA-Z\- ]+)\s*$', t, flags=re.I)
    if m:
        raw = m.group(1).strip()
        lang_name = re.sub(r'\s+', ' ', m.group(2).strip().lower())
        return {"lang": LANG_ALIASES.get(lang_name), "text": raw}

    # e.g., "Homemade Tocino in Kapampangan"
    m2 = re.search(r'^(.*\S)\s+in\s+([a-zA-Z\- ]+)\s*$', t, flags=re.I)
    if m2:
        raw = m2.group(1).strip()
        lang_name = re.sub(r'\s+', ' ', m2.group(2).strip().lower())
        return {"lang": LANG_ALIASES.get(lang_name), "text": raw}

    # e.g., "translate to kapampangan" (no source text)
    m3 = re.search(r'^\s*translate(?:\s+(?:to|into))?\s+([a-zA-Z\- ]+)\s*$', t, flags=re.I)
    if m3:
        lang_name = re.sub(r'\s+', ' ', m3.group(1).strip().lower())
        return {"lang": LANG_ALIASES.get(lang_name), "text": None}

    return None

def ensure_reply_language(text: str, target_lang: Optional[str]) -> str:
    """Return text in target_lang. If it's not already, translate while preserving bullets/format."""
    if not text or not target_lang:
        return text or ""
    try:
        cands = detect_langs(text)
        if cands and cands[0].lang == target_lang and cands[0].prob >= 0.5:
            return text
    except Exception:
        pass
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Translate into the target language. Preserve line breaks, bullets, numbers, and formatting. "
                   "Use natural culinary terms. Return ONLY the translated text."),
        ("human", "Target language: {lang}\n\nText:\n{txt}")
    ])
    out = (prompt | llm_zero()).invoke({"lang": target_lang, "txt": text})
    return out.content.strip()

CALORIE_TRIGGERS = r"(?:calorie|calories|kcal|nutrition|nutritional|macros?|protein|carbs?|fat|kilocal)"
def extract_nutrition_intent(text: str):
    """Return {'query': 'ceviche' or None} if the user is asking for calories/macros."""
    if not text: return None
    t = text.strip().lower()
    if not re.search(CALORIE_TRIGGERS, t): 
        return None
    # Try to pull a dish name from patterns like "calorie count of X", "how many calories in X"
    m = re.search(rf"(?:of|for|in)\s+(.+)$", t)
    dish = (m.group(1).strip(" .?!")) if m else None
    # If the whole message is just "how many calories?" there's no dish
    return {"query": dish}

def locale_from_lang(lang: str) -> str:
    # EU vs US affects some naming; keep simple for now
    return "EU" if (lang or "en").lower() not in {"en-us","arz"} else "US"

def get_ingredients_for_query(q: str, k: int = 1) -> List[str]:
    """Use the in-memory vector store to get ingredients for a query."""
    global VS
    if not VS or not q: 
        return []
    try:
        docs_scores = VS.similarity_search_with_score(q, k=k)
    except Exception:
        return []
    for doc, _ in docs_scores:
        m = doc.metadata
        ings = _as_list(m.get("ingredients_json") or m.get("ingredients") or m.get("ingredients_text"))
        if ings: 
            return ings
    return []

def build_recipe_text_from_query(q: str) -> Optional[str]:
    """Return a compact English recipe text (title + ingredients + steps) for q.
    Try vector store first; fallback to an LLM-generated outline."""
    if not q:
        return None

    # Try vector store (best-effort)
    try:
        global VS
        if VS:
            docs_scores = VS.similarity_search_with_score(q, k=1)
            for doc, _ in docs_scores:
                m = doc.metadata
                title = m.get("title") or q.title()
                ings  = _as_list(m.get("ingredients_json") or m.get("ingredients") or m.get("ingredients_text"))
                steps = _as_list(m.get("steps_json") or m.get("steps"))
                if ings or steps:
                    parts = [f"{title}", ""]
                    if ings:
                        parts.append("Ingredients:")
                        parts += [f"- {i}" for i in ings[:15]]
                        parts.append("")
                    if steps:
                        parts.append("Steps:")
                        parts += [f"{i+1}. {s}" for i, s in enumerate(steps[:8])]
                    return "\n".join(parts).strip()
    except Exception:
        pass

    # Fallback: ask the LLM for a concise recipe skeleton in English
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Write a concise, standard recipe in English for the dish. Keep it practical, 6–12 ingredients and 4–6 short steps. Return ONLY text."),
        ("human", "Dish: {q}")
    ])
    out = (prompt | llm_zero()).invoke({"q": q})
    return out.content.strip() if out and out.content else None

def draft_ingredients_with_llm(dish: str) -> List[str]:
    if not dish:
        return []
    prompt = ChatPromptTemplate.from_messages([
        ("system", "List core ingredients for the dish, 1–2 servings. One item per line. No steps, no chit-chat."),
        ("human", "Dish: {dish}\n\nReturn just the list.")
    ])
    out = (prompt | llm_zero()).invoke({"dish": dish})
    return [ln.strip("-• ").strip() for ln in out.content.splitlines() if ln.strip()]

def chat_once(agent, user_text: str, session_reply_lang: Optional[str] = None, last_bot_text: str = "") -> str:
    t = (user_text or "").strip()

    # ---------- EARLY: TRANSLATION SHORT-CIRCUIT ----------
    m  = re.search(r'^\s*translate\s+(.+?)\s+(?:to|into|in)\s+([a-zA-Z\- ]+)\s*$', t, flags=re.I)
    m2 = re.search(r'^\s*translate(?:\s+(?:to|into))?\s+([a-zA-Z\- ]+)\s*$', t, flags=re.I)
    m3 = re.search(r'^(.*\S)\s+in\s+([a-zA-Z\- ]+)\s*$', t, flags=re.I)
    m4 = re.search(r'^\s*(?:in|en)\s+([a-zA-Z\- ]+)\s*$', t, flags=re.I)
    target_lang, payload = None, None
    if m:
        payload = m.group(1).strip()
        lang_name = re.sub(r'\s+', ' ', m.group(2).strip().lower())
        target_lang = LANG_ALIASES.get(lang_name) or LANG_ALIASES.get(lang_name.lower())
    elif m2:
        payload = last_bot_text
        lang_name = re.sub(r'\s+', ' ', m2.group(1).strip().lower())
        target_lang = LANG_ALIASES.get(lang_name) or LANG_ALIASES.get(lang_name.lower())
    elif m3:
        payload = m3.group(1).strip()
        lang_name = re.sub(r'\s+', ' ', m3.group(2).strip().lower())
        target_lang = LANG_ALIASES.get(lang_name) or LANG_ALIASES.get(lang_name.lower())
    elif m4:
        payload = last_bot_text
        lang_name = re.sub(r'\s+', ' ', m4.group(1).strip().lower())
        target_lang = LANG_ALIASES.get(lang_name) or LANG_ALIASES.get(lang_name.lower())
    if target_lang:
        if not payload:
            return "Paste the text you want me to translate. 🙂"
        try:
            translated = tool_translate_text.invoke({"text": payload, "target_lang": target_lang})
            return (translated.get("content") if isinstance(translated, dict) and "content" in translated else str(translated)).strip()
        except Exception:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a careful translator. Translate into the target language. Preserve bullets, numbers, and line breaks. Use natural culinary terms. Return ONLY the translated text."),
                ("human", "Target language: {lang}\n\nText:\n{txt}")
            ])
            out = (prompt | llm_zero()).invoke({"lang": target_lang, "txt": payload})
            return out.content.strip()

    # ---------- REPLY LANGUAGE ----------
    user_lang_guess = detect_language(user_text)
    if (not user_lang_guess or user_lang_guess == "unknown") and len(user_text) < 20 and user_text.isascii():
        user_lang_guess = "en"
    reply_lang = (FORCE_REPLY_LANG or session_reply_lang or user_lang_guess or "en")

    # ---------- NON-BLOCKING YOUTUBE SUMMARY PATH ----------
    media = maybe_media_url(user_text)
    if media and ("youtube.com" in media or "youtu.be" in media):
        vid = _extract_video_id(media)

        # helpers scoped here to keep this patch self-contained
        def _load_local_yt_json(video_id: Optional[str]) -> Optional[dict]:
            try:
                if not video_id: return None
                p = Path(DEFAULT_YT_DIR) / f"{video_id}.json"
                if p.exists():
                    return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
            return None

        def _kickoff_pipeline_async(url: str):
            """Fire-and-forget: run your pipeline in background to cache structured recipe for next time."""
            try:
                import subprocess, sys, tempfile
                PIPELINE_PATH = Path(r"C:\Users\happy\Documents\ironhack\kusina-bot\final-project\src\backend\scrapers\youtube_recipe_pipeline.py")
                tmp = Path(DEFAULT_YT_DIR) / "_oneurl.txt"
                tmp.write_text(url + "\n", encoding="utf-8")
                # prefer API; keep it quiet
                subprocess.Popen(
                    [sys.executable, str(PIPELINE_PATH), "--urls-file", str(tmp), "--prefer-api", "--max", "1"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
                )
            except Exception:
                pass

        def _mini_card_from_record(rec: dict) -> str:
            r = (rec or {}).get("recipe") or {}
            title = r.get("title") or rec.get("video_title") or "Recipe"
            ings  = r.get("ingredients") or []
            steps = r.get("steps") or []
            url   = rec.get("url") or media
            ings_local = localize_ingredients(ings, reply_lang)
            # cache for shopping list / nutrition
            global LAST_HITS
            LAST_HITS = [{
                "id": f"yt:{vid}" if vid else (rec.get("video_id") or "yt"),
                "title": title, "url": url, "source": "youtube",
                "image_url": f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None,
                "ingredients": ings, "ingredients_display": ings_local,
                "steps": steps[:4]
            }]
            bullets = "\n".join(f"- {i}" for i in ings_local[:6]) if ings_local else "- (walang listahan ng sangkap)"
            steps_b = "\n".join(f"• {s}" for s in steps[:4]) if steps else "• (walang hakbang sa transcript)"
            txt = f"**[{title}]({url})**\n\n**Key ingredients:**\n{bullets}\n\n**Quick steps:**\n{steps_b}\n\nGusto mo bang makita ang buong recipe o gumawa ng shopping list? 🥗"
            return ensure_reply_language(txt, reply_lang)

        # 1) Local per-video JSON
        rec = _load_local_yt_json(vid)
        if rec:
            # also refresh cache in the background (fast API)
            _kickoff_pipeline_async(media)
            return _mini_card_from_record(rec)

        # 2) Fast API transcript (no Whisper)
        tx = _transcript_via_api(vid) if vid else None
        if tx:
            # Quick extract via LLM (no heavy schema), then background cache
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Extract a friendly mini-recipe from transcript: title, 6 key ingredients, 3–5 concise steps. Return in the target language."),
                ("human", "Target language: {lang}\n\nTranscript:\n{tx}")
            ])
            out = (prompt | llm_zero()).invoke({"lang": reply_lang, "tx": tx})
            _kickoff_pipeline_async(media)
            # cache a minimal hit so shopping list / nutrition works immediately
            global LAST_HITS
            LAST_HITS = [{"title": "From video", "url": media, "source": "youtube", "ingredients": [], "ingredients_display": [], "steps": []}]
            return ensure_reply_language(out.content.strip(), reply_lang)

        # 3) Nothing quick available: start background pipeline and respond immediately
        _kickoff_pipeline_async(media)
        return ensure_reply_language("Kukunin ko ang buod ng video sa likod, tapos babalikan kita agad. Samantala, gusto mo ba ng ibang mabilis na ideya habang naghihintay? 🍳", reply_lang)

    # ---------- EARLY: CALORIES / MACROS SHORT-CIRCUIT ----------
    if re.search(r"(?:calorie|calories|kcal|nutrition|nutritional|macros?|protein|carbs?|fat|kilocal)", t, flags=re.I):
        def _ingredients_from_hits():
            try:
                hit = (LAST_HITS or [])[0]
                return (hit.get("ingredients") or hit.get("ingredients_display") or []) if hit else []
            except Exception:
                return []
        def _ingredients_from_query(q: str):
            global VS
            if not VS or not q:
                return []
            try:
                docs_scores = VS.similarity_search_with_score(q, k=1)
                for doc, _d in docs_scores:
                    m = doc.metadata
                    return _as_list(m.get("ingredients_json") or m.get("ingredients") or m.get("ingredients_text")) or []
            except Exception:
                return []
            return []
        def _ingredients_from_text(txt: str):
            if not txt: return []
            rough = re.findall(r"(?m)^\s*[-•]\s*(.+)$", txt)
            return rough[:12]
        def _draft_ingredients_llm(dish: Optional[str]) -> List[str]:
            dish = (dish or "").strip()
            if not dish: return []
            prompt = ChatPromptTemplate.from_messages([
                ("system", "List 6–12 typical ingredients for the dish. Newline-separated. No quantities."),
                ("human", "{dish}")
            ])
            out = (prompt | llm_zero()).invoke({"dish": dish})
            lines = [s.strip(" -•\t") for s in out.content.splitlines() if s.strip()]
            return lines[:12]

        ings = _ingredients_from_hits()
        if not ings:
            m_dish = re.search(r"(?:of|for|in)\s+(.+)$", t)
            dish = (m_dish.group(1).strip(" .?!")) if m_dish else None
            ings = _ingredients_from_query(dish) if dish else ings
        if not ings and last_bot_text:
            ings = _ingredients_from_text(last_bot_text)
        if not ings:
            dish_guess = re.sub(r"^(how many|how much|what.*?)(calories?|kcal|nutrition|macros?).*?(of|for|in)?\s*", "", t, flags=re.I).strip(" .?!")
            ings = _draft_ingredients_llm(dish_guess or "classic ceviche")
        if not ings:
            return ensure_reply_language("Tell me which recipe you want the calorie estimate for (or paste ingredients). 🙂", reply_lang)

        loc = "US" if reply_lang.lower() in {"en", "en-us", "arz"} else "EU"
        try:
            out = tool_estimate_nutrition.invoke({"ingredients": ings, "servings": 1, "locale": loc})
            txt = str(out).strip()
            answer = f"Here’s a rough estimate **per serving**:\n{txt}\n\nWant me to adjust servings or ingredients?"
            return ensure_reply_language(answer, reply_lang)
        except Exception:
            return ensure_reply_language("I couldn't estimate right now. Share the ingredient list and servings, and I’ll calculate it.", reply_lang)

    # ---------- TRANSLATE TO EN FOR SEARCH ----------
    text_en = translate_to_english(user_text)

    # ---------- SHOPPING LIST SHORT-CIRCUIT ----------
    if re.search(r"\b(shopping|grocery)\s+list\b", user_text, flags=re.I):
        try:
            out = tool_make_shopping_list.invoke({
                "recipes": None, "servings_multiplier": 1.0, "target_lang": reply_lang
            })
            return ensure_reply_language(str(out), reply_lang)
        except Exception:
            if not LAST_HITS:
                return ensure_reply_language("Tell me which recipes you want in the shopping list. 🙂", reply_lang)

    # ---------- NO-INDEX QUICK FALLBACK ----------
    if VS is None or not DOCS:
        quick = ChatPromptTemplate.from_messages([
            ("system", "You are a warm kitchen buddy. Write 3 snack/meal ideas that fit the user's vibe. Keep each idea to 1–2 lines with 2–3 quick steps. Use the target language."),
            ("human", "Target language: {lang}\nUser request (English): {req}")
        ])
        out = (quick | llm_zero()).invoke({"lang": reply_lang, "req": text_en})
        return ensure_reply_language(out.content.strip(), reply_lang)

    # ---------- PREFERENCES + SEARCH PLAN ----------
    prefs = extract_prefs(text_en)
    prefs.language = reply_lang
    include = [x.strip() for x in (prefs.include_ingredients or []) if x and x.strip()]
    exclude = [x.strip() for x in (prefs.exclude_ingredients or []) if x and x.strip()]
    avoid   = [x.strip() for x in (prefs.allergens or []) if x and x.strip()]
    search_plan = {
        "time_limit": prefs.time_minutes,
        "cuisine": prefs.cuisine,
        "must_include": include,
        "exclude_ingredients": exclude,
        "avoid_allergens": avoid,
        "display_lang": reply_lang,
    }

    # ---------- AGENT CALL ----------
    directive = {
        "translated_text_en": text_en,
        "raw_user_text": user_text,
        "preferences": prefs.model_dump(),
        "media_transcribed": False,  # avoid tool-triggered transcribe
        "reply_language": reply_lang,
        "vector_search_plan": search_plan,
    }
    result = agent.invoke({"input": f"USER_REQUEST_JSON: {json.dumps(directive, ensure_ascii=False)}"})
    answer = result.get("output", str(result)) or ""
    return ensure_reply_language(answer, reply_lang)


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
    last_bot_text = ""

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
                # If we have something to translate, show it immediately in the new language
                if last_bot_text.strip():
                    print("\nAssistant:\n" + ensure_reply_language(last_bot_text, maybe_lang) + "\n")
                else:
                    print(f"\nAssistant:\nOkay! I’ll reply in {pretty} from now on.\n")
                continue

            # 2) Ephemeral auto-switch for THIS TURN if confident + long message
            ephemeral_lang = SESSION_REPLY_LANG
            det = detect_language(user)
            if det in LANG_ALIASES.values() and det != SESSION_REPLY_LANG and len(user) >= 60:
                ephemeral_lang = det
                # (Optional) let them know how to lock it
                # print(f"[note] Detected {det}; replying in that language for this turn. Use '/lang {det}' to switch permanently.")

            answer = chat_once(agent, user, session_reply_lang=ephemeral_lang, last_bot_text=last_bot_text)

            print("\nAssistant:\n" + answer + "\n")
            last_bot_text = answer

        except KeyboardInterrupt:
            print("\nBye!")
            break

if __name__ == "__main__":
    main()

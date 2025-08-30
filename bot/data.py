from __future__ import annotations
# --- hydrate env from Streamlit Secrets (cloud) + .env (local) ---
import os
try:
    import streamlit as st  # will be available in Streamlit Cloud
    for k, v in st.secrets.items():
        if isinstance(v, (str, int, float, bool)):
            os.environ.setdefault(str(k), str(v))
except Exception:
    pass

try:
    from dotenv import load_dotenv  # no-op in cloud if .env not present
    load_dotenv()
except Exception:
    pass

import json, os, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
# --- Embedding backend selector ---
# import os
# from dotenv import load_dotenv
# load_dotenv()  # pick up .env locally without deploy
def get_embedder():
    """
    Backends:
      - EMBED_BACKEND=openai -> OpenAIEmbeddings (needs OPENAI_API_KEY)
      - EMBED_BACKEND=fastembed -> FastEmbed (no Torch, uses onnxruntime)
      - EMBED_BACKEND=local -> SentenceTransformer, but falls back to FastEmbed in cloud
    """
    backend = os.getenv("EMBED_BACKEND", "fastembed").lower()

    if backend == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if key:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=os.getenv("CHEF_EMBED_MODEL", "text-embedding-3-small"),
                timeout=30,
                max_retries=1,
            )
        # no key → fall through to fastembed

    if backend in ("fastembed", "auto"):
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
        model = os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
        return FastEmbedEmbeddings(model_name=model)

    if backend == "local":
        # Try SentenceTransformer, but avoid breaking if Torch not OK
        try:
            from langchain_community.embeddings import SentenceTransformerEmbeddings
            model = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")
            return SentenceTransformerEmbeddings(model_name=model, model_kwargs={"device": "cpu"})
        except Exception:
            from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
            model = os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
            return FastEmbedEmbeddings(model_name=model)

    # safety net
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    return FastEmbedEmbeddings(model_name=os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5"))

# def get_embedder():
#     """
#     Backends:
#       - EMBED_BACKEND=openai  -> OpenAIEmbeddings (needs OPENAI_API_KEY)
#       - EMBED_BACKEND=fastembed (recommended on Streamlit Cloud w/o key)
#       - EMBED_BACKEND=local -> SentenceTransformer (Torch); falls back to fastembed if Torch fails
#     """
#     backend = os.getenv("EMBED_BACKEND", "openai").lower()

#     if backend == "openai":
#         key = os.getenv("OPENAI_API_KEY")
#         if key:
#             from langchain_openai import OpenAIEmbeddings
#             return OpenAIEmbeddings(
#                 model=os.getenv("CHEF_EMBED_MODEL", "text-embedding-3-small"),
#                 timeout=30,
#                 max_retries=1,
#             )
#         # no key → fall through to fastembed

#     if backend in ("fastembed", "auto"):
#         from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
#         # good, small, widely supported model:
#         model = os.getenv("FASTEMBED_MODEL", "intfloat/e5-small-v2")
#         return FastEmbedEmbeddings(model_name=model)

#     if backend == "local":
#         # Try SentenceTransformer (Torch) but auto-fallback to FastEmbed if it explodes
#         try:
#             from langchain_community.embeddings import SentenceTransformerEmbeddings
#             model = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")
#             return SentenceTransformerEmbeddings(model_name=model, model_kwargs={"device": "cpu"})
#         except Exception:
#             from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
#             model = os.getenv("FASTEMBED_MODEL", "intfloat/e5-small-v2")
#             return FastEmbedEmbeddings(model_name=model)

#     # default safety net
#     from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
#     return FastEmbedEmbeddings(model_name=os.getenv("FASTEMBED_MODEL", "intfloat/e5-small-v2"))

# def get_embedder():
#     """
#     Choose embeddings backend from env/Secrets.
#     - EMBED_BACKEND=local => SentenceTransformer
#     - EMBED_BACKEND=openai (default) => OpenAIEmbeddings
#       If OPENAI_API_KEY is missing, gracefully fall back to local.
#     """
#     backend = os.getenv("EMBED_BACKEND", "openai").lower()

#     # Local backend (no key needed)
#     if backend == "local":
#         from langchain_community.embeddings import SentenceTransformerEmbeddings
#         model = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")
#         return SentenceTransformerEmbeddings(model_name=model)

#     # OpenAI backend (preferred)
#     key = os.getenv("OPENAI_API_KEY")
#     if key:
#         from langchain_openai import OpenAIEmbeddings
#         return OpenAIEmbeddings(
#             model=os.getenv("CHEF_EMBED_MODEL", "text-embedding-3-small"),
#             timeout=30,
#             max_retries=1,
#         )

#     # Safety net: auto-fallback to local if key missing
#     from langchain_community.embeddings import SentenceTransformerEmbeddings
#     model = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")
#     return SentenceTransformerEmbeddings(model_name=model)

# def get_embedder():
#     """
#     Returns a LangChain embeddings object based on env:
#       - EMBED_BACKEND=openai + OPENAI_API_KEY
#       - EMBED_BACKEND=local + LOCAL_EMBED_MODEL (e.g., all-MiniLM-L6-v2)
#     """
#     backend = os.getenv("EMBED_BACKEND", "openai").lower()
#     if backend == "local":
#         from langchain_community.embeddings import SentenceTransformerEmbeddings
#         model = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")
#         return SentenceTransformerEmbeddings(model_name=model)

#     # Default: OpenAI
#     from langchain_openai import OpenAIEmbeddings
#     if not os.getenv("OPENAI_API_KEY"):
#         raise RuntimeError(
#             "OPENAI_API_KEY missing. Set it or switch to local embeddings "
#             "(EMBED_BACKEND=local, LOCAL_EMBED_MODEL=all-MiniLM-L6-v2)."
#         )
#     return OpenAIEmbeddings(
#         model=os.getenv("CHEF_EMBED_MODEL", "text-embedding-3-small"),
#         timeout=30,        # don’t hang forever
#         max_retries=1,     # fail fast
#     )

# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings

# ---- Centralized paths (update here only) ----
# DEFAULT_YT_DIR   = Path(r"C:\Users\happy\Documents\ironhack\kusina-bot\final-project\src\backend\scrapers\data\recipes")
# DEFAULT_YT_JSONL = Path(r"C:\Users\happy\Documents\ironhack\kusina-bot\final-project\src\backend\scrapers\data\recipes\recipes.jsonl")
# DEFAULT_WB_DIR   = Path(r"C:\Users\happy\Documents\ironhack\kusina-bot\final-project\src\backend\scrapers\data\open_wikibooks_toc")
# DEFAULT_WB_JSONL = Path(r"C:\Users\happy\Documents\ironhack\kusina-bot\final-project\src\backend\scrapers\data\open_wikibooks_toc\recipes.jsonl")
# DEFAULT_VS_DIR   = Path(r"C:\Users\happy\Documents\ironhack\kusina-bot\final-project\src\backend\scrapers\data\vs")
# # --- NEW: local data you collected ---
# DEFAULT_SCRAPED_CSV = Path(r"C:\Users\happy\Documents\ironhack\kusina-bot\Reverse1\final-project\bot\data_recipes\txt\recipes_extracted_20250823_114251.csv")
# DEFAULT_KUSINA_DIR  = Path(r"C:\Users\happy\Documents\ironhack\kusina-bot\Reverse1\final-project\bot\data_kusina")
# KUSINA_INDEX_JSONL  = DEFAULT_KUSINA_DIR / "index" / "recipes_transcripts.jsonl"
# KUSINA_CHUNKS_DIR   = DEFAULT_KUSINA_DIR / "chunks"
# KUSINA_RAW_DIR      = DEFAULT_KUSINA_DIR / "raw"

STATE_DIR  = Path(__file__).resolve().parents[1] / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

CANON_PATH = Path(__file__).resolve().parent / "canon_foods.jsonl"


# bot/data.py – put near the top
APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_YT_DIR   = DATA_DIR / "recipes"
DEFAULT_YT_JSONL = DEFAULT_YT_DIR / "recipes.jsonl"
DEFAULT_WB_DIR   = DATA_DIR / "open_wikibooks_toc"
DEFAULT_WB_JSONL = DEFAULT_WB_DIR / "recipes.jsonl"
DEFAULT_VS_DIR   = DATA_DIR / "vs"

DEFAULT_SCRAPED_CSV = APP_ROOT / "bot" / "data_recipes" / "txt" / "recipes_extracted.csv"
DEFAULT_KUSINA_DIR  = APP_ROOT / "bot" / "data_kusina"

STATE_DIR  = APP_ROOT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
KUSINA_INDEX_JSONL  = DEFAULT_KUSINA_DIR / "index" / "recipes_transcripts.jsonl"
KUSINA_CHUNKS_DIR   = DEFAULT_KUSINA_DIR / "chunks"
KUSINA_RAW_DIR      = DEFAULT_KUSINA_DIR / "raw"

class MemoryStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("{}", encoding="utf-8")
    def get(self, user_id: str) -> Dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8")).get(user_id, {})
        except Exception:
            return {}
    def update(self, user_id: str, fields: Dict[str, Any]) -> None:
        try:
            blob = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            blob = {}
        cur = blob.get(user_id, {})
        cur.update(fields or {})
        blob[user_id] = cur
        self.path.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")

# --- in bot/data.py ---

CONTINENT_BY_COUNTRY = {
    # tiny starter map; extend anytime
    "philippines": "asia", "japan": "asia", "china": "asia",
    "italy": "europe", "spain": "europe", "france": "europe",
    "mexico": "north america", "usa": "north america",
    "peru": "south america", "brazil": "south america",
    "morocco": "africa", "egypt": "africa",
}

class RecipeDoc(BaseModel):
    id: str
    title: str
    url: str
    source: str                 # youtube | wikibooks | web | canonical
    image_url: Optional[str] = None
    ingredients: List[str] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list)

    # NEW: geography + taxonomy
    continent: Optional[str] = None          # "asia", "europe", ...
    country: Optional[str] = None            # "philippines", "italy", ...
    region: Optional[str] = None             # "luson", "sicily", etc (optional)
    cuisine: Optional[str] = None            # keep existing (e.g., "Filipino")
    dish_type: Optional[str] = None          # "stew", "soup", "noodles", "stir-fry"
    course: Optional[str] = None             # "breakfast", "main", "dessert"

    cook_time_minutes: Optional[int] = None
    servings: Optional[int] = None
    dietary_tags: List[str] = Field(default_factory=list)

    # NEW: popularity/meta-signals
    popularity_score: Optional[float] = None     # 0..1 blended (views, upvotes, “popular dish” lists)
    signals: Dict[str, Any] = Field(default_factory=dict)  # {"yt_views": 1_200_000, "rank_in_list": 3, ...}

    extras: Dict[str, Any] = Field(default_factory=dict)

    @property
    def search_text(self) -> str:
        return " ".join([
            self.title or "",
            self.cuisine or "",
            self.country or "",
            self.continent or "",
            " ".join(self.ingredients or []),
            self.dish_type or "",
            self.course or "",
        ]).lower()

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,      # ~800–1200 chars is a good start
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " "]
)

def _chunk(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    return _splitter.split_text(text)


def _jsonl_load(path: Optional[Path]) -> List[Dict[str, Any]]:
    out = []
    if path and path.exists():
        for ln in path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                try: out.append(json.loads(ln))
                except Exception: pass
    return out

def _load_canon_foods_as_texts_metas():
    """Return (texts, metas) for canon_foods.jsonl so we can index with Chroma.from_texts."""
    if not CANON_PATH.exists():
        return [], []
    texts, metas = [], []
    for line in CANON_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        aliases = ", ".join(a.get("text", "") for a in obj.get("aliases", []))
        blob = (
            f"{obj.get('name','')} ({obj.get('country','')})\n"
            f"Aliases: {aliases}\n"
            f"Category: {obj.get('category','')}\n"
           f"Def: {obj.get('short_def','')}\n"
            f"Typical ingredients: {', '.join(obj.get('typical_ingredients', []))}"
        ).strip()

        texts.append(blob)
        metas.append({
            "id": obj.get("id", ""),
            "title": obj.get("name", ""),
            "url": "",
            "source": "canon",
            "image_url": "",
            "cuisine": obj.get("category",""),
            "country": obj.get("country",""),
            "continent": "",
            "region": obj.get("region",""),
            "dish_type": obj.get("category",""),
            "course": "",
            "cook_time": None,
            "servings": None,
            "popularity_score": 100,  # small boost so canon ranks well
            "signals_json": json.dumps({}, ensure_ascii=False),

            "ingredients_json": json.dumps(obj.get("typical_ingredients", []), ensure_ascii=False),
            "steps_json": json.dumps([], ensure_ascii=False),
            "dietary_tags_json": json.dumps([], ensure_ascii=False),

            "ingredients_text": ", ".join(obj.get("typical_ingredients", [])),
            "lang": obj.get("lang",""),
            "aliases": aliases,
        })
    return texts, metas

def load_scraped_recipes(csv_path: Path = DEFAULT_SCRAPED_CSV) -> List[RecipeDoc]:
    docs: List[RecipeDoc] = []
    if not csv_path or not csv_path.exists():
        return docs
    import csv
    seen = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = (row.get("title") or "").strip()
            if not title:
                continue
            rid = f"scraped:{title.lower()[:80]}"
            if rid in seen:
                continue
            # Ingredients & steps normalization
            ings = [x.strip("•·- ").strip() for x in (row.get("ingredients") or "").split("\n") if x.strip()]
            steps = [x.strip("1234567890). ").strip() for x in (row.get("steps") or "").split("\n") if x.strip()]

            # time/servings if present
            cook_minutes = None
            for k in ("time_total", "time_cook", "time_prep"):
                v = row.get(k)
                if v and str(v).strip().isdigit():
                    cook_minutes = int(v); break
            servings = None
            sv = (row.get("servings") or "").strip()
            if sv.isdigit():
                servings = int(sv)

            d = RecipeDoc(
                id=rid,
                title=title,
                url=row.get("url") or "",
                source="scraped",
                image_url=row.get("image_url") or None,
                ingredients=ings,
                steps=steps[:20],   # cap very long lists
                cook_time_minutes=cook_minutes,
                servings=servings,
                cuisine=row.get("cuisine") or None,
            )
            docs.append(d)
            seen.add(rid)
    return docs

def load_scraped_recipes(csv_path: Path = DEFAULT_SCRAPED_CSV) -> List[RecipeDoc]:
    docs: List[RecipeDoc] = []
    if not csv_path or not csv_path.exists():
        return docs
    import csv
    seen = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = (row.get("title") or "").strip()
            if not title:
                continue
            rid = f"scraped:{title.lower()[:80]}"
            if rid in seen:
                continue
            # Ingredients & steps normalization
            ings = [x.strip("•·- ").strip() for x in (row.get("ingredients") or "").split("\n") if x.strip()]
            steps = [x.strip("1234567890). ").strip() for x in (row.get("steps") or "").split("\n") if x.strip()]

            # time/servings if present
            cook_minutes = None
            for k in ("time_total", "time_cook", "time_prep"):
                v = row.get(k)
                if v and str(v).strip().isdigit():
                    cook_minutes = int(v); break
            servings = None
            sv = (row.get("servings") or "").strip()
            if sv.isdigit():
                servings = int(sv)

            d = RecipeDoc(
                id=rid,
                title=title,
                url=row.get("url") or "",
                source="scraped",
                image_url=row.get("image_url") or None,
                ingredients=ings,
                steps=steps[:20],   # cap very long lists
                cook_time_minutes=cook_minutes,
                servings=servings,
                cuisine=row.get("cuisine") or None,
            )
            docs.append(d)
            seen.add(rid)
    return docs

def load_transcript_sketches(index_jsonl: Path = KUSINA_INDEX_JSONL, chunks_dir: Path = KUSINA_CHUNKS_DIR) -> List[RecipeDoc]:
    """
    Convert your chunked transcripts into light recipe docs:
      - keep chunks with step_likeness >= 2.0
      - collect unique ingredient_hints as ingredients[]
      - steps[] are sentence-ish extractions from high-scoring chunks
    """
    docs: List[RecipeDoc] = []
    if not index_jsonl or not index_jsonl.exists() or not chunks_dir or not chunks_dir.exists():
        return docs

    # Load index to know which videos exist
    index_objs = _jsonl_load(index_jsonl)
    by_vid = {}
    for o in index_objs:
        vid = o.get("id") or o.get("video_id")
        if vid:
            by_vid[vid] = o

    import json
    import re
    def sentencify(text: str) -> List[str]:
        # tiny sentence splitter
        parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
        return [p.strip() for p in parts if len(p.strip()) > 0]

    for vid, meta in by_vid.items():
        p = chunks_dir / f"{vid}.jsonl"
        if not p.exists():
            continue
        hints: List[str] = []
        steps: List[str] = []
        seen_hint = set()
        try:
            for ln in p.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                obj = json.loads(ln)
                step_score = float(obj.get("step_likeness") or 0.0)
                txt = obj.get("text") or ""
                # ingredient hints
                for ing in obj.get("ingredient_hints") or []:
                    ing_l = ing.lower().strip()
                    if ing_l and ing_l not in seen_hint:
                        hints.append(ing)
                        seen_hint.add(ing_l)
                # step-like sentences
                if step_score >= 2.0 and txt:
                    steps += sentencify(txt)
        except Exception:
            continue

        title = meta.get("title") or f"YouTube video {vid}"
        url = meta.get("webpage_url") or ""
        img = f"https://img.youtube.com/vi/{vid}/hqdefault.jpg"
        # keep it compact
        steps = steps[:14]
        hints = hints[:24]

        # If we ended up with nothing substantial, skip
        if len(hints) < 2 and len(steps) < 2:
            continue

        d = RecipeDoc(
            id=f"yttr:{vid}",
            title=title,
            url=url,
            source="youtube:transcript",
            image_url=img,
            ingredients=hints,
            steps=steps,
        )
        docs.append(d)
    return docs

def _yt_obj_to_doc(obj: Dict[str, Any]) -> Optional[RecipeDoc]:
    rec = obj.get("recipe", {}) or {}
    vid = obj.get("video_id") or obj.get("id")
    if not vid and "url" in obj:
        m = re.search(r"(?:v=|\.be/|/shorts/|/embed/)([\w-]{11})", obj["url"])
        vid = m.group(1) if m else None
    url = obj.get("url") or (f"https://www.youtube.com/watch?v={vid}" if vid else "")
    title = rec.get("title") or obj.get("video_title") or "Untitled"
    img = f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None
    return RecipeDoc(
        id=f"yt:{vid or title}",
        title=title, url=url, source="youtube", image_url=img,
        ingredients=rec.get("ingredients") or [], steps=rec.get("steps") or [],
        cuisine=rec.get("cuisine"), cook_time_minutes=rec.get("cook_time_minutes"),
        extras={"channel": obj.get("channel")}
    )

def _wb_obj_to_doc(obj: Dict[str, Any]) -> Optional[RecipeDoc]:
    rec = obj.get("recipe", {}) or {}
    title = rec.get("title") or obj.get("title") or "Untitled"
    url = obj.get("source_url") or obj.get("url") or ""
    return RecipeDoc(
        id=f"wb:{title}", title=title, url=url, source="wikibooks",
        ingredients=rec.get("ingredients") or [], steps=rec.get("steps") or [],
        cuisine=rec.get("cuisine"), cook_time_minutes=rec.get("cook_time_minutes"),
        extras={"license": obj.get("license"), "attribution": obj.get("attribution")}
    )

def load_youtube(yt_dir: Path = DEFAULT_YT_DIR, yt_jsonl: Optional[Path] = DEFAULT_YT_JSONL) -> List[RecipeDoc]:
    docs, seen = [], set()
    for obj in _jsonl_load(yt_jsonl):
        d = _yt_obj_to_doc(obj); 
        if d and d.id not in seen: docs.append(d); seen.add(d.id)
    if yt_dir and yt_dir.exists():
        for p in yt_dir.glob("*.json"):
            if p.suffix.lower()==".jsonl": continue
            try: obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception: continue
            d = _yt_obj_to_doc(obj)
            if d and d.id not in seen: docs.append(d); seen.add(d.id)
    return docs

def load_wikibooks(wb_dir: Path = DEFAULT_WB_DIR, wb_jsonl: Optional[Path] = DEFAULT_WB_JSONL) -> List[RecipeDoc]:
    docs, seen = [], set()
    for obj in _jsonl_load(wb_jsonl):
        d = _wb_obj_to_doc(obj); 
        if d and d.id not in seen: docs.append(d); seen.add(d.id)
    if wb_dir and wb_dir.exists():
        for p in wb_dir.glob("*.json"):
            if p.suffix.lower()==".jsonl": continue
            try: obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception: continue
            d = _wb_obj_to_doc(obj)
            if d and d.id not in seen: docs.append(d); seen.add(d.id)
    return docs

def load_all_docs() -> List[RecipeDoc]:
    # 1) canonical sources (YouTube + Wikibooks)
    base = load_youtube() + load_wikibooks()
    # 2) new scraped CSV
    scraped = load_scraped_recipes()
    # 3) transcript-derived sketches
    transcripts = load_transcript_sketches()
    return base + scraped + transcripts

def _doc_to_page(d: RecipeDoc) -> str:
    return "\n".join([
        d.title or "",
        f"Cuisine: {d.cuisine}" if d.cuisine else "",
        "Ingredients:\n" + "\n".join(d.ingredients or []),
        "Steps:\n" + "\n".join(d.steps or []),
    ]).strip()

# def build_or_load_vectorstore(docs: List["RecipeDoc"], persist_dir: Path, rebuild: bool = False):
# def build_or_load_vectorstore(docs: List[RecipeDoc], persist_dir: Path = DEFAULT_VS_DIR, rebuild: bool = False):
#     """
#     Merge your scraped recipe docs with the canon and build/load a Chroma index.
#     """
#     # EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
#     # embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
#     embed = OpenAIEmbeddings(
#         model=os.getenv("CHEF_EMBED_MODEL", "text-embedding-3-small"),
#         timeout=30,
#         max_retries=1,
#     )
#     persist_dir.mkdir(parents=True, exist_ok=True)

#     # 1) Existing recipe docs → texts+metas
#     texts, metas = [], []
#     for d in docs:
#         texts.append(_doc_to_page(d))
#         metas.append({
#             "id": d.id, "title": d.title, "url": d.url, "source": d.source,
#             "image_url": d.image_url,
#             "cuisine": d.cuisine, "country": d.country, "continent": d.continent,
#             "region": d.region, "dish_type": d.dish_type, "course": d.course,
#             "cook_time": d.cook_time_minutes, "servings": d.servings,
#             "popularity_score": d.popularity_score,
#             "signals_json": json.dumps(d.signals or {}, ensure_ascii=False),

#             "ingredients_json": json.dumps(d.ingredients or [], ensure_ascii=False),
#             "steps_json": json.dumps(d.steps or [], ensure_ascii=False),
#             "dietary_tags_json": json.dumps(d.dietary_tags or [], ensure_ascii=False),

#             "ingredients_text": "; ".join(d.ingredients or []),
#             "lang": getattr(d, "lang", "") or "",
#             "aliases": "",
#         })

#     # 2) Append canon
#     canon_texts, canon_metas = _load_canon_foods_as_texts_metas()
#     texts.extend(canon_texts)
#     metas.extend(canon_metas)

#     # 3) Build/load Chroma
#     db_file = persist_dir / "chroma.sqlite"
#     if rebuild or not db_file.exists():
#         vs = Chroma.from_texts(texts=texts, embedding=embed, metadatas=metas, persist_directory=str(persist_dir))
#     else:
#         vs = Chroma(persist_directory=str(persist_dir), embedding_function=embed)
#     return vs


def build_or_load_vectorstore(
    docs: List["RecipeDoc"],
    persist_dir: Path = DEFAULT_VS_DIR,
    rebuild: bool = False,
) -> Chroma:
    """
    Build or load a persistent Chroma index.
    - Merges provided RecipeDoc pages with canonical docs.
    - Chunks long pages to improve embedding/retrieval.
    - Embedding backend is selected by get_embedder().
    """
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Rebuild := wipe directory to avoid stale collections
    if rebuild and persist_dir.exists():
        try:
            shutil.rmtree(persist_dir)
        except Exception:
            pass
        persist_dir.mkdir(parents=True, exist_ok=True)

    # 1) Convert RecipeDoc -> page text + metadata
    texts: list[str] = []
    metas: list[dict] = []

    for d in docs or []:
        page = _doc_to_page(d)  # your existing formatter
        chunks = _chunk(page)
        if not chunks:
            continue
        for i, ch in enumerate(chunks):
            texts.append(ch)
            metas.append({
                "id": f"{d.id}::chunk:{i}",
                "parent_id": d.id,
                "title": d.title,
                "url": d.url,
                "source": d.source,
                "image_url": d.image_url,
                "cuisine": d.cuisine,
                "country": d.country,
                "continent": d.continent,
                "region": d.region,
                "dish_type": d.dish_type,
                "course": d.course,
                "cook_time": d.cook_time_minutes,
                "servings": d.servings,
                "popularity_score": d.popularity_score,
                "signals_json": json.dumps(d.signals or {}, ensure_ascii=False),
                "ingredients_json": json.dumps(d.ingredients or [], ensure_ascii=False),
                "steps_json": json.dumps(d.steps or [], ensure_ascii=False),
                "dietary_tags_json": json.dumps(d.dietary_tags or [], ensure_ascii=False),
                "ingredients_text": "; ".join(d.ingredients or []),
                "lang": getattr(d, "lang", "") or "",
                "aliases": "",
            })

    # 2) Append canonical docs
    canon_texts, canon_metas = _load_canon_foods_as_texts_metas()
    texts.extend(canon_texts or [])
    metas.extend(canon_metas or [])

    # 3) Filter out empties and keep lists aligned
    keep: list[Tuple[str, dict]] = [(t, m) for t, m in zip(texts, metas) if (t or "").strip()]
    if keep:
        texts, metas = list(zip(*keep))
        texts, metas = list(texts), list(metas)
    else:
        texts, metas = [], []

    # 4) Create / load Chroma
    embed = get_embedder()
    db_file = persist_dir / "chroma.sqlite"
    if (texts and rebuild) or (texts and not db_file.exists()):
        vs = Chroma.from_texts(
            texts=texts,
            embedding=embed,
            metadatas=metas,
            persist_directory=str(persist_dir),
            collection_name="recipes",
        )
    else:
        # load existing collection (may still add later in your app)
        vs = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embed,
            collection_name="recipes",
        )

    return vs
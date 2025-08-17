from __future__ import annotations
import json, os, re
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# ---- Centralized paths (update here only) ----
DEFAULT_YT_DIR   = Path(r"C:\Users\happy\Documents\ironhack\kusina-bot\final-project\src\backend\scrapers\data\recipes")
DEFAULT_YT_JSONL = Path(r"C:\Users\happy\Documents\ironhack\kusina-bot\final-project\src\backend\scrapers\data\recipes\recipes.jsonl")
DEFAULT_WB_DIR   = Path(r"C:\Users\happy\Documents\ironhack\kusina-bot\final-project\src\backend\scrapers\data\open_wikibooks_toc")
DEFAULT_WB_JSONL = Path(r"C:\Users\happy\Documents\ironhack\kusina-bot\final-project\src\backend\scrapers\data\open_wikibooks_toc\recipes.jsonl")
DEFAULT_VS_DIR   = Path(r"C:\Users\happy\Documents\ironhack\kusina-bot\final-project\src\backend\scrapers\data\vs")

STATE_DIR  = Path(__file__).resolve().parents[1] / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

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


def _jsonl_load(path: Optional[Path]) -> List[Dict[str, Any]]:
    out = []
    if path and path.exists():
        for ln in path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                try: out.append(json.loads(ln))
                except Exception: pass
    return out

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
    return load_youtube() + load_wikibooks()

def _doc_to_page(d: RecipeDoc) -> str:
    return "\n".join([
        d.title or "",
        f"Cuisine: {d.cuisine}" if d.cuisine else "",
        "Ingredients:\n" + "\n".join(d.ingredients or []),
        "Steps:\n" + "\n".join(d.steps or []),
    ]).strip()

def build_or_load_vectorstore(docs: List[RecipeDoc], persist_dir: Path = DEFAULT_VS_DIR, rebuild: bool = False):
    embed = OpenAIEmbeddings(model=os.getenv("CHEF_EMBED_MODEL", "text-embedding-3-small"))
    persist_dir.mkdir(parents=True, exist_ok=True)
    texts, metas = [], []
    for d in docs:
        texts.append(_doc_to_page(d))
        metas.append({
            "id": d.id, "title": d.title, "url": d.url, "source": d.source,
            "image_url": d.image_url,
            "cuisine": d.cuisine, "country": d.country, "continent": d.continent,
            "region": d.region, "dish_type": d.dish_type, "course": d.course,
            "cook_time": d.cook_time_minutes, "servings": d.servings,
            "popularity_score": d.popularity_score,
            "signals_json": json.dumps(d.signals or {}, ensure_ascii=False),

            "ingredients_json": json.dumps(d.ingredients or [], ensure_ascii=False),
            "steps_json": json.dumps(d.steps or [], ensure_ascii=False),
            "dietary_tags_json": json.dumps(d.dietary_tags or [], ensure_ascii=False),

            "ingredients_text": "; ".join(d.ingredients or []),
        })
        
    db_file = persist_dir / "chroma.sqlite"
    if rebuild or not db_file.exists():
        vs = Chroma.from_texts(texts=texts, embedding=embed, metadatas=metas, persist_directory=str(persist_dir))
    else:
        vs = Chroma(persist_directory=str(persist_dir), embedding_function=embed)
    return vs

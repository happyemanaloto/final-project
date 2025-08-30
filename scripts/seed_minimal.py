# --- bootstrap: add project root to sys.path so `bot/` is importable ---
from pathlib import Path
import sys, os, json

ROOT = Path(__file__).resolve().parents[1]  # -> final-project/
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Load .env (so EMBED_BACKEND / OPENAI_API_KEY etc. are picked up)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Chroma import (new package first, fallback to legacy)
try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

from bot.data import get_embedder, DEFAULT_VS_DIR

def _flat_meta(m: dict) -> dict:
    """
    Chroma only accepts str/int/float/bool/None in metadata.
    Convert lists/dicts/tuples to JSON strings with *_json suffix.
    """
    out = {}
    for k, v in (m or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[f"{k}_json"] = json.dumps(v, ensure_ascii=False)
    return out

def main():
    print("Persist dir:", DEFAULT_VS_DIR)
    DEFAULT_VS_DIR.mkdir(parents=True, exist_ok=True)

    embed = get_embedder()

    texts = [
        "Chicken Adobo: chicken, soy sauce, vinegar, garlic, bay leaf. Simmer 20–25 minutes; glaze at the end.",
        "Vegetable Stir-Fry: broccoli, bell pepper, carrots, soy, ginger, garlic, noodles. 10–15 minutes total.",
    ]
    metas_raw = [
        {"id": "seed-demo-1", "title": "Chicken Adobo Express", "cuisine": "filipino",
         "cook_time": 25, "tags": ["chicken","adobo","under-30"]},
        {"id": "seed-demo-2", "title": "Quick Veg Stir-Fry",   "cuisine": "asian",
         "cook_time": 15, "tags": ["vegetarian","stir-fry","under-30"]},
    ]
    # 🔧 Flatten complex metadata
    metas = [_flat_meta(m) for m in metas_raw]
    ids = [m["id"] for m in metas_raw]

    vs = Chroma(
        collection_name="recipes",
        embedding_function=embed,
        persist_directory=str(DEFAULT_VS_DIR),
    )

    try:
        vs.add_texts(texts=texts, metadatas=metas, ids=ids)
    except Exception as e:
        # If run twice, IDs may already exist -> ignore that specific case
        if "already exists" not in str(e).lower():
            raise

    try:
        n = vs._collection.count()
    except Exception:
        n = -1
    print("Seed complete. Collection count:", n)

if __name__ == "__main__":
    main()

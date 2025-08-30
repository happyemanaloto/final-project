# --- bootstrap: add project root to sys.path ---
from pathlib import Path
import sys, os
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from bot.data import get_embedder, DEFAULT_VS_DIR
from langchain_community.vectorstores import Chroma

vs = Chroma(collection_name="recipes",
            embedding_function=get_embedder(),
            persist_directory=str(DEFAULT_VS_DIR))
try:
    n = vs._collection.count()
except Exception:
    n = -1
print("Collection count:", n)

items = vs._collection.get(limit=3, include=["documents","metadatas"])
for i, (doc, meta) in enumerate(zip(items.get("documents", []), items.get("metadatas", [])), 1):
    print(f"\nDoc {i}:")
    print((doc or "")[:200].replace("\n", " "))
    print("Meta keys:", list((meta or {}).keys()))

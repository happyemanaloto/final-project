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

from langchain_community.vectorstores import Chroma
from bot.data import get_embedder, DEFAULT_VS_DIR

if len(sys.argv) < 2:
    print("Usage: python -m scripts.retrieve_probe 'your query'")
    raise SystemExit

q = sys.argv[1]
vs = Chroma(collection_name="recipes",
            embedding_function=get_embedder(),
            persist_directory=str(DEFAULT_VS_DIR))
hits = vs.similarity_search_with_score(q, k=5)
for i, (doc, dist) in enumerate(hits, 1):
    prev = (getattr(doc, "page_content", "") or "")[:160].replace("\n"," ")
    print(f"{i}. distance={float(dist):.3f} :: {prev} …")

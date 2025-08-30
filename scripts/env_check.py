# debug/env_check.py
import os, sys
print("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))
print("EMBED_BACKEND:", os.getenv("EMBED_BACKEND","openai"))
print("CHEF_EMBED_MODEL:", os.getenv("CHEF_EMBED_MODEL","text-embedding-3-small"))
try:
    if os.getenv("EMBED_BACKEND","openai").lower() == "local":
        from langchain_community.embeddings import SentenceTransformerEmbeddings as E
        emb = E(model_name=os.getenv("LOCAL_EMBED_MODEL","all-MiniLM-L6-v2"))
    else:
        from langchain_openai import OpenAIEmbeddings as E
        emb = E(model=os.getenv("CHEF_EMBED_MODEL","text-embedding-3-small"), timeout=30, max_retries=1)
    vec = emb.embed_query("hello world")
    print("Embedding dim:", len(vec), "first 3:", vec[:3])
    print("OK ✅")
except Exception as e:
    print("Embedding failed ❌:", e)
    sys.exit(1)

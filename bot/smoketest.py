print("[smoketest] importing…")
from bot.data import load_all_docs, build_or_load_vectorstore
from bot.tools import bind_vectorstore
print("[smoketest] building docs…")
docs = load_all_docs()
print("[smoketest] docs:", len(docs))
vs = build_or_load_vectorstore(docs, rebuild=True)
bind_vectorstore(vs)
print("[smoketest] OK")

# scripts/build_popularity.py (optional helper)
from pathlib import Path
from bot.scrapers.popular_country import scrape_popular_country
from bot.ingest import normalize_popular_country, merge_docs
from bot.data import load_all_docs, build_or_load_vectorstore, DEFAULT_VS_DIR

if __name__ == "__main__":
    out = scrape_popular_country(Path("data/popularity"))
    pop_docs = normalize_popular_country(out)
    base = load_all_docs()
    docs = merge_docs(base, pop_docs)
    build_or_load_vectorstore(docs, rebuild=True)
    print(f"Indexed {len(docs)} docs (with popularity).")

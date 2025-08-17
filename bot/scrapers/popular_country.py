# bot/scrapers/popular_country.py
from pathlib import Path
from typing import List, Dict
from .common import fetch_readable, extract_candidates_from_list, write_jsonl

SEEDS = {
    # country -> list URLs that list “popular dishes”
    "philippines": [
        "https://en.wikipedia.org/wiki/List_of_Filipino_dishes",
        "https://www.seriouseats.com/filipino-food-guide",
    ],
    "japan": [
        "https://en.wikipedia.org/wiki/List_of_Japanese_dishes",
        "https://www.justonecookbook.com/recipes/",
    ],
    "italy": [
        "https://en.wikipedia.org/wiki/List_of_Italian_dishes",
    ],
}

def scrape_popular_country(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "popular_country.jsonl"
    rows: List[Dict] = []
    for country, urls in SEEDS.items():
        seen=set()
        for u in urls:
            text = fetch_readable(u)
            dishes = extract_candidates_from_list(text)
            rank = 0
            for d in dishes:
                k = (country, d.lower())
                if k in seen: continue
                seen.add(k)
                rank += 1
                rows.append({
                    "country": country,
                    "dish": d,
                    "source_url": u,
                    "rank_in_list": rank
                })
    write_jsonl(out_path, rows)
    return out_path

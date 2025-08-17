# bot/ingest.py
from __future__ import annotations
import json, math
from pathlib import Path
from typing import Dict, List
from .data import RecipeDoc
from .taxonomy import continent_for_country, infer_course_from_title, infer_dish_type_from_title

def normalize_popular_country(jsonl_path: Path) -> List[RecipeDoc]:
    docs: List[RecipeDoc] = []
    seen=set()

    for ln in jsonl_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(ln)
        country = (row.get("country") or "").lower()
        dish = row.get("dish") or ""
        if not dish: continue
        key=(country, dish.lower())
        if key in seen: continue
        seen.add(key)

        cont = continent_for_country(country)
        rank = row.get("rank_in_list") or 999
        # popularity: higher if ranked near top; naive 1/log(rank+1)
        pop = 1.0 / math.log(rank + 1.5)

        docs.append(RecipeDoc(
            id=f"pop:{country}:{dish.lower()}",
            title=dish,
            url=row.get("source_url") or "",
            source="web",
            image_url=None,
            ingredients=[], steps=[],
            country=country,
            continent=cont,
            cuisine=country.capitalize() if country else None,
            dish_type=infer_dish_type_from_title(dish),
            course=infer_course_from_title(dish),
            popularity_score=round(min(pop, 1.0), 4),
            signals={"rank_in_list": rank, "source_url": row.get("source_url")},
        ))
    return docs

def merge_docs(base_docs: List[RecipeDoc], *more_lists: List[RecipeDoc]) -> List[RecipeDoc]:
    by_id = {d.id: d for d in base_docs}
    for lst in more_lists:
        for d in lst:
            if d.id not in by_id:
                by_id[d.id] = d
    return list(by_id.values())

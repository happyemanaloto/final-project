# bot/taxonomy.py
from __future__ import annotations
import re

CONTINENT_BY_COUNTRY = {
    "philippines":"asia","japan":"asia","china":"asia","india":"asia","thailand":"asia","vietnam":"asia",
    "italy":"europe","spain":"europe","france":"europe","germany":"europe","greece":"europe","turkiye":"europe",
    "mexico":"north america","united states":"north america","usa":"north america","canada":"north america",
    "peru":"south america","brazil":"south america","argentina":"south america","chile":"south america",
    "morocco":"africa","egypt":"africa","ethiopia":"africa","nigeria":"africa",
}

# fuzzy keyword maps (you can extend any time)
COUNTRY_CUES = {
    "Filipino": "philippines", "Pinoy": "philippines", "adobo": "philippines", "sinigang":"philippines",
    "Italian": "italy", "pasta":"italy", "carbonara":"italy", "risotto":"italy",
    "Japanese":"japan","ramen":"japan","sushi":"japan","tempura":"japan",
    "Mexican":"mexico","taco":"mexico","pozole":"mexico","mole":"mexico",
    "Peruvian":"peru","ceviche":"peru","lomo saltado":"peru",
}

COURSE_CUES = {"breakfast","appetizer","snack","main","dessert","soup","stew"}
DISH_TYPES = {"soup","stew","noodles","stir-fry","rice","grill","roast","salad","bread","pastry"}

def infer_country_from_text(title: str, cuisine: str | None) -> str | None:
    blob = f"{title} {cuisine or ''}".lower()
    for k, v in COUNTRY_CUES.items():
        if k.lower() in blob:
            return v
    # try "X cuisine" pattern
    m = re.search(r"\b(\w+)\s+cuisine\b", blob)
    if m:
        cand = m.group(1)
        return COUNTRY_CUES.get(cand.capitalize())
    return None

def continent_for_country(country: str | None) -> str | None:
    if not country: return None
    return CONTINENT_BY_COUNTRY.get(country.lower())

def infer_course_from_title(title: str) -> str | None:
    t = title.lower()
    for c in COURSE_CUES:
        if c in t: return "soup" if c=="soup" else c
    return None

def infer_dish_type_from_title(title: str) -> str | None:
    t = title.lower()
    for d in DISH_TYPES:
        if d in t: return d
    return None

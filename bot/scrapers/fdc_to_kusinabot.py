#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USDA FDC → KusinaBot nutrition with fuzzy lookup, synonyms, and persistent cache.

Modes:
  1) Build USDA tables (run once):
     python fdc_lookup_with_fuzzy_cached.py --build FoundationFoods.json

  2) Query a single ingredient line:
     python fdc_lookup_with_fuzzy_cached.py --query "1 tbsp olive oil"

  3) Score all recipes in a JSONL (each line has ingredients[]):
     python fdc_lookup_with_fuzzy_cached.py --from-recipes recipes.jsonl

  4) Warm the cache from a recipes JSONL (no printing, just caching):
     python fdc_lookup_with_fuzzy_cached.py --warm-cache recipes.jsonl
"""

import sys, re, json, math, argparse, sqlite3, hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
import ijson
from rapidfuzz import process, fuzz

CACHE_PATH = Path("kb_nutri_cache.sqlite")

# ------------------- Core nutrient map (FDC "number") -------------------
NUTRIENTS = {
    "208": ("kcal", "kcal"),
    "203": ("protein_g", "g"),
    "204": ("fat_g", "g"),
    "205": ("carb_g", "g"),
    "291": ("fiber_g", "g"),
    "269.3": ("sugars_g", "g"),
    "307": ("sodium_mg", "mg"),
}
FALLBACK_NUMBERS = {"1005": "carb_g", "1050": "carb_g"}

# ------------------- Synonyms / Normalization -------------------
SYNONYMS = {
    "calamansi": "lime", "kalamansi": "lime",
    "moringa": "moringa leaves", "malunggay": "moringa leaves",
    "spring onion": "green onion", "scallion": "green onion",
    "aubergine": "eggplant", "courgette": "zucchini",
    "caster sugar": "granulated sugar", "white sugar": "granulated sugar",
    "cane sugar": "granulated sugar", "muscovado sugar": "brown sugar",
    "cooking oil": "vegetable oil", "patis": "fish sauce",
    "toyomansi": "soy sauce", "all purpose flour": "wheat flour",
    "plain flour": "wheat flour", "lady finger": "okra",
}

UNIT_SYNONYMS = {
    "tsps": "tsp", "teaspoons": "tsp", "teaspoon": "tsp",
    "tbsps": "tbsp", "tablespoons": "tbsp", "tablespoon": "tbsp",
    "cups": "cup", "cup(s)": "cup",
    "grams": "g", "gram": "g", "gms": "g",
    "kilograms": "kg", "kilogram": "kg",
    "milliliters": "ml", "millilitre": "ml", "millilitres": "ml",
    "liters": "l", "litres": "l", "liter": "l",
    "cloves": "clove", "pieces": "piece", "pcs": "piece", "pc": "piece"
}

FRACTIONS = {"½":0.5,"¼":0.25,"¾":0.75,"⅓":1/3,"⅔":2/3,"⅛":0.125,"⅜":0.375,"⅝":0.625,"⅞":0.875}

UNIT_TO_GRAMS = {
    "tsp": 5.0, "tbsp": 15.0, "cup": 240.0,
    "ml": 1.0, "l": 1000.0, "g": 1.0, "kg": 1000.0,
    "clove": 3.0, "piece": 50.0, "slice": 30.0, "pinch": 0.36,
}

PIECE_OVERRIDES = [
    (re.compile(r"\begg\b"), 50.0),
    (re.compile(r"\bgarlic\b.*\bclove\b"), 3.0),
    (re.compile(r"\bginger\b"), 5.0),
]

QTY_PAT = re.compile(r"(?P<qty>(?:\d+/\d+|\d+(?:\.\d+)?|[{}])(?:\s+\d+/\d+)?)\s*".format("".join(FRACTIONS.keys())))
UNIT_PAT = re.compile(r"(?P<unit>tsp|teaspoon[s]?|tbsp|tablespoon[s]?|cup[s]?|ml|l|g|kg|clove[s]?|piece[s]?|slice[s]?|pinch(?:es)?|oz|ounce[s]?|lb|pound[s]?)", re.I)

# ------------------- Utils -------------------
def norm_name(s: str) -> str:
    s = s or ""
    s = s.strip().lower()
    s = re.sub(r"[,\(\)\[\]{}]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def apply_synonyms(name: str) -> str:
    n = norm_name(name)
    words = n.split()
    out = []
    i = 0
    while i < len(words):
        if i + 1 < len(words):
            bi = f"{words[i]} {words[i+1]}"
            if bi in SYNONYMS:
                out.append(SYNONYMS[bi]); i += 2; continue
        w = words[i]
        out.append(SYNONYMS.get(w, w)); i += 1
    return re.sub(r"\s+", " ", " ".join(out)).strip()

# ------------------- Build USDA tables (streaming) -------------------
def parse_foundation_json(path: Path):
    with open(path, "rb") as f:
        foods_iter = ijson.items(f, "FoundationFoods.item")
        for food in foods_iter:
            fdc_id = food.get("fdcId")
            desc = (food.get("description") or "").strip()
            category = (food.get("foodCategory", {}) or {}).get("description")

            per100 = {v[0]: None for v in NUTRIENTS.values()}

            for fn in food.get("foodNutrients", []) or []:
                nut = fn.get("nutrient", {})
                number = str(nut.get("number", "")).strip()
                amount = fn.get("amount")
                if amount is None: 
                    continue
                if number in NUTRIENTS:
                    key, _u = NUTRIENTS[number]
                    per100[key] = float(amount)
                elif number in FALLBACK_NUMBERS and per100.get("carb_g") is None:
                    per100["carb_g"] = float(amount)

            if per100.get("kcal") is None:
                for fn in food.get("foodNutrients", []) or []:
                    nut = fn.get("nutrient", {})
                    if str(nut.get("number", "")).strip() == "268" and fn.get("amount") is not None:
                        per100["kcal"] = float(fn["amount"]) / 4.184
                        break

            portions = []
            for p in food.get("foodPortions", []) or []:
                gw = p.get("gramWeight")
                if not gw or gw <= 0: continue
                mu = (p.get("measureUnit") or {}).get("name")
                amount = p.get("amount")
                modifier = p.get("modifier") or ""
                label = f"{amount or 1} {mu}".strip() if mu else f"{amount or 1} portion"
                if modifier: label = f"{label} {modifier}".strip()
                kcal_pp = per100["kcal"] * (float(gw)/100.0) if per100.get("kcal") is not None else None
                portions.append({
                    "label": label, "gram_weight": float(gw),
                    "amount": amount, "measure_unit": (mu or "").lower(),
                    "modifier": modifier.lower(),
                    "kcal_per_portion": round(kcal_pp, 2) if kcal_pp is not None else None
                })

            yield {
                "fdcId": fdc_id, "description": desc, "category": category,
                "per100g": per100, "portions": portions
            }

def build_frames(stream_iter):
    rows100, rowsPort, rowsLookup = [], [], []
    for item in stream_iter:
        fdc = item["fdcId"]; name = item["description"]; cat = item["category"]; p100 = item["per100g"]
        rows100.append({
            "fdcId": fdc, "name": name, "name_norm": norm_name(name), "category": cat,
            "kcal_per_100g": p100.get("kcal"), "protein_g_per_100g": p100.get("protein_g"),
            "fat_g_per_100g": p100.get("fat_g"), "carb_g_per_100g": p100.get("carb_g"),
            "fiber_g_per_100g": p100.get("fiber_g"), "sugars_g_per_100g": p100.get("sugars_g"),
            "sodium_mg_per_100g": p100.get("sodium_mg"),
        })
        for pr in item["portions"]:
            rowsPort.append({
                "fdcId": fdc, "name": name, "name_norm": norm_name(name), "label": pr["label"],
                "gram_weight": pr["gram_weight"], "kcal_per_portion": pr["kcal_per_portion"],
                "measure_unit": pr["measure_unit"], "amount": pr["amount"], "modifier": pr["modifier"],
                "category": cat,
            })
        rowsLookup.append({"name_norm": norm_name(name), "fdcId": fdc, "kcal_per_100g": p100.get("kcal")})
    df100 = pd.DataFrame(rows100).drop_duplicates(subset=["fdcId"])
    dfPort = pd.DataFrame(rowsPort)
    dfLk = pd.DataFrame(rowsLookup).drop_duplicates(subset=["name_norm"])
    return df100, dfPort, dfLk

# ------------------- DB Cache -------------------
def cache_init():
    con = sqlite3.connect(CACHE_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS line_cache (
        key TEXT PRIMARY KEY,
        result_json TEXT NOT NULL
    );""")
    con.execute("""CREATE TABLE IF NOT EXISTS match_cache (
        target_norm TEXT PRIMARY KEY,
        fdcId INTEGER,
        matched_name TEXT,
        score REAL
    );""")
    con.commit()
    return con

def cache_key_for_line(line: str) -> str:
    # include versioning hook if you edit logic later
    h = hashlib.sha256(("v1|" + line.strip().lower()).encode("utf-8")).hexdigest()
    return h

def cache_get_line(con, line: str) -> Optional[Dict[str,Any]]:
    key = cache_key_for_line(line)
    cur = con.execute("SELECT result_json FROM line_cache WHERE key=?", (key,))
    row = cur.fetchone()
    if row:
        try:
            return json.loads(row[0])
        except Exception:
            return None
    return None

def cache_put_line(con, line: str, result: Dict[str,Any]):
    key = cache_key_for_line(line)
    con.execute("REPLACE INTO line_cache(key, result_json) VALUES(?,?)", (key, json.dumps(result, ensure_ascii=False)))
    con.commit()

def cache_get_match(con, target_norm: str):
    cur = con.execute("SELECT fdcId, matched_name, score FROM match_cache WHERE target_norm=?", (target_norm,))
    row = cur.fetchone()
    if row:
        return int(row[0]) if row[0] is not None else None, row[1], float(row[2])
    return None, None, 0.0

def cache_put_match(con, target_norm: str, fdcId: Optional[int], matched_name: Optional[str], score: float):
    con.execute("REPLACE INTO match_cache(target_norm, fdcId, matched_name, score) VALUES(?,?,?,?)",
                (target_norm, fdcId, matched_name, score))
    con.commit()

# ------------------- Qty/Unit parsing -------------------
def parse_qty_unit(text: str):
    s = text.strip()
    qty = 1.0; unit = None
    m = QTY_PAT.match(s)
    if m:
        raw = m.group("qty"); parts = raw.split(); total = 0.0
        for p in parts:
            if p in FRACTIONS: total += FRACTIONS[p]
            elif "/" in p: a,b = p.split("/",1); total += float(a)/float(b)
            else:
                try: total += float(p)
                except: pass
        qty = total if total>0 else 1.0
        s = s[m.end():].lstrip()
    m2 = UNIT_PAT.match(s)
    if m2:
        unit_raw = m2.group("unit").lower()
        unit = UNIT_SYNONYMS.get(unit_raw, unit_raw)
        s = s[m2.end():].lstrip()
    name = s
    return qty, unit, name

# ------------------- Portion grams helpers -------------------
def pick_portion_grams(unit: Optional[str], qty: float, dfPort: pd.DataFrame, fdc_id: Optional[int]) -> Optional[float]:
    if fdc_id is None or dfPort.empty or unit is None: return None
    sub = dfPort[dfPort["fdcId"] == fdc_id]
    if sub.empty: return None
    u = unit.lower()
    mu = sub[sub["measure_unit"] == u]
    if not mu.empty:
        amt = mu.iloc[0]["amount"] or 1
        return float(mu.iloc[0]["gram_weight"]) * (qty / float(amt))
    for _, r in sub.iterrows():
        if u in (r["label"] or ""):
            amt = r["amount"] or 1
            return float(r["gram_weight"]) * (qty / float(amt))
    return None

def fallback_grams(unit: Optional[str], qty: float, item_name: str) -> Optional[float]:
    if unit is None:
        for rx, g in PIECE_OVERRIDES:
            if rx.search(item_name.lower()): return g * qty
        return None
    u = unit.lower()
    if u == "piece":
        for rx, g in PIECE_OVERRIDES:
            if rx.search(item_name.lower()): return g * qty
    if u in UNIT_TO_GRAMS: return UNIT_TO_GRAMS[u] * qty
    if u in {"oz","ounce","ounces"}: return 28.35 * qty
    if u in {"lb","pound","pounds"}: return 453.6 * qty
    return None

# ------------------- Fuzzy match (with cache) -------------------
def fuzzy_match(name_raw: str, df100: pd.DataFrame, con) -> Tuple[Optional[int], Optional[str], float]:
    target = apply_synonyms(name_raw)
    target_norm = norm_name(target)

    # cache check
    fdc_id, matched_name, score = cache_get_match(con, target_norm)
    if matched_name is not None or fdc_id is not None:
        return fdc_id, matched_name, score

    choices = df100["name_norm"].tolist()
    if not choices: return None, None, 0.0
    best, score, idx = process.extractOne(target_norm, choices, scorer=fuzz.WRatio, score_cutoff=78)
    if best is None:
        cache_put_match(con, target_norm, None, None, 0.0)
        return None, None, 0.0
    row = df100.iloc[idx]
    fdc_id = int(row["fdcId"]); matched_name = row["name"]
    cache_put_match(con, target_norm, fdc_id, matched_name, float(score))
    return fdc_id, matched_name, float(score)

# ------------------- Compute nutrition (with cache) -------------------
def compute_line_nutrition(line: str, df100: pd.DataFrame, dfPort: pd.DataFrame, con) -> Dict[str, Any]:
    # Check cache first
    cached = cache_get_line(con, line)
    if cached: return cached

    qty, unit, name = parse_qty_unit(line)
    fdc_id, matched_name, score = fuzzy_match(name, df100, con)

    grams = None
    grams = pick_portion_grams(unit, qty, dfPort, fdc_id)
    if grams is None: grams = fallback_grams(unit, qty, matched_name or name)

    out = {
        "input": line, "parsed_qty": qty, "parsed_unit": unit,
        "name_after_parse": name.strip(),
        "matched_fdcId": fdc_id, "matched_name": matched_name, "match_score": score,
        "grams": grams,
        "kcal": None, "protein_g": None, "fat_g": None, "carb_g": None,
        "fiber_g": None, "sugars_g": None, "sodium_mg": None
    }

    if fdc_id is None:
        cache_put_line(con, line, out); return out

    row = df100.loc[df100["fdcId"] == fdc_id]
    if row.empty:
        cache_put_line(con, line, out); return out

    per100 = {
        "kcal": row.iloc[0]["kcal_per_100g"],
        "protein_g": row.iloc[0]["protein_g_per_100g"],
        "fat_g": row.iloc[0]["fat_g_per_100g"],
        "carb_g": row.iloc[0]["carb_g_per_100g"],
        "fiber_g": row.iloc[0]["fiber_g_per_100g"],
        "sugars_g": row.iloc[0]["sugars_g_per_100g"],
        "sodium_mg": row.iloc[0]["sodium_mg_per_100g"],
    }

    if grams is not None:
        scale = grams / 100.0
        out.update({
            "kcal": round((per100["kcal"] or 0)*scale, 2) if per100["kcal"] is not None else None,
            "protein_g": round((per100["protein_g"] or 0)*scale, 3) if per100["protein_g"] is not None else None,
            "fat_g": round((per100["fat_g"] or 0)*scale, 3) if per100["fat_g"] is not None else None,
            "carb_g": round((per100["carb_g"] or 0)*scale, 3) if per100["carb_g"] is not None else None,
            "fiber_g": round((per100["fiber_g"] or 0)*scale, 3) if per100["fiber_g"] is not None else None,
            "sugars_g": round((per100["sugars_g"] or 0)*scale, 3) if per100["sugars_g"] is not None else None,
            "sodium_mg": round((per100["sodium_mg"] or 0)*scale, 1) if per100["sodium_mg"] is not None else None,
        })

    cache_put_line(con, line, out)
    return out

# ------------------- I/O helpers -------------------
def ensure_tables():
    p100_pq = Path("fdc_per_100g.parquet")
    port_pq = Path("fdc_portions.parquet")
    if not (p100_pq.exists() and port_pq.exists()):
        print("[ERROR] Missing fdc_per_100g.parquet / fdc_portions.parquet. Run --build first.")
        sys.exit(2)
    return pd.read_parquet(p100_pq), pd.read_parquet(port_pq)

def build_from_json(json_path: Path):
    stream = parse_foundation_json(json_path)
    df100, dfPort, _ = build_frames(stream)
    df100.to_csv("fdc_per_100g.csv", index=False)
    dfPort.to_csv("fdc_portions.csv", index=False)
    df100.to_parquet("fdc_per_100g.parquet", index=False)
    dfPort.to_parquet("fdc_portions.parquet", index=False)
    with open("fdc_lookup.jsonl", "w", encoding="utf-8") as f:
        for _, row in df100.iterrows():
            f.write(json.dumps({
                "fdcId": int(row["fdcId"]),
                "name": row["name"],
                "name_norm": row["name_norm"],
                "kcal_per_100g": row["kcal_per_100g"]
            }, ensure_ascii=False) + "\n")
    print("[DONE] Built nutrition tables.")

def score_recipe_file(recipes_jsonl: Path, quiet=False):
    df100, dfPort = ensure_tables()
    con = cache_init()
    total_recipes = 0
    for line in open(recipes_jsonl, "r", encoding="utf-8"):
        rec = json.loads(line)
        title = rec.get("title") or "(untitled)"
        items = rec.get("ingredients") or []
        agg = {"kcal":0.0,"protein_g":0.0,"fat_g":0.0,"carb_g":0.0,"fiber_g":0.0,"sugars_g":0.0,"sodium_mg":0.0}
        details = []
        for ing in items:
            res = compute_line_nutrition(ing, df100, dfPort, con)
            details.append(res)
            for k in agg.keys():
                if res.get(k) is not None:
                    agg[k] += float(res[k])
        if not quiet:
            print(f"\n=== {title} ===")
            print("Ingredients scored:")
            for d in details:
                g = f"{d['grams']:.0f} g" if d.get("grams") else "n/a"
                kc = f"{d['kcal']:.0f} kcal" if d.get("kcal") else "n/a"
                print(f" - {d['input']}  →  {d.get('matched_name') or 'no match'}  [{g}, {kc}] (score {int(d['match_score'])})")
            print("Totals (approx):",
                  f"{agg['kcal']:.0f} kcal; P {agg['protein_g']:.1f} g / F {agg['fat_g']:.1f} g / C {agg['carb_g']:.1f} g")
        total_recipes += 1
    if not quiet:
        print(f"\n[DONE] Scored {total_recipes} recipe(s).")

def warm_cache_from_recipes(recipes_jsonl: Path):
    print(f"[INFO] Warming cache from {recipes_jsonl} ...")
    df100, dfPort = ensure_tables()
    con = cache_init()
    c = 0
    for line in open(recipes_jsonl, "r", encoding="utf-8"):
        rec = json.loads(line)
        for ing in rec.get("ingredients") or []:
            compute_line_nutrition(ing, df100, dfPort, con)
            c += 1
    print(f"[DONE] Cached {c} ingredient lines into {CACHE_PATH.name}")

# ------------------- CLI -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=str, help="Path to USDA FoundationFoods JSON to build tables")
    ap.add_argument("--query", type=str, help="Ingredient line to parse & score (e.g., '1 tbsp olive oil')")
    ap.add_argument("--from-recipes", type=str, help="Path to recipes JSONL (each line has ingredients[])")
    ap.add_argument("--warm-cache", type=str, help="Precompute and cache results from a recipes JSONL")
    args = ap.parse_args()

    if args.build:
        build_from_json(Path(args.build)); return
    if args.query:
        df100, dfPort = ensure_tables()
        con = cache_init()
        res = compute_line_nutrition(args.query, df100, dfPort, con)
        print(json.dumps(res, ensure_ascii=False, indent=2)); return
    if args.from_recipes:
        score_recipe_file(Path(args.from_recipes)); return
    if args.warm_cache:
        warm_cache_from_recipes(Path(args.warm_cache)); return

    print("Nothing to do. Use --build, --query, --from-recipes, or --warm-cache.")

if __name__ == "__main__":
    main()

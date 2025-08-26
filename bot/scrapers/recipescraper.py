#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse recipe .txt files that use headings like:
    Recipe 1: Crispy Squash Fries With Tahini-Yogurt Dip
    ...
    Ingredients
    ...
    Directions
    ...

Usage:
    python txt_recipe_extractor_v2.py <file-or-folder>

Outputs (in current working directory):
    recipes_extracted_<timestamp>.jsonl
    recipes_extracted_<timestamp>.csv
"""

import sys, re, json, csv
from pathlib import Path
from datetime import datetime

# --- Headings we recognize (case-insensitive) ---
HEAD_ING   = {"ingredients", "ingredient", "you’ll need", "you will need"}
HEAD_DIR   = {"directions", "instructions", "method", "preparation", "steps"}
HEAD_VAR   = {"variations", "variation"}
HEAD_SERVE = {"how to serve"}
HEAD_NOTE  = {"notes", "tips", "storing and reheating", "storage"}

META_KEYS  = {"prep time", "cook time", "total time", "active time", "servings", "yield"}
META_PUB   = {"published on", "updated on"}
META_AUTH  = {"by "}  # lines starting with "By "

BULLET_PREFIX = r"(?:^[\u2022\u2023\u25E6\u2043\u2219•·●▪\-–—]+\s*|^\d+\.\s*)"

# Recipe block header, e.g. "Recipe 1: Title" or "Recipe 12 : Title"
RE_RECIPE_HEADER = re.compile(r"^recipe\s+(\d+)\s*:\s*(.+)$", re.I)

def is_heading(line: str, group: set) -> bool:
    return line.strip().lower() in group

def strip_bullet(line: str) -> str:
    return re.sub(BULLET_PREFIX, "", line).strip()

def normalize(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())

def parse_one_block(block_lines):
    """
    block_lines: list of lines belonging to one recipe (title already known)
    Returns dict or None if not valid.
    """
    if not block_lines:
        return None

    # First line must be "Recipe N: Title"
    m = RE_RECIPE_HEADER.match(block_lines[0])
    if not m:
        return None

    idx = int(m.group(1))
    title = m.group(2).strip()

    recipe = {
        "recipe_index": idx,
        "title": title,
        "author": None,
        "published_on": None,
        "prep_time": None,
        "cook_time": None,
        "total_time": None,
        "active_time": None,
        "servings": None,
        "yield": None,
        "ingredients": [],
        "steps": [],
        "variations": [],
        "how_to_serve": [],
        "notes": [],
    }

    section = None

    # Iterate remaining lines
    for raw in block_lines[1:]:
        if not raw.strip():
            continue

        line = raw.strip()
        low  = line.lower()

        # --- meta: author/published ---
        if any(low.startswith(k) for k in META_PUB):
            # "Published on December 16, 2024"
            recipe["published_on"] = line.split(":", 1)[-1].strip() if ":" in line else line
            continue

        if low.startswith("by "):  # author
            recipe["author"] = line[3:].strip()
            continue

        # --- time/servings meta ---
        for key in META_KEYS:
            if low.startswith(key):
                # value may be on same line (after colon) or next line; we assume same block places it next line or same line
                val = line.split(":", 1)[-1].strip() if ":" in line else ""
                if not val:
                    # value might be on next non-empty line; handled implicitly since the next line won't look like a heading
                    pass
                field = key.replace(" ", "_")
                if not val:
                    # leave None; the next line may just be a standalone "25 mins" which will be caught below
                    pass
                else:
                    recipe[field] = val
                # continue parsing; don't 'continue' so we can also catch single-line values below
                break
        else:
            # Single-line values appearing right after a meta key (e.g., a line that is "25 mins")
            if section is None and re.fullmatch(r"\d+\s*(?:m|min|mins|minute|minutes|h|hr|hrs|hour|hours)(?:\s*\d*\s*(?:m|min|mins))?", low):
                # Try to assign to the first None of [prep, cook, total] in a sensible order
                for field in ("prep_time", "cook_time", "total_time"):
                    if recipe.get(field) in (None, ""):
                        recipe[field] = line
                        break

        # --- section changes ---
        if is_heading(line, HEAD_ING):
            section = "ingredients";  continue
        if is_heading(line, HEAD_DIR):
            section = "steps";        continue
        if is_heading(line, HEAD_VAR):
            section = "variations";   continue
        if is_heading(line, HEAD_SERVE):
            section = "how_to_serve"; continue
        if is_heading(line, HEAD_NOTE):
            section = "notes";        continue

        # --- content by section ---
        if section == "ingredients":
            recipe["ingredients"].append(strip_bullet(line))
            continue
        if section == "steps":
            recipe["steps"].append(strip_bullet(line))
            continue
        if section == "variations":
            recipe["variations"].append(strip_bullet(line))
            continue
        if section == "how_to_serve":
            recipe["how_to_serve"].append(strip_bullet(line))
            continue
        if section == "notes":
            recipe["notes"].append(strip_bullet(line))
            continue

        # If we aren't in a section, ignore descriptive lines (subtitles, taglines)
        # or keep them as notes if useful
        # (no-op)

    # --- minimal validity gate: must have real content ---
    if len([i for i in recipe["ingredients"] if i]) < 3:
        return None
    if len([s for s in recipe["steps"] if s]) < 2:
        return None

    return recipe

def split_into_blocks(lines):
    """
    Split the entire file into blocks starting with 'Recipe N:'.
    """
    blocks = []
    current = []
    for ln in lines:
        if RE_RECIPE_HEADER.match(ln.strip()):
            if current:
                blocks.append(current)
            current = [ln]
        else:
            if current:
                current.append(ln)
    if current:
        blocks.append(current)
    return blocks

def parse_txt_file(path: Path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [l.rstrip("\n") for l in f]
    blocks = split_into_blocks(lines)
    out = []
    for b in blocks:
        r = parse_one_block(b)
        if r:
            r["source_file"] = str(path)
            out.append(r)
    return out

def main():
    if len(sys.argv) < 2:
        print("Usage: python txt_recipe_extractor_v2.py <file-or-folder>")
        sys.exit(1)

    target = Path(sys.argv[1])
    files = []
    if target.is_file():
        if target.suffix.lower() == ".txt":
            files = [target]
        else:
            print(f"[WARN] Not a .txt file: {target}")
    else:
        files = list(target.rglob("*.txt"))

    if not files:
        print("[ERROR] No .txt files found.")
        sys.exit(2)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_jsonl = Path.cwd() / f"recipes_extracted_{ts}.jsonl"
    out_csv   = Path.cwd() / f"recipes_extracted_{ts}.csv"

    total = 0
    with open(out_jsonl, "w", encoding="utf-8") as jf, open(out_csv, "w", encoding="utf-8", newline="") as cf:
        fieldnames = [
            "recipe_index","title","author","published_on","prep_time","cook_time","total_time","active_time",
            "servings","yield","ingredients","steps","variations","how_to_serve","notes","source_file"
        ]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()

        for f in files:
            recs = parse_txt_file(f)
            print(f"[OK] {f.name}: {len(recs)} recipe(s)")
            for r in recs:
                jf.write(json.dumps(r, ensure_ascii=False) + "\n")
                row = r.copy()
                # stringify lists for CSV
                for k in ["ingredients","steps","variations","how_to_serve","notes"]:
                    row[k] = " | ".join(row.get(k, []))
                writer.writerow(row)
                total += 1

    print(f"\n[DONE] Wrote {total} recipe(s)")
    print(f" JSONL: {out_jsonl}")
    print(f" CSV  : {out_csv}")

if __name__ == "__main__":
    main()

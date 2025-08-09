"""
Wikibooks Cookbook: Table of Contents scraper
- Starts from https://en.wikibooks.org/wiki/Cookbook:Table_of_Contents
- Discovers recipe pages (Cookbook:...)
- Fetches each page via MediaWiki API (safe & stable)
- Extracts ingredients + steps using heuristics
- Saves per-recipe JSON and a master JSONL (and optional CSV)

License note:
Wikibooks content is CC BY-SA 3.0. This script stores:
  - license metadata
  - attribution (title + source URL)
You MUST provide attribution and share-alike for any redistribution or derivatives.

Usage:
  pip install requests beautifulsoup4 lxml
  python cookbook_toc_scraper.py --max 50 --delay 0.5 --resume --export-csv
"""

from __future__ import annotations
import argparse, csv, json, re, time
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse, quote

import requests
from bs4 import BeautifulSoup

WB_ROOT = "https://en.wikibooks.org"
TOC_URL_DEFAULT = "https://en.wikibooks.org/wiki/Cookbook:Table_of_Contents"
WB_API = "https://en.wikibooks.org/w/api.php"


# ---------- tiny utils ----------
def log(msg: str) -> None:
    print(msg, flush=True)

def safe_slug(text: str, max_len: int = 120) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", (text or "")).strip("_")[:max_len]

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "KusinaBot/1.0 (educational, non-commercial)"})
    s.timeout = 25
    return s


# ---------- TOC discovery ----------
def discover_cookbook_links_from_toc(session: requests.Session, toc_url: str) -> List[str]:
    """
    Returns a list of page titles like 'Cookbook:Adobo' discovered on the TOC page.
    We intentionally collect *titles* and use the API to parse pages later.
    """
    log(f"Fetching TOC: {toc_url}")
    r = session.get(toc_url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    titles: List[str] = []
    for a in soup.select("div.mw-parser-output a[href]"):
        href = a.get("href") or ""
        if not href.startswith("/wiki/"):
            continue
        # Keep Cookbook:... pages only; skip talk/category/index pages
        path = href.split("#", 1)[0]  # drop anchors
        title = path[len("/wiki/"):]
        # Accept main namespace "Cookbook:..."
        if not title.startswith("Cookbook:"):
            continue
        # Skip non-recipe namespaces under Cookbook (e.g., "Cookbook talk:")
        if title.lower().startswith(("cookbook_talk:", "cookbook:table_of_contents".lower())):
            continue
        # De-dup
        if title not in titles:
            titles.append(title)
    log(f"Discovered {len(titles)} Cookbook pages from TOC")
    return titles


# ---------- Parse one page via API ----------
def wb_parse_page(session: requests.Session, title: str) -> Dict[str, Any]:
    """
    Parse a Wikibooks 'Cookbook:...' page into a structured recipe.
    Returns a dict with license + attribution and a 'recipe' object for your bot.
    """
    params = {
        "action": "parse",
        "page": title,
        "prop": "text|sections|displaytitle",
        "format": "json",
    }
    r = session.get(WB_API, params=params)
    r.raise_for_status()
    data = r.json()

    if "error" in data:
        raise RuntimeError(f"API error for {title}: {data['error']}")

    html = data.get("parse", {}).get("text", {}).get("*", "")
    display_title = data.get("parse", {}).get("displaytitle") or title
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("div.mw-parser-output")

    # Heuristics to extract ingredients and steps
    def collect_after_heading(names: List[str]) -> List[str]:
        out: List[str] = []
        if not content:
            return out
        headings = content.find_all(["h2", "h3"])
        for h in headings:
            label = (h.get_text() or "").strip().lower()
            if any(n in label for n in names):
                cur = h.find_next_sibling()
                while cur and cur.name not in ["h2", "h3"]:
                    if cur.name in ["ul", "ol"]:
                        for li in cur.find_all("li", recursive=False):
                            txt = " ".join(li.get_text(" ", strip=True).split())
                            if txt:
                                out.append(txt)
                    cur = cur.find_next_sibling()
                break
        return out

    ingredients = collect_after_heading(["ingredient"])
    steps = collect_after_heading(["direction", "method", "preparation", "procedure", "steps"])

    # Fallbacks
    if not ingredients and content:
        first_ul = content.find("ul")
        if first_ul:
            ingredients = [" ".join(li.get_text(" ", strip=True).split())
                           for li in first_ul.find_all("li", recursive=False)]

    if not steps and content:
        paras = [p.get_text(" ", strip=True) for p in content.find_all("p", recursive=False)]
        steps = [p for p in paras if len(p.split()) > 6][:8]

    source_url = f"{WB_ROOT}/wiki/{quote(title.replace(' ', '_'))}"

    record = {
        "source": "wikibooks",
        "source_url": source_url,
        "license": "CC BY-SA 3.0",
        "attribution": {
            "project": "Wikibooks",
            "title": display_title,
            "url": source_url,
            "notice": "Text available under CC BY-SA 3.0; attribution and share-alike required.",
        },
        # Standard schema for your chatbot
        "recipe": {
            "title": display_title,
            "ingredients": ingredients,
            "steps": steps,
            "servings": None,
            "cook_time_minutes": None,
            "cuisine": None,
            "tools": None,
        },
    }
    return record


# ---------- Save helpers ----------
def save_json(record: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = safe_slug(record["recipe"]["title"]) or "recipe"
    path = out_dir / f"{slug}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return path

def append_jsonl(record: Dict[str, Any], master_path: Path) -> None:
    master_path.parent.mkdir(parents=True, exist_ok=True)
    with master_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------- Orchestrators ----------
def run_from_toc(
    toc_url: str,
    out_dir: Path,
    max_n: Optional[int],
    delay: float,
    resume: bool,
    export_csv: bool,
) -> None:
    session = make_session()
    titles = discover_cookbook_links_from_toc(session, toc_url)
    if max_n:
        titles = titles[:max_n]

    master = out_dir / "recipes.jsonl"
    seen_titles = set()
    if resume and master.exists():
        for ln in master.read_text(encoding="utf-8").splitlines():
            try:
                seen_titles.add(json.loads(ln)["attribution"]["title"])
            except Exception:
                continue

    csv_path = out_dir / "recipes.csv" if export_csv else None
    csv_writer = None
    if csv_path:
        csv_file = csv_path.open("a", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        if csv_path.stat().st_size == 0:
            csv_writer.writerow(["title", "url", "n_ingredients", "n_steps"])

    saved = 0
    for i, title in enumerate(titles, 1):
        log(f"[{i}/{len(titles)}] {title}")
        try:
            # Skip already saved titles
            if resume and title in seen_titles:
                log("  -> exists, skipping")
                continue

            rec = wb_parse_page(session, title)

            # Skip obvious non-recipe pages (no ingredients & no steps)
            if not rec["recipe"]["ingredients"] and not rec["recipe"]["steps"]:
                log("  !! empty recipe (no ingredients/steps); skipping")
                time.sleep(delay)
                continue

            p = save_json(rec, out_dir)
            append_jsonl(rec, master)
            if csv_writer:
                csv_writer.writerow([
                    rec["recipe"]["title"],
                    rec["attribution"]["url"],
                    len(rec["recipe"]["ingredients"]),
                    len(rec["recipe"]["steps"]),
                ])
            saved += 1
            log(f"  -> saved {p}")
        except Exception as e:
            log(f"  !! failed: {e}")
        time.sleep(delay)

    if csv_writer:
        csv_file.close()
    log(f"Done. Saved {saved} recipes to {out_dir}")


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Scrape Wikibooks Cookbook via Table of Contents")
    ap.add_argument("--toc-url", type=str, default=TOC_URL_DEFAULT,
                    help="TOC page URL (default: Cookbook:Table_of_Contents)")
    ap.add_argument("--out-dir", type=str, default="data/open_wikibooks_toc",
                    help="Output directory")
    ap.add_argument("--max", type=int, default=None, help="Max recipes to process")
    ap.add_argument("--delay", type=float, default=0.5, help="Delay (s) between requests")
    ap.add_argument("--resume", action="store_true", help="Skip recipes already written to JSONL")
    ap.add_argument("--export-csv", action="store_true", help="Also write a summary CSV")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_from_toc(
        toc_url=args.toc_url,
        out_dir=out_dir,
        max_n=args.max,
        delay=args.delay,
        resume=args.resume,
        export_csv=args.export_csv,
    )

if __name__ == "__main__":
    main()

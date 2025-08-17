# bot/scrapers/common.py
import time, json, re
from pathlib import Path
from typing import Dict, List, Optional
import trafilatura  # pip install trafilatura

def fetch_readable(url: str):
    downloaded = trafilatura.fetch_url(url) 
    if not downloaded:
        return ""
    return trafilatura.extract(downloaded, include_links=False) or ""

def extract_candidates_from_list(article_text: str) -> List[str]:
    """Very simple heuristic: pick capitalized dish names from numbered/bulleted lists."""
    out=[]
    for line in article_text.splitlines():
        line=line.strip()
        if re.match(r"^(\d+[\).]|[-•])\s+", line):
            name = re.sub(r"^(\d+[\).]|[-•])\s+","",line)
            # stop at punctuation after dish name
            name = re.split(r"[:–\-–—(]", name)[0].strip()
            if 2 <= len(name) <= 60:
                out.append(name)
    return list(dict.fromkeys(out))  # dedupe, keep order

def write_jsonl(path: Path, objs: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

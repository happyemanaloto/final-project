import os, re, tempfile, time
from pathlib import Path
from typing import Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .telemetry import init_langsmith
from langdetect import DetectorFactory, detect_langs
DetectorFactory.seed = 0

CHEF_TEMP = float(os.getenv("CHEF_TEMP", "0.5"))
LLM_MODEL = os.getenv("CHEF_BOT_MODEL", "gpt-4o-mini")

_tracer_cache = None

# Reply language control
LANG_ALIASES: Dict[str,str] = {}
def _alias(code, *names):
    for n in (code, *names): LANG_ALIASES[n.lower()] = code
_alias("en","english"); _alias("tl","tagalog","fil","filipino"); _alias("ko","korean")
_alias("es","spanish","español"); _alias("nl","dutch"); _alias("pam","kapampangan")

LANG_TO_GTTS = {"en":"en","tl":"tl","ko":"ko","es":"es","nl":"nl","pam":"tl"}
def _pick_gtts_lang(prefer: Optional[str]) -> str:
    key = (prefer or "en").lower()
    return LANG_TO_GTTS.get(key, LANG_TO_GTTS.get(key.split("-")[0], "en"))

def detect_language(text: str) -> str:
    try:
        cands = detect_langs(text)
        if text.isascii() and len(text)<40:
            for c in cands:
                if c.lang=="en" and c.prob>=0.15: return "en"
        return cands[0].lang if cands else "en"
    except Exception:
        return "en"

def parse_language_switch(user_text: str) -> Optional[str]:
    t = re.sub(r"[^\w\s\-]", "", (user_text or "").strip().lower())
    m = re.search(r"(?:^|[\s:/])(?:/lang|lang(?:uage)?|switch|reply|answer|speak|use|set|respond)\s*(?:to|in|:)?\s*([a-z][a-z0-9\- ]+)\s*$", t)
    if m:
        key = re.sub(r"\s+"," ", m.group(1).strip()); return LANG_ALIASES.get(key)
    m2 = re.search(r"^(?:in\s+)?([a-z][a-z0-9\- ]+)\s+(?:please|pls)\s*$", t)
    if m2:
        key = re.sub(r"\s+"," ", m2.group(1).strip()); return LANG_ALIASES.get(key)
    return None

_tracer_cache = None

def llm_zero(temperature: float | None = None, model: str | None = None):
    global _tracer_cache
    if _tracer_cache is None:
        _tracer_cache = init_langsmith(project=os.getenv("LANGCHAIN_PROJECT", "KusinaBot"))

    kwargs = {
        "model": model or LLM_MODEL,
        "temperature": CHEF_TEMP if temperature is None else float(temperature),
    }
    # Only attach callbacks if present
    if _tracer_cache:
        kwargs["callbacks"] = _tracer_cache
    return ChatOpenAI(**kwargs)

TRANS_PROMPT = ChatPromptTemplate.from_messages([
    ("system","Translate to English. Return translation only if input is not English."),
    ("human","{text}")
])
def translate_to_english(text: str) -> str:
    if not text: return text
    try:
        cands = detect_langs(text)
        if cands and cands[0].lang=="en" and cands[0].prob>=0.6: return text
    except Exception: pass
    out = (TRANS_PROMPT | llm_zero()).invoke({"text": text})
    return out.content.strip()

def ensure_reply_language(text: str, target_lang: Optional[str]) -> str:
    if not text or not target_lang: return text or ""
    try:
        cands = detect_langs(text)
        if cands and cands[0].lang==target_lang and cands[0].prob>=0.5: return text
    except Exception: pass
    prompt = ChatPromptTemplate.from_messages([
        ("system","Translate into the target language. Preserve bullets and formatting. Return ONLY the translated text."),
        ("human","Target language: {lang}\n\nText:\n{txt}")
    ])
    out = (prompt | llm_zero()).invoke({"lang": target_lang, "txt": text})
    return out.content.strip()

ING_TRANSLATIONS = {
    "tl": {"lime":"dayap","garlic":"bawang","onion":"sibuyas","soy sauce":"toyo"}
}
def localize_ingredients(ings: List[str], lang: Optional[str]) -> List[str]:
    mapping = ING_TRANSLATIONS.get((lang or "").lower())
    if not mapping: return ings
    import re
    keys = sorted(mapping.keys(), key=len, reverse=True)
    out=[]
    for s in ings:
        t = s
        for k in keys:
            t = re.sub(rf"(?i)\b{re.escape(k)}\b", mapping[k], t)
        out.append(t)
    return out

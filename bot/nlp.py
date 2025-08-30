# bot/nlp.py
import os, re
from typing import Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .telemetry import init_langsmith
from langdetect import DetectorFactory, detect_langs

# --- deterministic langdetect ---
DetectorFactory.seed = 0

# --- model config ---
CHEF_TEMP = float(os.getenv("CHEF_TEMP", "0.5"))
# LLM_MODEL = os.getenv("CHEF_BOT_MODEL", "gpt-4o-mini")
LLM_MODEL = os.getenv("CHEF_BOT_MODEL", "gpt‑3.5‑turbo")

_tracer_cache = None
# Cache ChatOpenAI instances by (model, temperature) to avoid re-instantiating
_LLM_CACHE: Dict[tuple, ChatOpenAI] = {}
# =========================================================
# Language tables
# NOTE: We keep your original return style: "English","Tagalog",…
# =========================================================
LANG_ALIASES: Dict[str, str] = {}
def _alias(code: str, *names: str):
    for n in (code, *names):
        LANG_ALIASES[n.strip().lower()] = code

_alias("English", "english", "eng", "en")
_alias("Tagalog", "tagalog", "fil", "filipino", "tl")
_alias("Korean", "korean", "ko")
_alias("Spanish", "spanish", "espanol", "español", "castellano", "es")
_alias("Dutch", "dutch", "nederlands", "nl")
_alias("French", "french", "francais", "français", "fr")
_alias("German", "german", "deutsch", "de")
_alias("Italian", "italian", "italiano", "it")
_alias("Portuguese", "portuguese", "portugues", "português", "pt")
_alias("Japanese", "japanese", "nihongo", "ja")
_alias("Chinese", "chinese", "mandarin", "zh", "zh-cn", "zh-tw", "中文", "普通话")

LANG_TO_GTTS = {
    "English": "en",
    "Tagalog": "tl",
    "Korean": "ko",
    "Spanish": "es",
    "Dutch":   "nl",
    "French":  "fr",
    "German":  "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Japanese": "ja",
    "Chinese":  "zh-CN",
}
def _pick_gtts_lang(prefer: Optional[str]) -> str:
    # prefer exact display name first; then try iso; fallback en
    if prefer in LANG_TO_GTTS:
        return LANG_TO_GTTS[prefer]
    key = (prefer or "en").lower()
    return LANG_TO_GTTS.get(LANG_ALIASES.get(key, ""), LANG_TO_GTTS.get(key, LANG_TO_GTTS.get(key.split("-")[0], "en")))

# =========================================================
# LLM plumbing
# =========================================================
# def llm_zero(temperature: float | None = None, model: str | None = None):
#     global _tracer_cache
#     if _tracer_cache is None:
#         _tracer_cache = init_langsmith(project=os.getenv("LANGCHAIN_PROJECT", "KusinaBot"))
#     kwargs = {
#         "model": model or LLM_MODEL,
#         "temperature": CHEF_TEMP if temperature is None else float(temperature),
#     }
#     if _tracer_cache:
#         kwargs["callbacks"] = _tracer_cache
#     return ChatOpenAI(**kwargs)
def llm_zero(temperature: float | None = None, model: str | None = None):
    global _tracer_cache, _LLM_CACHE
    if _tracer_cache is None:
        _tracer_cache = init_langsmith(project=os.getenv("LANGCHAIN_PROJECT", "KusinaBot"))
    chosen_model = model or LLM_MODEL
    chosen_temp = CHEF_TEMP if temperature is None else float(temperature)
    cache_key = (chosen_model, chosen_temp)
    if cache_key not in _LLM_CACHE:
        kwargs =  {
            "model": chosen_model,
            "temperature": chosen_temp,
            # ↓ add these so calls don’t hang
            "timeout": 30,          # hard cap per request
            "max_retries": 1,       # fail fast; we’ll fallback in app
        }
        if _tracer_cache:
            kwargs["callbacks"] = _tracer_cache
        _LLM_CACHE[cache_key] = ChatOpenAI(**kwargs)
    return _LLM_CACHE[cache_key]

# =========================================================
# Language detection
# =========================================================
def detect_language(text: str) -> str:
    """Return langdetect ISO like 'en','tl',… (fallback 'en')."""
    try:
        cands = detect_langs(text or "")
        if (text or "").isascii() and len(text or "") < 40:
            for c in cands:
                if c.lang == "en" and c.prob >= 0.15:
                    return "en"
        return cands[0].lang if cands else "en"
    except Exception:
        return "en"

# =========================================================
# Explicit language switch parsing
# 1) Try fast regex on known aliases (/lang: …, language to …, in … please)
# 2) If ambiguous, ask the LLM to decide (prompt-based parser)
# =========================================================

# whitelist alternation of known aliases (for strict matching)
_LANG_ALT = r"|".join(sorted(re.escape(k) for k in LANG_ALIASES.keys()))
_LANG_GROUP = rf"(?P<lang>{_LANG_ALT})"

# Strict command patterns
_CMD_PATTERNS = [
    rf"""^(?:\s*(?:/lang)\s*(?:to|in|:)?\s*{_LANG_GROUP}\s*)$""",
    rf"""^(?:\s*(?:language|switch|reply|answer|speak|use|set|respond)\s*(?:to|in|:)?\s*{_LANG_GROUP}\s*)$""",
    rf"""^(?:\s*(?:in\s+)?{_LANG_GROUP}\s+(?:please|pls)\s*)$""",
]
_CMD_REGEXES = [re.compile(p, re.IGNORECASE) for p in _CMD_PATTERNS]

def _norm(s: str) -> str:
    t = re.sub(r"[^\w\s\-]", " ", (s or "")).strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t

# Prompt-based disambiguation
LANG_SWITCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict parser. Decide if the user's text is an explicit language switch.\n"
     "- If YES, return ONLY a 2-letter ISO code among: en, tl, es, nl, fr, de, it, pt, ja, zh, ko.\n"
     "- If NO (e.g., 'recipe please', 'calorie count please', 'filling please', or any non-language request), return NONE.\n"
     "No explanations."),
    ("human", "{text}")
])

def parse_language_switch(user_text: str) -> Optional[str]:
    """Return a display name like 'English','Tagalog',… if the user explicitly asked to switch; else None."""
    if not user_text:
        return None

    # 1) Fast & deterministic: strict regex on known aliases
    norm = _norm(user_text)
    for rx in _CMD_REGEXES:
        m = rx.match(norm)
        if m:
            raw = (m.group("lang") or "").strip().lower()
            resolved = LANG_ALIASES.get(raw)
            if resolved:
                return resolved

    # 2) Ambiguous? Ask the LLM parser
    try:
        out = (LANG_SWITCH_PROMPT | llm_zero(temperature=0.0)).invoke({"text": user_text})
        val = (out.content or "").strip().lower()
        if val == "none" or not val:
            return None
        # map ISO → display name via aliases (e.g., "tl" -> "Tagalog")
        return LANG_ALIASES.get(val, LANG_ALIASES.get(val.split("-")[0], None))
    except Exception:
        return None

# =========================================================
# Translation helpers
# =========================================================
TRANS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Translate to English. Return translation only if input is not English."),
    ("human", "{text}"),
])

def translate_to_english(text: str) -> str:
    if not text:
        return text
    try:
        cands = detect_langs(text)
        if cands and cands[0].lang == "en" and cands[0].prob >= 0.6:
            return text
    except Exception:
        pass
    out = (TRANS_PROMPT | llm_zero(temperature=0.0)).invoke({"text": text})
    return out.content.strip()

def ensure_reply_language(text: str, target_lang: Optional[str]) -> str:
    """
    Keep your existing signature/behavior:
    - target_lang is a display name like "English","Tagalog",…
    - If text already matches the intended language (best-effort), return as-is;
      otherwise translate preserving formatting.
    """
    if not text or not target_lang:
        return text or ""
    try:
        cands = detect_langs(text)
        if cands and cands[0].prob >= 0.5:
            det = cands[0].lang
            tl = (target_lang or "").lower()
            if (tl.startswith("en") and det == "en") \
               or (tl.startswith("tl") and det in ("tl","fil")) \
               or (tl.startswith("es") and det == "es") \
               or (tl.startswith("nl") and det == "nl") \
               or (tl.startswith("fr") and det == "fr") \
               or (tl.startswith("de") and det == "de") \
               or (tl.startswith("it") and det == "it") \
               or (tl.startswith("pt") and det == "pt") \
               or (tl.startswith("ja") and det == "ja") \
               or (tl.startswith("ko") and det == "ko") \
               or (tl.startswith("zh") and det.startswith("zh")):
                return text
    except Exception:
        pass

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Translate into the target language. Preserve bullets and formatting. Return ONLY the translated text."),
        ("human", "Target language: {lang}\n\nText:\n{txt}"),
    ])
    out = (prompt | llm_zero(temperature=0.0)).invoke({"lang": target_lang, "txt": text})
    return out.content.strip()

# =========================================================
# Ingredient localization (kept)
# =========================================================
ING_TRANSLATIONS = {
    "tl": {"lime": "dayap", "garlic": "bawang", "onion": "sibuyas", "soy sauce": "toyo"},
}
def localize_ingredients(ings: List[str], lang: Optional[str]) -> List[str]:
    mapping = ING_TRANSLATIONS.get((lang or "").lower())
    if not mapping:
        return ings
    keys = sorted(mapping.keys(), key=len, reverse=True)
    out = []
    for s in ings:
        t = s
        for k in keys:
            t = re.sub(rf"(?i)\b{re.escape(k)}\b", mapping[k], t)
        out.append(t)
    return out

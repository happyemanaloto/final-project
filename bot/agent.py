from __future__ import annotations
from .telemetry import init_langsmith

import json, re, os
from typing import Optional, List, Dict

from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document

# ✅ FIXED: no trailing comma; and import LANG_ALIASES
from .nlp import ensure_reply_language, llm_zero, LANG_ALIASES
# Build a strict alternation of known language aliases to avoid false positives
_LANG_ALT = r"|".join(sorted(re.escape(k) for k in LANG_ALIASES.keys()))
_LANG_GROUP = rf"(?P<lang>{_LANG_ALT})"

from .tools import (
    vector_search, keyword_search, transcribe_media, estimate_nutrition,
    make_shopping_list, create_cookbook, add_feedback, translate_text,
    summarize_video, qa_video, ingest_link, calories_from_url,
    _session_get_hits, _session_set_hits,
)
from .session import SessionMemory
from bot.taxonomy import ALIASES

import warnings

try:
    from langchain_core._api import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:
    pass
warnings.filterwarnings("ignore", category=DeprecationWarning)

RAG_LOWCONF_DISTANCE = float(os.getenv("RAG_LOWCONF_DISTANCE", "0.6"))  # FAISS: lower is better

# -------- Heuristics / regex --------
URL_RE = re.compile(r"(https?://\S+)", re.I)
CALORIE_TRIGGERS = r"(?:calorie|calories|kcal|nutrition|macros?|protein|carbs?|fat|kilocal)"

# -------- System Prompt --------
SYSTEM = SystemMessage(content="""You are Happy Kusina-Bot: a cheerful, funny home cook and nutrition coach.

Voice & style
- Sound human, warm, and encouraging. Use contractions and at most 1–2 emojis total (🍳🥗), not every line.
- Prefer short sentences and compact dash bullets. Avoid literal asterisks/markdown artifacts in output.
- Keep dish names in their native form when appropriate (croissant, pintxos).
- If a dish term appears, add a 1-line friendly definition before instructions.

Language
- Always reply in reply_language. Detect user/media language for retrieval, but do not change reply_language unless user requests.

Uncertainty & safety
- If evidence is weak or you’re unsure:
  1) Say you’re not fully sure.
  2) Ask 1–2 concise clarifying questions (origin, key ingredient, style).
  3) Offer a cautious best-guess: “This might be similar to ___, so you could ___”.
- Never fabricate precise facts when uncertain. Prefer safe, generic techniques.

Workflow
1) If input has media, call transcribe_media first.
2) Translate the user request (and any transcript) to English for retrieval; keep reply_language for the final answer.
3) Extract preferences as JSON with keys:
   language, cuisine, part_of_meal, part_of_day, heavy_or_light, time_minutes, difficulty,
   budget, available_ingredients, servings, allergens, goals, include_ingredients, exclude_ingredients, free_text.
4) If seed_recipe_id is provided, prioritize that recipe; you may summarize it directly without calling transcription.                      
5) Call vector_search first using vector_search_plan (time_limit, cuisine, must_include, exclude_ingredients, avoid_allergens, display_lang). Fallback to keyword_search if needed.
6) If request info is sparse, still suggest 2–3 practical, healthy recipes using common/easy-to-source ingredients in reply_language (no apologies).
7) If the user asks for a specific recipe, summarize it in reply_language with title, ingredients, steps, and a tip. Use the most relevant hit from vector_search or keyword_search.
8) Share the recipe's source URL and image URL if available. If no image, use a placeholder.
9) When summarizing a recipe, add a one-line time estimate (‘⏱️ ~NN min’).
10) If the user asks for nutrition info, call estimate_nutrition with the ingredients of the most relevant recipe.  
11) If the user asks for a shopping list, call make_shopping_list with the recipes shown.
12) If the user asks for a cookbook, call create_cookbook with selected recipe_ids.

Tools & follow-ups
- Calories/macros: if the user asks (e.g., “how many calories?”, “calorie count of X”), call estimate_nutrition with the ingredients of the most relevant recipe (use recent results if available). Answer in reply_language with compact numbers per serving.
- Shopping list: if the user asks for a shopping/grocery list and recipes were just shown, call make_shopping_list (recipes may be omitted; use cached hits). Respond in reply_language.
- Feedback: if the user gives feedback on a recipe, call add_feedback.
- Translation: if the user asks to translate text, call translate_text with raw_user_text (or the shown recipe text) and reply with ONLY the translated text.

Keep it concise, friendly, and helpful.
""")

def maybe_lookup_term(q: str) -> str | None:
    key = (q or "").lower().strip()
    # strip common punctuation/quotes around one-word queries
    key = key.strip("¿?!().,;: \"'“”’")
    return ALIASES.get(key)

def retrieve_with_scores(vs, q: str, k: int = 4):
    """Return List[(Document, distance)]. Lower distance = more similar."""
    try:
        return vs.similarity_search_with_score(q, k=k)
    except Exception:
        docs = vs.similarity_search(q, k=k)
        return [(d, 0.0) for d in docs]

def guarded_chat_once(agent, session, message: str, reply_lang: str = "en") -> str:
    """
    1) If retrieval is low-confidence → ask user to clarify (no hallucinations).
    2) If confident → prepend concise context + control line, then call your existing agent.
    """
    vs = getattr(session, "vectorstore", None)
    hits = retrieve_with_scores(vs, message, k=4) if vs else []

    def _clarifier():
        prefix = (
            "No estoy seguro" if reply_lang.startswith("es")
            else "Je ne suis pas certain" if reply_lang.startswith("fr")
            else "I’m not fully sure"
        )
        return (
            f"{prefix} what you mean by “{message}”. "
            "Could you describe it (country, main ingredient, how it’s served)? "
            "I’ll tailor the recipe once I know more. 😊"
        )

    # Low confidence → ask instead of inventing
    if (not hits) or (hits and hits[0][1] > RAG_LOWCONF_DISTANCE):
        return _clarifier()

    # Confident → build compact context block
    ctx = "\n".join(f"- {d.page_content}" for d, _ in hits)

    # Control line tells the agent the target language & tone (since SYSTEM is static)
    control = (
        f"Reply language: {reply_lang}\n"
        "Tone: cheerful, friendly, short sentences, dash bullets; avoid literal markdown symbols."
    )

    # Enrich the user's input (works with OPENAI_FUNCTIONS / memory in your agent)
    enriched_input = (
        f"{control}\n\n"
        f"Relevant food facts:\n{ctx}\n\n"
        f"User:\n{message}"
    )

    # Call your existing agent (has memory + tools)
    out = agent.invoke({"input": enriched_input})
    # AgentExecutor usually returns dict with 'output'
    return (out.get("output") if isinstance(out, dict) and "output" in out else str(out))


def make_agent_chain(llm, session):
    vs = getattr(session, "vectorstore", None)
    if vs is None:
        raise RuntimeError("Session has no vectorstore. Set session.vectorstore = vs when you build it.")

    def run(q: str, lang: str):
        hits = retrieve_with_scores(vs, q, k=4)
        top_doc, top_dist = (hits[0] if hits else (Document(page_content=""), 1.0))

        # Low confidence → ask the user instead of inventing
        if (not hits) or (top_dist > RAG_LOWCONF_DISTANCE):
            prefix = (
                "No estoy seguro" if lang.startswith("es")
                else "Je ne suis pas certain" if lang.startswith("fr")
                else "I’m not fully sure"
            )
            return (
                f"{prefix} what you mean by “{q}”. "
                "Could you describe it (country, main ingredient, how it’s served)? "
                "I’ll tailor the recipe once I know more. 😊"
            )

        # Confident → include retrieved context
        ctx = "\n\n".join(f"- {d.page_content}" for d, _ in hits)
        prompt = ChatPromptTemplate.from_messages([
            ("system", STYLE_SYS),
            ("system", "Relevant food facts:\n{context}"),
            ("system", SYSTEM),
            ("human", "{question}")
        ])
        out = (prompt | llm).invoke({"lang": lang, "context": ctx, "question": q})
        return out.content

    return run

# -------- Build the agent (with memory) --------
def build_agent(llm, session: SessionMemory):
    tools = [
        vector_search, keyword_search, transcribe_media, estimate_nutrition,
        make_shopping_list, create_cookbook, add_feedback, translate_text,
        summarize_video, qa_video, ingest_link, calories_from_url
    ]
    callbacks = init_langsmith(project=os.getenv("LANGCHAIN_PROJECT", "KusinaBot"))

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="input",
        output_key="output",
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=False,
        handle_parsing_errors=True,
        agent_kwargs={"system_message": SYSTEM},
        memory=memory,
        callbacks=callbacks or None,   # <— attaches tracer if enabled

    )
    return agent

# -------- Helpers --------
def _extract_url(text: str) -> Optional[str]:
    m = URL_RE.search(text or "")
    return m.group(1) if m else None

def _get_last_ai_text(agent) -> str:
    """Pull the last assistant message from ConversationBufferMemory."""
    try:
        msgs = agent.memory.chat_memory.messages
        for m in reversed(msgs):
            role = getattr(m, "type", None) or getattr(m, "role", None) or m.__class__.__name__.lower()
            if "ai" in str(role):
                return m.content or ""
    except Exception:
        pass
    return ""

def _translation_intent(user_text: str) -> Optional[Dict[str, str]]:
    """
    Returns {"lang_name": <alias>, "payload": <text or ''>} or None.
    Only fires when a KNOWN language alias is present.
    """
    t = (user_text or "").strip()

    # A) 'translate <X> to|into|in <lang>'
    m = re.search(rf'^\s*translate\s+(?P<payload>.*?)\s+(?:to|into|in)\s+{_LANG_GROUP}\s*$', t, re.I)
    if m:
        payload = (m.group("payload") or "").strip()
        lang_alias = (m.group("lang") or "").strip().lower()
        return {"lang_name": lang_alias, "payload": payload}

    # B) 'translate to|into <lang>' (no explicit text)
    m = re.search(rf'^\s*translate(?:\s+(?:to|into))?\s+{_LANG_GROUP}\s*$', t, re.I)
    if m:
        lang_alias = (m.group("lang") or "").strip().lower()
        return {"lang_name": lang_alias, "payload": ""}

    # C) Flexible: 'can you translate ... in <lang>' (anywhere in sentence)
    m = re.search(rf'\btranslate\b.*\b(?:to|into|in)\s+{_LANG_GROUP}\b', t, re.I)
    if m:
        lang_alias = (m.group("lang") or "").strip().lower()
        return {"lang_name": lang_alias, "payload": ""}

    # D) '<lang> please' / 'in <lang> please'  — only if <lang> is a known alias
    m = re.search(rf'^\s*(?:in\s+)?{_LANG_GROUP}\s+(?:please|pls)\s*$', t, re.I)
    if m:
        lang_alias = (m.group("lang") or "").strip().lower()
        return {"lang_name": lang_alias, "payload": ""}

    # E) 'previous ... in <lang>'
    m = re.search(rf'^\s*previous(?:\s+\w+)*\s+in\s+{_LANG_GROUP}\s*$', t, re.I)
    if m:
        lang_alias = (m.group("lang") or "").strip().lower()
        return {"lang_name": lang_alias, "payload": ""}

    return None

def _cache_from_last_ai_text(agent, reply_lang: str) -> List[Dict]:
    """
    When user asks 'calorie count for these?' but we have no session hits,
    parse the previous assistant message to extract up to 2 dish ideas and
    draft core ingredient lists for each. Cache them into session hits.
    Returns the cached hits.
    """
    last_txt = _get_last_ai_text(agent)
    if not last_txt:
        return []

    # Escape literal braces in ChatPromptTemplate using double braces.
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "From the assistant text, extract up to TWO concrete dish ideas and list core ingredients for each. "
            "Return ONLY compact JSON as a list of objects like:\n"
            "[{{\"title\": \"Dish name\", \"ingredients\": [\"item 1\", \"item 2\", \"item 3\"]}}]. "
            "No steps, no chatter."
        ),
        ("human", "Assistant text:\n{t}")
    ])

    out = (prompt | llm_zero(temperature=0)).invoke({"t": last_txt})
    try:
        items = json.loads(out.content)
        hits: List[Dict] = []
        for it in items[:2]:
            title = (it.get("title") or "Dish").strip()
            ings = [s.strip() for s in (it.get("ingredients") or []) if s and s.strip()]
            if not ings:
                continue
            hits.append({
                "id": f"synth:{title.lower()}",
                "title": title,
                "url": "",
                "source": "assistant",
                "ingredients": ings,
                "steps": []
            })
        if hits:
            _session_set_hits(hits)
        return hits
    except Exception:
        return []

# -------- Single turn router --------
def chat_once(agent, user_text: str, reply_lang: str) -> str:
    """
    Routes:
      - Translation (runs first; uses last AI message if no payload)
      - URL + calories -> calories_from_url
      - URL only       -> ingest_link
      - 'calories'     -> use session hits or synthesize from last AI; then estimate
      - else           -> agent flow (has conversation memory)
    """
    # 0) TRANSLATION SHORT-CIRCUIT
    # intent = _translation_intent(user_text)
    # if intent:
    #     lang_code = LANG_ALIASES.get(intent["lang_name"])
    #     if not lang_code:
    #         return ensure_reply_language("Which language should I translate to?", reply_lang)

    #     payload = intent["payload"]
    #     # If payload missing or pronoun, translate the last assistant message
    #     if not payload or payload.lower() in {"this", "that", "them", "it", "above", "previous", "previous recipes", "those"}:
    #         payload = _get_last_ai_text(agent)

    #     if not payload:
    #         return ensure_reply_language("Paste the text you want me to translate. 🙂", reply_lang)

    #     out = translate_text.invoke({"text": payload, "target_lang": lang_code})
    #     return out if isinstance(out, str) else str(out)
    intent = _translation_intent(user_text)
    if intent:
        lang_code = LANG_ALIASES.get(intent["lang_name"])
        if not lang_code:
            # Not a known language after all — treat as a normal message.
            intent = None
        else:
            payload = intent["payload"]
            if not payload or payload.lower() in {"this", "that", "them", "it", "above", "previous", "previous recipes", "those"}:
                payload = _get_last_ai_text(agent)
            if not payload:
                return ensure_reply_language("Paste the text you want me to translate. 🙂", reply_lang)
            out = translate_text.invoke({"text": payload, "target_lang": lang_code})
            return out if isinstance(out, str) else str(out)

    # 1) URL routing
    url = _extract_url(user_text)

    # 1a) URL + calories
    if url and re.search(CALORIE_TRIGGERS, user_text, flags=re.I):
        res = calories_from_url.invoke({
            "url": url,
            "servings": 1,
            "locale": "EU" if reply_lang.lower() not in {"en-us", "arz"} else "US",
            "target_lang": reply_lang
        })
        return res if isinstance(res, str) else str(res)

    # 1b) URL only: ingest and cache hit
    if url:
        res = ingest_link.invoke({"url": url, "target_lang": reply_lang})
        return res if isinstance(res, str) else str(res)

    # 2) Calories short-circuit (no URL)
    if re.search(CALORIE_TRIGGERS, user_text, flags=re.I):
        hits = _session_get_hits()

        # If no cached hits (typical after ideas), synthesize from last AI text
        if not hits:
            hits = _cache_from_last_ai_text(agent, reply_lang)

        if hits:
            lines = []
            for h in hits[:2]:
                ings = h.get("ingredients") or []
                if isinstance(ings, str):
                    ings = [ings]
                if not ings:
                    continue
                try:
                    est = estimate_nutrition.invoke({
                        "ingredients": ings,
                        "servings": 1,
                        "locale": "EU" if reply_lang.lower() not in {"en-us", "arz"} else "US"
                    })
                    est_txt = est if isinstance(est, str) else str(est)
                    title = h.get("title") or "Dish"
                    lines.append(f"{title} per serving:\n{est_txt}")
                except Exception:
                    continue

            if lines:
                return ensure_reply_language("\n\n".join(lines), reply_lang)

    # 3) Normal agent path (with memory)
    directive = {"reply_language": reply_lang, "raw_user_text": user_text}
    res = agent.invoke({"input": json.dumps(directive, ensure_ascii=False)})
    answer = res.get("output", str(res)) if isinstance(res, dict) else str(res)

    # Cache the dishes mentioned in this reply so they can be used for calories/shopping
    try:
        _cache_from_last_ai_text(agent, reply_lang)
    except Exception:
        pass

    return ensure_reply_language(answer, reply_lang)

# from __future__ import annotations
# from .telemetry import init_langsmith

# import json, re, os, warnings
# from typing import Optional, List, Dict

# from langchain.agents import AgentType, initialize_agent
# from langchain.memory import ConversationBufferMemory
# from langchain_core.messages import SystemMessage
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.docstore.document import Document

# from .nlp import ensure_reply_language, llm_zero, LANG_ALIASES
# from .tools import (
#     vector_search, keyword_search, transcribe_media, estimate_nutrition,
#     make_shopping_list, create_cookbook, add_feedback, translate_text,
#     summarize_video, qa_video, ingest_link, calories_from_url,
#     _session_get_hits, _session_set_hits,
# )

# from .session import SessionMemory
# from bot.taxonomy import ALIASES

# # quiet deprecations
# try:
#     from langchain_core._api import LangChainDeprecationWarning
#     warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
# except Exception:
#     pass
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# # ---------------- Config ----------------
# # LOWER distance = better similarity (Chroma/FAISS with cosine)
# RAG_LOWCONF_DISTANCE = float(os.getenv("RAG_LOWCONF_DISTANCE", "0.3"))

# # -------- Heuristics / regex --------
# URL_RE = re.compile(r"(https?://\S+)", re.I)
# CALORIE_TRIGGERS = r"(?:calorie|calories|kcal|nutrition|macros?|protein|carbs?|fat|kilocal)"

# # -------- System Prompt --------
# SYSTEM = SystemMessage(content="""You are Happy Kusina-Bot: a cheerful, funny home cook and nutrition coach.

# Voice & style
# - Sound human, warm, and encouraging. Use contractions and at most 1–2 emojis total (🍳🥗).
# - Prefer short sentences and compact dash bullets. Avoid literal asterisks/markdown artifacts.
# - Keep dish names in their native form when appropriate (croissant, pintxos).
# - If a dish term appears, add a 1-line friendly definition before instructions.

# Language
# - Always reply in reply_language. Detect user/media language for retrieval, but do not change reply_language unless user requests.

# Uncertainty & safety
# - If evidence is weak or you’re unsure:
#   1) Say you’re not fully sure.
#   2) Ask 1–2 concise clarifying questions (origin, key ingredient, style).
#   3) Offer a cautious best-guess: “This might be similar to ___, so you could ___”.
# - Never fabricate precise facts when uncertain. Prefer safe, generic techniques.

# Workflow
# 1) If input has media, call transcribe_media first.
# 2) Translate the user request (and any transcript) to English for retrieval; keep reply_language for the final answer.
# 3) Extract preferences as JSON with keys:
#    language, cuisine, part_of_meal, part_of_day, heavy_or_light, time_minutes, difficulty,
#    budget, available_ingredients, servings, allergens, goals, include_ingredients, exclude_ingredients, free_text.
# 4) If seed_recipe_id is provided, prioritize that recipe; you may summarize it directly without calling transcription.
# 5) Call vector_search first using vector_search_plan (time_limit, cuisine, must_include, exclude_ingredients, avoid_allergens, display_lang). Fallback to keyword_search if needed.
# 6) If request info is sparse, still suggest 2–3 practical, healthy recipes using common/easy-to-source ingredients in reply_language (no apologies).
# 7) If the user asks for a specific recipe, summarize it in reply_language with title, ingredients, steps, and a tip. Use the most relevant hit from vector_search or keyword_search.
# 8) Share the recipe's source URL and image URL if available. If no image, use a placeholder.
# 9) When summarizing a recipe, add a one-line time estimate (‘⏱️ ~NN min’).
# 10) If the user asks for nutrition info, call estimate_nutrition with the ingredients of the most relevant recipe.
# 11) If the user asks for a shopping list, call make_shopping_list with the recipes shown.
# 12) If the user asks for a cookbook, call create_cookbook with selected recipe_ids.

# Keep it concise, friendly, and helpful.
# """)

# # ---------------- Alias helper ----------------
# def maybe_lookup_term(q: str) -> str | None:
#     key = (q or "").lower().strip()
#     key = key.strip("¿?!().,;: \"'“”’")
#     return ALIASES.get(key)

# # --- Friendly intent/ingredient helpers ---

# # Soften when to clarify vs help
# QUICK_HINT_WORDS = {
#     "quick","fast","hurry","in a hurry","under 15","under 20","busy",
#     "mabilis","nagmamadali","madali","bilis",
# }
# PANTRY_HINT_WORDS = {
#     "have","only have","leftover","pantry","just","only",
#     "meron","meron lang","may","lang","ulam","pwede","pwedeng",
# }
# CLEAR_TASK_HINTS = {
#     "recipe","receta","recette","how to make","how do i make",
#     "what can i make","ano ang pwedeng ulam","pwedeng ulam","lutuin","gawin","cook",
# }

# def _has_quick_intent(txt: str) -> bool:
#     t = (txt or "").lower()
#     return any(w in t for w in QUICK_HINT_WORDS)

# def _has_pantry_signal(txt: str) -> bool:
#     t = (txt or "").lower()
#     return any(w in t for w in PANTRY_HINT_WORDS)

# def _is_clear_task(txt: str) -> bool:
#     t = (txt or "").lower()
#     return any(h in t for h in CLEAR_TASK_HINTS) or len(t.split()) >= 4

# def _looks_like_short_dish_query(txt: str) -> bool:
#     t = (txt or "").strip().strip("¿?!().,;:\"'“”’").split()
#     return 0 < len(t) <= 2

# def _extract_simple_ingredients(txt: str) -> list[str]:
#     """
#     Super-lightweight ingredient sniffing for EN/TL. Splits on commas and 'and/at'.
#     """
#     t = (txt or "").lower()
#     # normalize common connectors
#     for sep in [" and ", " y ", " et ", " at "]:  # EN/ES/FR/TL 'and'
#         t = t.replace(sep, ", ")
#     # strip obvious non-ingredient words
#     junk = {"i", "im", "i’m", "ako", "ano", "pwede", "pwedeng", "ulam", "meron", "lang", "only", "have", "just", "recipe", "cook"}
#     cand = [w.strip(" .?!:;’“”\"'()") for w in t.split(",")]
#     ings = [w for w in cand if w and w not in junk and len(w.split()) <= 3]
#     # common TL terms map to English hints so LLM sees both
#     tl_gloss = {"sitaw": "string beans", "talong": "eggplant"}
#     out = []
#     for w in ings:
#         out.append(w)
#         if w in tl_gloss:
#             out.append(tl_gloss[w])
#     # dedupe preserving order
#     seen = set()
#     clean = []
#     for x in out:
#         if x not in seen:
#             seen.add(x)
#             clean.append(x)
#     return clean

# # def _looks_like_short_dish_query(txt: str) -> bool:
# #     """True for very short queries that are likely a dish name (1–2 words, no punctuation)."""
# #     t = (txt or "").strip().strip("¿?!().,;:\"'“”’").split()
# #     return 0 < len(t) <= 2
# def _looks_like_short_dish_query(txt: str) -> bool:
#     t = (txt or "").lower()
#     if "calorie" in t or "kcal" in t or "nutrition" in t:
#         return False
#     t = t.strip().strip("¿?!().,;:\"'“”’").split()
#     return 0 < len(t) <= 2


# def guarded_chat_once(agent, session, message: str, reply_lang: str = "en") -> str:
#     """
#     Softer router:
#       - If alias hit or high-confidence RAG: add context → call agent.
#       - If low-confidence BUT we see pantry/quick intent or ingredients: skip clarifier → call agent with a concise control block (no retrieval needed).
#       - Otherwise: short, friendly clarifier (only for very short dish-like queries).
#     """
#     vs = getattr(session, "vectorstore", None)
#     if re.search(CALORIE_TRIGGERS, message or "", flags=re.I):
#         hits = _session_get_hits() or _cache_from_last_ai_text(agent, reply_lang)
#         if not hits:
#             # graceful ask if we truly have nothing to estimate from
#             if reply_lang.startswith("es"):
#                 return "¿Puedes decirme los ingredientes o pegar la receta? Así calculo las calorías por ración. 🙂"
#             if reply_lang.startswith("fr"):
#                 return "Tu peux me donner les ingrédients ou coller la recette ? Je calcule les calories par portion. 🙂"
#             if reply_lang.startswith(("tl","fil")):
#                 return "Pakilagay ang mga sangkap o i-paste ang recipe para ma-estimate ko ang calories kada serving. 🙂"
#             return "Paste the ingredients or recipe text and I’ll estimate per-serving calories. 🙂"

#         lines = []
#         for h in hits[:2]:
#             ings = h.get("ingredients") or []
#             if isinstance(ings, str):
#                 ings = [ings]
#             if not ings:
#                 continue
#             try:
#                 est = estimate_nutrition.invoke({
#                     "ingredients": ings,
#                     "servings": 1,
#                     "locale": "EU" if reply_lang.lower() not in {"en-us", "arz"} else "US"
#                 })
#                 title = h.get("title") or "Dish"
#                 est_txt = est if isinstance(est, str) else str(est)
#                 lines.append(f"**{title}** — per serving:\n{est_txt}")
#             except Exception:
#                 continue
#         if lines:
#             return ensure_reply_language("\n\n".join(lines), reply_lang)
#         # fall through if nothing worked…
#     # 0) Alias fast-path
#     alias_query = maybe_lookup_term(message)
#     alias_hits = []
#     if vs and alias_query:
#         try:
#             alias_hits = vs.similarity_search(alias_query, k=1, filter={"source": "canon"})
#         except Exception:
#             alias_hits = vs.similarity_search(alias_query, k=1)

#     # 1) Normal retrieval with scores
#     # hits = retrieve_with_scores(vs, message, k=4) if vs else []
#     # low_conf = (not hits) or (hits and hits[0][1] > RAG_LOWCONF_DISTANCE)
#     hits = retrieve_with_scores(vs, message, k=4) if vs else []
#     low_conf = (not hits) or (hits and hits[0][1] > RAG_LOWCONF_DISTANCE)

#     # 2) Pantry/quick fallback: be helpful instead of asking
#     # ingredients = _extract_simple_ingredients(message)
#     # has_quick = _has_quick_intent(message)
#     # has_pantry = _has_pantry_signal(message) or len(ingredients) >= 1
#     is_short_dishy = _looks_like_short_dish_query(message)
#     has_quick = _has_quick_intent(message)
#     has_pantry = _has_pantry_signal(message)
#     is_clear = _is_clear_task(message)
#     # 👉 Only clarify if it's BOTH low-confidence AND a super-short dish-like query,
#     #    with no quick/pantry/clear-task hints.
#     needs_clarify = low_conf and is_short_dishy and not (has_quick or has_pantry or is_clear or alias_hits)

#     if needs_clarify:
#         # very short & ambiguous → soft clarifier
#         if reply_lang.startswith("es"):
#             return f"No estoy del todo seguro sobre “{message}”. ¿Es el nombre de un plato? Dime el país o ingrediente principal y te ayudo. 🙂"
#         if reply_lang.startswith("fr"):
#             return f"Je ne suis pas sûr de “{message}”. C’est le nom d’un plat ? Dis-moi le pays ou l’ingrédient principal. 🙂"
#         if reply_lang.startswith("tl") or reply_lang.startswith("fil"):
#             return f"Hindi ako sigurado sa “{message}”. Pangalan ba ito ng ulam? Sabihin mo lang ang bansa o pangunahing sangkap. 🙂"
#         return f"I’m not totally sure about “{message}”. Is it a dish name? Share the country or main ingredient and I’ll nail it. 🙂"

#     # if low_conf and not alias_hits and (has_quick or has_pantry or not _looks_like_short_dish_query(message)):
#         control = (
#             f"Reply language: {reply_lang}\n"
#             "Tone: cheerful, friendly, short sentences; dash bullets; avoid literal markdown symbols.\n"
#             "If the user seems in a hurry, give 2 quick options (~15–20 min) with 3–4 steps each.\n"
#             "Use the given ingredients; if something is missing, suggest simple substitutions.\n"
#             "If uncertain, say so briefly and explain why the suggestion should still work."
#         )
#         pantry_line = f"User ingredients (parsed): {', '.join(ingredients)}" if ingredients else "User ingredients: (not explicitly parsed)"
#         enriched_input = f"{control}\n\n{pantry_line}\n\nUser:\n{message}"
#         out = agent.invoke({"input": enriched_input})
#         return (out.get("output") if isinstance(out, dict) and "output" in out else str(out))
# # Low confidence but it's clearly a cooking request → BE HELPFUL (no nagging)
#     if low_conf and not needs_clarify:
#         ingredients = _extract_simple_ingredients(message) if ' _extract_simple_ingredients' in globals() else []
#         control = (
#             f"Reply language: {reply_lang}\n"
#             "Tone: cheerful, friendly, short sentences; dash bullets; avoid literal markdown symbols.\n"
#             "If the user seems in a hurry, give 2 quick options (~15–20 min) with 3–4 steps each.\n"
#             "Use the given ingredients; if something is missing, suggest simple substitutions.\n"
#             "If uncertain, say so briefly and explain why the suggestion should still work."
#         )
#         pantry_line = (
#             f"User ingredients (parsed): {', '.join(ingredients)}"
#             if ingredients else
#             "User ingredients: (not explicitly parsed)"
#         )
#         enriched_input = (
#             f"{control}\n\n"
#             f"{pantry_line}\n\n"
#             f"User:\n{message}"
#         )
#         out = agent.invoke({"input": enriched_input})

#         return (out.get("output") if isinstance(out, dict) and "output" in out else str(out))

#     # 3) High-confidence path (or alias): include retrieved context
#     ordered_docs = [*alias_hits, *[d for d, _ in hits]]
#     ctx = "\n".join(f"- {d.page_content}" for d in ordered_docs[:4]) if ordered_docs else ""
#     control = (
#         f"Reply language: {reply_lang}\n"
#         "Tone: cheerful, friendly, short sentences; dash bullets; avoid literal markdown symbols."
#     )
#     enriched_input = f"{control}\n\nRelevant food facts:\n{ctx}\n\nUser:\n{message}"
#     out = agent.invoke({"input": enriched_input})
#     return (out.get("output") if isinstance(out, dict) and "output" in out else str(out))

# # ---------------- Retrieval helpers ----------------
# def retrieve_with_scores(vs, q: str, k: int = 4):
#     """Return List[(Document, distance)]. Lower distance = more similar."""
#     if not vs:
#         return []
#     try:
#         return vs.similarity_search_with_score(q, k=k)
#     except Exception:
#         docs = vs.similarity_search(q, k=k)
#         return [(d, 0.0) for d in docs]

# def _extract_url(text: str) -> Optional[str]:
#     m = URL_RE.search(text or "")
#     return m.group(1) if m else None

# def _get_last_ai_text(agent) -> str:
#     """Pull the last assistant message from ConversationBufferMemory."""
#     try:
#         msgs = agent.memory.chat_memory.messages
#         for m in reversed(msgs):
#             role = getattr(m, "type", None) or getattr(m, "role", None) or m.__class__.__name__.lower()
#             if "ai" in str(role):
#                 return m.content or ""
#     except Exception:
#         pass
#     return ""

# # ---------------- Translation intent ----------------
# def _translation_intent(user_text: str) -> Optional[Dict[str, str]]:
#     """
#     Returns {"lang_name": <str>, "payload": <text or ''>} or None.
#     """
#     t = (user_text or "").strip()

#     m = re.search(r'^\s*translate\s+(.*?)\s+(?:to|into|in)\s+([A-Za-z\- ]+)\s*$', t, re.I)
#     if m:
#         return {"lang_name": m.group(2).strip().lower(), "payload": m.group(1).strip()}

#     m = re.search(r'^\s*translate(?:\s+(?:to|into))?\s+([A-Za-z\- ]+)\s*$', t, re.I)
#     if m:
#         return {"lang_name": m.group(1).strip().lower(), "payload": ""}

#     m = re.search(r'\btranslate\b.*\b(?:to|into|in)\s+([A-Za-z\- ]+)\b', t, re.I)
#     if m:
#         return {"lang_name": m.group(1).strip().lower(), "payload": ""}

#     m = re.search(r'^\s*(?:in\s+)?([A-Za-z\- ]+)\s+please\s*$', t, re.I)
#     if m:
#         return {"lang_name": m.group(1).strip().lower(), "payload": ""}

#     m = re.search(r'^\s*previous(?:\s+\w+)*\s+in\s+([A-Za-z\- ]+)\s*$', t, re.I)
#     if m:
#         return {"lang_name": m.group(1).strip().lower(), "payload": ""}

#     return None

# # ---------------- Cache helpers for nutrition ----------------
# def _cache_from_last_ai_text(agent, reply_lang: str) -> List[Dict]:
#     """
#     When user asks 'calorie count for these?' but we have no session hits,
#     parse the previous assistant message to extract up to 2 dish ideas
#     and core ingredients for each. Cache them into session hits.
#     """
#     last_txt = _get_last_ai_text(agent)
#     if not last_txt:
#         return []

#     prompt = ChatPromptTemplate.from_messages([
#         (
#             "system",
#             "From the assistant text, extract up to TWO concrete dish ideas and list core ingredients for each. "
#             "Return ONLY compact JSON as a list like: "
#             "[{\"title\": \"Dish name\", \"ingredients\": [\"item1\",\"item2\",\"item3\"]}]."
#         ),
#         ("human", "Assistant text:\n{t}")
#     ])

#     out = (prompt | llm_zero(temperature=0)).invoke({"t": last_txt})
#     try:
#         items = json.loads(out.content)
#         hits: List[Dict] = []
#         for it in items[:2]:
#             title = (it.get("title") or "Dish").strip()
#             ings = [s.strip() for s in (it.get("ingredients") or []) if s and s.strip()]
#             if not ings:
#                 continue
#             hits.append({
#                 "id": f"synth:{title.lower()}",
#                 "title": title,
#                 "url": "",
#                 "source": "assistant",
#                 "ingredients": ings,
#                 "steps": []
#             })
#         if hits:
#             _session_set_hits(hits)
#         return hits
#     except Exception:
#         return []

# # ---------------- Main router ----------------
# def chat_once(agent, user_text: str, reply_lang: str) -> str:
#     """
#     Routes:
#       - Translation (runs first; uses last AI message if no payload)
#       - URL + calories -> calories_from_url
#       - URL only       -> ingest_link
#       - 'calories'     -> use session hits or synthesize from last AI; then estimate
#       - else           -> guarded RAG (alias fast-path + low-confidence gate) → agent with memory/tools
#     """
#     # 0) TRANSLATION SHORT-CIRCUIT
#     intent = _translation_intent(user_text)
#     if intent:
#         lang_code = LANG_ALIASES.get(intent["lang_name"])
#         if not lang_code:
#             return ensure_reply_language("Which language should I translate to?", reply_lang)

#         payload = intent["payload"]
#         if not payload or payload.lower() in {"this","that","them","it","above","previous","previous recipes","those"}:
#             payload = _get_last_ai_text(agent)
#         if not payload:
#             return ensure_reply_language("Paste the text you want me to translate. 🙂", reply_lang)

#         out = translate_text.invoke({"text": payload, "target_lang": lang_code})
#         return out if isinstance(out, str) else str(out)

#     # 1) URL routing
#     url = _extract_url(user_text)

#     # 1a) URL + calories
#     if url and re.search(CALORIE_TRIGGERS, user_text, flags=re.I):
#         res = calories_from_url.invoke({
#             "url": url,
#             "servings": 1,
#             "locale": "EU" if reply_lang.lower() not in {"en-us", "arz"} else "US",
#             "target_lang": reply_lang
#         })
#         return res if isinstance(res, str) else str(res)

#     # 1b) URL only: ingest and cache hit
#     if url:
#         res = ingest_link.invoke({"url": url, "target_lang": reply_lang})
#         return res if isinstance(res, str) else str(res)

#     # 2) Calories short-circuit (no URL)
#     if re.search(CALORIE_TRIGGERS, user_text, flags=re.I):
#         hits = _session_get_hits() or _cache_from_last_ai_text(agent, reply_lang)
#         if hits:
#             lines = []
#             for h in hits[:2]:
#                 ings = h.get("ingredients") or []
#                 if isinstance(ings, str):
#                     ings = [ings]
#                 if not ings:
#                     continue
#                 try:
#                     est = estimate_nutrition.invoke({
#                         "ingredients": ings,
#                         "servings": 1,
#                         "locale": "EU" if reply_lang.lower() not in {"en-us", "arz"} else "US"
#                     })
#                     est_txt = est if isinstance(est, str) else str(est)
#                     title = h.get("title") or "Dish"
#                     lines.append(f"**{title}** — per serving:\n{est_txt}")
#                 except Exception:
#                     continue
#             if lines:
#                 return ensure_reply_language("\n\n".join(lines), reply_lang)

#     # 3) Guarded RAG in front of your LC agent (no hallucinations)
#     vs = getattr(agent, "vectorstore", None) or getattr(getattr(agent, "memory", None), "vectorstore", None)
#     # Better: we expect Streamlit to put it on the session:
#     # st.session_state.session.vectorstore = vs
#     if vs is None:
#         # try pulling from our SessionMemory instance that Streamlit stores
#         vs = getattr(SessionMemory, "vectorstore", None)

#     alias_query = maybe_lookup_term(user_text)
#     alias_hits = []
#     if vs and alias_query:
#         try:
#             alias_hits = vs.similarity_search(alias_query, k=1, filter={"source": "canon"})
#         except Exception:
#             alias_hits = vs.similarity_search(alias_query, k=1)

#     hits = retrieve_with_scores(vs, user_text, k=4) if vs else []
#     low_conf = (not hits) or (hits and hits[0][1] > RAG_LOWCONF_DISTANCE)
#     if low_conf and not alias_hits:
#         prefix = (
#             "No estoy seguro" if reply_lang.startswith("es")
#             else "Je ne suis pas certain" if reply_lang.startswith("fr")
#             else "I’m not fully sure"
#         )
#         clarifier = (
#             f"{prefix} what you mean by “{user_text}”. "
#             "Could you describe it (country, main ingredient, how it’s served)? "
#             "I’ll tailor the recipe once I know more. 😊"
#         )
#         return ensure_reply_language(clarifier, reply_lang)

#     # Build concise context (alias hit first)
#     ordered_docs = [*alias_hits, *[d for d, _ in hits]]
#     seen_ids = set()
#     ctx_lines = []
#     for d in ordered_docs:
#         mid = d.metadata.get("id") if hasattr(d, "metadata") else None
#         if mid and mid in seen_ids:
#             continue
#         seen_ids.add(mid)
#         ctx_lines.append(f"- {d.page_content}")
#         if len(ctx_lines) >= 4:
#             break
#     ctx = "\n".join(ctx_lines)

#     control = (
#         f"Reply language: {reply_lang}\n"
#         "Tone: cheerful, friendly, short sentences, dash bullets; avoid literal markdown symbols."
#     )
#     enriched = f"{control}\n\nRelevant food facts:\n{ctx}\n\nUser:\n{user_text}"

#     # 4) Normal agent path (with memory + tools), but with enriched input
#     directive = {"reply_language": reply_lang, "raw_user_text": enriched}
#     res = agent.invoke({"input": json.dumps(directive, ensure_ascii=False)})
#     answer = res.get("output", str(res)) if isinstance(res, dict) else str(res)
#     return ensure_reply_language(answer, reply_lang)

# # ---------------- Builder (keep memory + tools) ----------------
# def build_agent(llm, session: SessionMemory):
#     tools = [
#         vector_search, keyword_search, transcribe_media, estimate_nutrition,
#         make_shopping_list, create_cookbook, add_feedback, translate_text,
#         summarize_video, qa_video, ingest_link, calories_from_url
#     ]
#     callbacks = init_langsmith(project=os.getenv("LANGCHAIN_PROJECT", "KusinaBot"))

#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         return_messages=True,
#         input_key="input",
#         output_key="output",
#     )

#     agent = initialize_agent(
#         tools=tools,
#         llm=llm,
#         agent=AgentType.OPENAI_FUNCTIONS,
#         verbose=False,
#         handle_parsing_errors=True,
#         agent_kwargs={"system_message": SYSTEM},
#         memory=memory,
#         callbacks=callbacks or None,   # attaches tracer if enabled
#     )
#     return agent

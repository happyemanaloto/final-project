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

# RAG_LOWCONF_DISTANCE = float(os.getenv("RAG_LOWCONF_DISTANCE", "0.6"))  # FAISS: lower is better
# Confidence & hybrid knobs (tweakable at runtime via env)
RAG_LOWCONF_DISTANCE = float(os.getenv("RAG_LOWCONF_DISTANCE", "0.35"))  # lower is better (distance)
RAG_MIN_MARGIN       = float(os.getenv("RAG_MIN_MARGIN", "0.08"))        # top2 - top1 must exceed this
RAG_TOPK             = int(os.getenv("RAG_TOPK", "6"))
HYBRID_KEYWORD_WEIGHT= float(os.getenv("HYBRID_KEYWORD_WEIGHT", "0.30")) # 0..1 blend

# Optional: use RapidFuzz if available (better), otherwise fallback to difflib
try:
    from rapidfuzz import fuzz as _fuzz
    def _kw_score(a: str, b: str) -> float:
        return _fuzz.partial_ratio(a, b) / 100.0
except Exception:
    from difflib import SequenceMatcher as _SM
    def _kw_score(a: str, b: str) -> float:
        # cheap approximate text overlap
        return _SM(None, a.lower(), b.lower()).ratio()

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
def _remember_turn(agent, user_text: str, ai_text: str) -> None:
    """Best-effort: write the turn into the agent's memory if present."""
    try:
        mem = getattr(agent, "memory", None)
        chat_mem = getattr(mem, "chat_memory", None)
        if chat_mem:
            chat_mem.add_user_message(user_text or "")
            chat_mem.add_ai_message(ai_text or "")
    except Exception:
        # swallow memory errors; don't break the reply path
        pass

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

def _is_low_conf(hits: list[tuple], abs_thresh: float, min_margin: float) -> bool:
    """hits is [(Document, distance), ...] where lower distance is better."""
    if not hits:
        return True
    d0 = float(hits[0][1])
    if len(hits) == 1:
        return d0 > abs_thresh
    d1 = float(hits[1][1])
    margin = d1 - d0                     # bigger margin means a clearer winner
    return (d0 > abs_thresh) and (margin < min_margin)

def _hybrid_rescore(query: str, hits: list[tuple], kw_weight: float) -> list[tuple]:
    """Blend vector similarity with quick keyword/text similarity and re-order."""
    rescored = []
    for doc, dist in hits:
        base_sim = max(0.0, min(1.0, 1.0 - float(dist)))   # distance→similarity
        txt_sim  = _kw_score(query, getattr(doc, "page_content", ""))
        combo    = (1.0 - kw_weight) * base_sim + kw_weight * txt_sim
        rescored.append((combo, doc, dist))
    rescored.sort(key=lambda x: x[0], reverse=True)
    # Return in original shape but re-ordered
    return [(doc, dist) for combo, doc, dist in rescored]

# Add this helper near make_agent_chain():
def _format_recent_history(messages, n=8) -> str:
    if not messages:
        return ""
    chunk = messages[-n:]
    lines = []
    for m in chunk:
        role = m.get("role") or "user"
        txt  = (m.get("content") or "").strip().replace("\n", " ")
        lines.append(f"{role}: {txt}")
    return "\n".join(lines)

def make_agent_chain(llm, session):
    vs = getattr(session, "vectorstore", None)
    if vs is None:
        raise RuntimeError("Session has no vectorstore. Set session.vectorstore = vs when you build it.")

    def run(q: str, lang: str):
        hits = retrieve_with_scores(vs, q, k=6)  # a bit wider pool helps
        top_doc, top_dist = (hits[0] if hits else (Document(page_content=""), 1.0))

        # low-confidence check (your threshold or improved one)
        if (not hits) or (float(top_dist) > RAG_LOWCONF_DISTANCE):
            # (optional) ask targeted clarifying Q here instead of generic
            prefix = (
                "No estoy seguro" if lang.startswith("es")
                else "Ik ben niet zeker" if lang.startswith("nl")
                else "I’m not fully sure"
            )
            return f"{prefix} what you mean by “{q}”. Could you specify protein and time (e.g., chicken, under 30 min)?"

        ctx = "\n\n".join(f"- {d.page_content}" for d, _ in hits[:4])

        # NEW: include conversation history
        history_txt = _format_recent_history(getattr(session, "messages", []), n=8)

        prompt = ChatPromptTemplate.from_messages([
            ("system", STYLE_SYS),
            ("system", "Use the conversation history below to resolve yes/no answers, pronouns, and references."),
            ("system", "Conversation so far:\n{history}"),
            ("system", "Relevant food facts:\n{context}"),
            ("system", "Do not invent facts; answer only using the context."),
            ("system", SYSTEM),
            ("human", "{question}")
        ])
        out = (prompt | llm).invoke({
            "lang": lang,
            "history": history_txt,
            "context": ctx,
            "question": q
        })
        return out.content

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
    IMPORTANT: Every route writes (user_text, ai_answer) into memory so the next
    turn can say "translate that", "yes", "calories for that", etc.
    intent = _translation_intent(user_text)
    """
    answer = None  # we'll set this in a branch and finalize at the end

    # ---- 0) Translation short-circuit (but still remembered) ----
    intent = _translation_intent(user_text)

    if intent:
        lang_code = LANG_ALIASES.get(intent["lang_name"])
        if lang_code:
            payload = intent["payload"]
            if not payload or payload.lower() in {"this", "that", "them", "it", "above", "previous", "previous recipes", "those"}:
                payload = _get_last_ai_text(agent)  # looks in memory; now this will work even after tool paths
            if not payload:
                answer = ensure_reply_language("Paste the text you want me to translate. 🙂", reply_lang)
            else:
                out = translate_text.invoke({"text": payload, "target_lang": lang_code})
                answer = out if isinstance(out, str) else str(out)

    # ---- 1) URL routing ----
    if answer is None:
        url = _extract_url(user_text)

        # 1a) URL + calories
        if url and re.search(CALORIE_TRIGGERS, user_text, flags=re.I):
            res = calories_from_url.invoke({
                "url": url,
                "servings": 1,
                "locale": "EU" if reply_lang.lower() not in {"en-us", "arz"} else "US",
                "target_lang": reply_lang
            })
            answer = res if isinstance(res, str) else str(res)

        # 1b) URL only: ingest and cache hit
        elif url:
            res = ingest_link.invoke({"url": url, "target_lang": reply_lang})
            answer = res if isinstance(res, str) else str(res)

    # ---- 2) Calories short-circuit (no URL) ----
    if answer is None and re.search(CALORIE_TRIGGERS, user_text, flags=re.I):
        hits = _session_get_hits()
        if not hits:
            # If no cached hits (typical after non-recipe ideas), synthesize from last AI
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
                answer = "\n\n".join(lines)
                answer = ensure_reply_language(answer, reply_lang)

    # ---- 3) Normal agent path (with memory) ----
    if answer is None:
        # Option A (keep your JSON directive to minimize changes)
        directive = {"reply_language": reply_lang, "raw_user_text": user_text}
        res = agent.invoke({"input": json.dumps(directive, ensure_ascii=False)})
        answer = res.get("output", str(res)) if isinstance(res, dict) else str(res)

        # Option B (cleaner memory): pass plain text (uncomment if your prompt doesn't require JSON)
        # res = agent.invoke({"input": user_text})
        # answer = res.get("output", str(res)) if isinstance(res, dict) else str(res)

    # ---- 4) Post-process + write to memory (for EVERY route) ----
    try:
        cleaned = remove_json_block(answer)
    except Exception:
        cleaned = answer or ""
    final = ensure_reply_language(cleaned, reply_lang)

    # Write this turn to memory so "translate that / yes / calories for that" works after tool-only routes
    _remember_turn(agent, user_text, final)

    # Keep your existing hit caching, now that memory has the latest AI text
    try:
        _cache_from_last_ai_text(agent, reply_lang)
    except Exception:
        pass

    return final
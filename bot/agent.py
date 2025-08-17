from __future__ import annotations
from .telemetry import init_langsmith

import json, re, os
from typing import Optional, List, Dict

from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# ✅ FIXED: no trailing comma; and import LANG_ALIASES
from .nlp import ensure_reply_language, llm_zero, LANG_ALIASES
from .tools import (
    vector_search, keyword_search, transcribe_media, estimate_nutrition,
    make_shopping_list, create_cookbook, add_feedback, translate_text,
    summarize_video, qa_video, ingest_link, calories_from_url,
    _session_get_hits, _session_set_hits,
)
from .session import SessionMemory

# -------- Heuristics / regex --------
URL_RE = re.compile(r"(https?://\S+)", re.I)
CALORIE_TRIGGERS = r"(?:calorie|calories|kcal|nutrition|macros?|protein|carbs?|fat|kilocal)"

# -------- System Prompt --------
SYSTEM = SystemMessage(content="""You are a cheerful kitchen buddy and nutrition coach.

Voice & style
- Sound human, warm, and encouraging. Use contractions and at most 1–2 emojis total (🍳🥗), not every line.
- Prefer short sentences and compact bullets. Avoid big headings/tables unless asked.
- Keep it actionable: give 2–3 concrete suggestions, each with 2–4 quick steps.
- Weave links inline with the title; don’t dump raw URLs or long source blocks.
- If results look generic (category pages with no real ingredients/steps), skip them.

Language rules
- Always reply in reply_language. Detect media/text language for search, but never change reply_language on your own.
- When replying in a non-English language, prefer ingredients_display if present; otherwise use ingredients.
- If the user says “in <language>” or “translate … to <language>”, translate the relevant text and continue in that language next turns. If they provide only a language (no text), translate the previous assistant message.

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
    Returns {"lang_name": <str>, "payload": <text or ''>} or None.
    Handles:
      - 'translate <X> to|into|in <lang>'
      - 'translate to|into <lang>' (no text)
      - 'can you translate ... in <lang>?'
      - '<lang> please' / 'in <lang> please'
      - 'previous ... in <lang>'
    """
    t = (user_text or "").strip()

    # A) 'translate <X> to|into|in <lang>'
    m = re.search(r'^\s*translate\s+(.*?)\s+(?:to|into|in)\s+([A-Za-z\- ]+)\s*$', t, re.I)
    if m:
        payload = m.group(1).strip()
        lang_name = re.sub(r"\s+", " ", m.group(2).strip().lower())
        return {"lang_name": lang_name, "payload": payload}

    # B) 'translate to|into <lang>' (no explicit text)
    m = re.search(r'^\s*translate(?:\s+(?:to|into))?\s+([A-Za-z\- ]+)\s*$', t, re.I)
    if m:
        lang_name = re.sub(r"\s+", " ", m.group(1).strip().lower())
        return {"lang_name": lang_name, "payload": ""}

    # C) Flexible: 'can you translate ... in <lang>' (anywhere in sentence)
    m = re.search(r'\btranslate\b.*\b(?:to|into|in)\s+([A-Za-z\- ]+)\b', t, re.I)
    if m:
        lang_name = re.sub(r"\s+", " ", m.group(1).strip().lower())
        return {"lang_name": lang_name, "payload": ""}

    # D) '<lang> please' / 'in <lang> please'
    m = re.search(r'^\s*(?:in\s+)?([A-Za-z\- ]+)\s+please\s*$', t, re.I)
    if m:
        lang_name = re.sub(r"\s+", " ", m.group(1).strip().lower())
        return {"lang_name": lang_name, "payload": ""}

    # E) 'previous ... in <lang>'
    m = re.search(r'^\s*previous(?:\s+\w+)*\s+in\s+([A-Za-z\- ]+)\s*$', t, re.I)
    if m:
        lang_name = re.sub(r"\s+", " ", m.group(1).strip().lower())
        return {"lang_name": lang_name, "payload": ""}

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
    intent = _translation_intent(user_text)
    if intent:
        lang_code = LANG_ALIASES.get(intent["lang_name"])
        if not lang_code:
            return ensure_reply_language("Which language should I translate to?", reply_lang)

        payload = intent["payload"]
        # If payload missing or pronoun, translate the last assistant message
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
                    lines.append(f"**{title}** — per serving:\n{est_txt}")
                except Exception:
                    continue

            if lines:
                return ensure_reply_language("\n\n".join(lines), reply_lang)

    # 3) Normal agent path (with memory)
    directive = {"reply_language": reply_lang, "raw_user_text": user_text}
    res = agent.invoke({"input": json.dumps(directive, ensure_ascii=False)})
    answer = res.get("output", str(res)) if isinstance(res, dict) else str(res)
    return ensure_reply_language(answer, reply_lang)

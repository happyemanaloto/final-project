"""
Chef & Nutritionist Chatbot (Agent + Tools)

Datasets supported:
- YouTube JSONs at: data/recipes/*.json (from your pipeline)
- Wikibooks JSONs at: data/open_wikibooks_toc/*.json (or data/open_wikibooks/*.json)

Core behaviors (MVP):
- Detect media links (YouTube / audio file) -> transcribe -> proceed
- Translate request to English for search; reply in original language
- Extract structured preferences -> search local recipes -> return top 3
- Show links and (for YouTube) thumbnails
- Offer alternatives, tips, and collect feedback
- Create a simple "cookbook" (Markdown) the user can download

Requirements (install in your kusinaenv):
pip install langchain langchain-openai pydantic python-dotenv youtube-transcript-api yt-dlp openai-whisper rapidfuzz langdetect

Run:
python src/backend/chef_agent.py --data-youtube data/recipes --data-wikibooks data/open_wikibooks_toc

Type your question at the prompt (in any language). Paste a YouTube link to trigger transcription.
"""

from __future__ import annotations
import os, json, re, argparse, datetime as dt
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv(filename=".env", usecwd=True))

# --- LLMs / LangChain
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableLambda

# --- Utilities
from langdetect import detect as lang_detect
from rapidfuzz import process, fuzz

# --- Transcription stack
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from yt_dlp import YoutubeDL
import whisper


# =========================
# Data loading / indexing
# =========================
class RecipeDoc(BaseModel):
    id: str
    title: str
    url: str
    source: str  # "youtube" | "wikibooks"
    image_url: Optional[str] = None
    ingredients: List[str] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list)
    cuisine: Optional[str] = None
    cook_time_minutes: Optional[int] = None
    extras: Dict[str, Any] = Field(default_factory=dict)

    @property
    def search_text(self) -> str:
        # Keep it compact; transcripts are huge—skip them here
        parts = [
            self.title or "",
            " ".join(self.ingredients or []),
            self.cuisine or "",
        ]
        return " ".join(p for p in parts if p).lower()


def load_youtube_dir(d: Path) -> List[RecipeDoc]:
    docs = []
    for p in sorted(d.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            rec = obj.get("recipe", {})
            vid = obj.get("video_id") or os.path.splitext(p.name)[0]
            url = obj.get("url") or f"https://www.youtube.com/watch?v={vid}"
            img = f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None
            docs.append(RecipeDoc(
                id=f"yt:{vid}",
                title=rec.get("title") or obj.get("video_title") or p.stem,
                url=url,
                source="youtube",
                image_url=img,
                ingredients=rec.get("ingredients") or [],
                steps=rec.get("steps") or [],
                cuisine=rec.get("cuisine"),
                cook_time_minutes=rec.get("cook_time_minutes"),
                extras={"channel": obj.get("channel")}
            ))
        except Exception:
            continue
    return docs


def load_wikibooks_dir(d: Path) -> List[RecipeDoc]:
    docs = []
    for p in sorted(d.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            rec = obj.get("recipe", {})
            url = obj.get("source_url") or obj.get("url") or ""
            docs.append(RecipeDoc(
                id=f"wb:{p.stem}",
                title=rec.get("title") or p.stem,
                url=url,
                source="wikibooks",
                image_url=None,  # keep it clean/legal
                ingredients=rec.get("ingredients") or [],
                steps=rec.get("steps") or [],
                cuisine=rec.get("cuisine"),
                cook_time_minutes=rec.get("cook_time_minutes"),
                extras={"license": obj.get("license"), "attribution": obj.get("attribution")}
            ))
        except Exception:
            continue
    return docs


class RecipeIndex:
    def __init__(self, docs: List[RecipeDoc]):
        self.docs = docs
        self.corpus = [d.search_text for d in docs]

    def search(self, query: str, top_k: int = 3, time_limit: Optional[int] = None) -> List[RecipeDoc]:
        """Simple fuzzy search with a soft penalty if cook_time > time_limit."""
        if not self.docs:
            return []
        query = (query or "").lower()
        scored = process.extract(
            query,
            self.corpus,
            scorer=fuzz.token_set_ratio,
            score_cutoff=0
        )
        # scored: list of tuples (match_text, score, index)
        items = []
        for _, score, idx in scored:
            d = self.docs[idx]
            penalty = 0
            if time_limit and d.cook_time_minutes and d.cook_time_minutes > time_limit:
                # penalize long recipes if user is short on time
                overflow = d.cook_time_minutes - time_limit
                penalty = min(30, overflow // 5)  # mild penalty up to 30
            items.append((score - penalty, d))
        items.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in items[:top_k]]


# =========================
# LLM helpers
# =========================
LLM_MODEL = os.getenv("CHEF_BOT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("CHEF_EMBED_MODEL", "text-embedding-3-small")  # reserved for future

def llm_zero():
    return ChatOpenAI(model=LLM_MODEL, temperature=0)

def detect_language(text: str) -> str:
    try:
        return lang_detect(text)
    except Exception:
        return "en"

class Prefs(BaseModel):
    language: Optional[str] = None
    cuisine: Optional[str] = None
    part_of_meal: Optional[str] = None     # appetizer, mains, dessert, sides, refreshments, pairings
    part_of_day: Optional[str] = None      # breakfast, lunch, dinner, snacks
    heavy_or_light: Optional[str] = None   # heavy | light
    time_minutes: Optional[int] = None
    difficulty: Optional[str] = None       # easy/medium/hard
    budget: Optional[str] = None           # low/medium/high
    available_ingredients: Optional[str] = None  # pantry/local/imported
    servings: Optional[int] = None
    allergens: Optional[List[str]] = None
    goals: Optional[List[str]] = None      # health objectives
    free_text: Optional[str] = None        # anything else

PREFS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Extract user cooking preferences as JSON for fields: "
     "language,cuisine,part_of_meal,part_of_day,heavy_or_light,time_minutes,"
     "difficulty,budget,available_ingredients,servings,allergens,goals,free_text. "
     "If not present, set null. Respond with JSON only."),
    ("human", "{text}")
])

def extract_prefs(text_en: str) -> Prefs:
    llm = llm_zero()
    out = (PREFS_PROMPT | llm).invoke({"text": text_en})
    try:
        data = json.loads(out.content)
        return Prefs(**data)
    except Exception:
        # very defensive fallback
        return Prefs(free_text=text_en)


TRANS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Translate to English. Return only the translation."),
    ("human", "{text}")
])

def translate_to_english(text: str) -> str:
    if not text:
        return text
    # Skip if already English (best effort)
    try:
        if detect_language(text) == "en":
            return text
    except Exception:
        pass
    llm = llm_zero()
    out = (TRANS_PROMPT | llm).invoke({"text": text})
    return out.content.strip()


# =========================
# Transcription tool
# =========================
def extract_video_id(url: str) -> Optional[str]:
    m = re.search(r"(?:v=|\.be/|/shorts/|/embed/)([\w-]{11})", url)
    return m.group(1) if m else None

def transcript_via_api(video_id: str) -> Optional[str]:
    try:
        segs = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        return " ".join(s["text"] for s in segs).strip()
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception:
        return None

_WHISPER = None
def whisper_model(size: str = "base"):
    global _WHISPER
    if _WHISPER is None:
        _WHISPER = whisper.load_model(size)
    return _WHISPER

def download_audio(url: str) -> Path:
    outdir = Path("data/tmp_audio"); outdir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(outdir / "%(id)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}],
        "quiet": True
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    vid = info.get("id")
    return outdir / f"{vid}.wav"

def transcribe_media(url_or_path: str) -> str:
    # YouTube fast path
    if "youtube.com" in url_or_path or "youtu.be" in url_or_path:
        vid = extract_video_id(url_or_path)
        if vid:
            t = transcript_via_api(vid)
            if t:
                return t
    # Fallback: local/remote audio -> Whisper
    path = Path(url_or_path)
    wav = path if path.exists() else download_audio(url_or_path)
    res = whisper_model().transcribe(str(wav), fp16=False)
    return (res.get("text") or "").strip()


# =========================
# Agent Tools
# =========================
INDEX: Optional[RecipeIndex] = None
FEEDBACK_PATH = Path("data/feedback.jsonl")
EXPORTS_DIR = Path("exports")

class SearchArgs(BaseModel):
    preferences_json: str
    top_k: int = 3

@tool("search_recipes", args_schema=SearchArgs, return_direct=False)
def tool_search_recipes(preferences_json: str, top_k: int = 3) -> str:
    """
    Search the local recipe database for the closest matches.
    preferences_json: JSON string with fields like cuisine, time_minutes, free_text...
    Returns JSON with a list of hits [{id,title,url,image_url,source,score_hint}]
    """
    global INDEX
    if INDEX is None or not INDEX.docs:
        return json.dumps({"hits": [], "note": "index is empty"})
    try:
        prefs = json.loads(preferences_json)
    except Exception:
        prefs = {}
    # Build a lightweight query text
    tokens = []
    for k in ["cuisine","part_of_meal","part_of_day","heavy_or_light","difficulty","budget","available_ingredients","free_text"]:
        v = prefs.get(k)
        if isinstance(v, list):
            tokens += v
        elif isinstance(v, str) and v:
            tokens.append(v)
    if prefs.get("servings"): tokens.append(f"servings:{prefs['servings']}")
    query = " ".join(tokens)
    time_limit = prefs.get("time_minutes")
    hits = INDEX.search(query=query, top_k=top_k, time_limit=time_limit)
    out = []
    for d in hits:
        out.append({
            "id": d.id,
            "title": d.title,
            "url": d.url,
            "image_url": d.image_url,
            "source": d.source,
        })
    return json.dumps({"hits": out})

class FeedbackArgs(BaseModel):
    recipe_id: str
    feedback_text: str
    user_lang: Optional[str] = None

@tool("add_feedback", args_schema=FeedbackArgs, return_direct=False)
def tool_add_feedback(recipe_id: str, feedback_text: str, user_lang: Optional[str] = None) -> str:
    """
    Append user feedback to data/feedback.jsonl. Returns 'ok'.
    """
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "recipe_id": recipe_id,
        "feedback": feedback_text,
        "lang": user_lang or "unknown",
    }
    with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return json.dumps({"status":"ok"})

class CookbookArgs(BaseModel):
    recipe_ids: List[str]
    language: Optional[str] = "en"
    title: Optional[str] = "My Personal Cookbook"

@tool("create_cookbook", args_schema=CookbookArgs, return_direct=False)
def tool_create_cookbook(recipe_ids: List[str], language: Optional[str] = "en", title: Optional[str] = "My Personal Cookbook") -> str:
    """
    Create a simple Markdown cookbook from selected recipes. Returns a file path.
    """
    global INDEX
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    idx_map = {d.id: d for d in INDEX.docs} if INDEX else {}
    lines = [f"# {title}\n"]
    for rid in recipe_ids:
        d = idx_map.get(rid)
        if not d:
            continue
        lines.append(f"## {d.title}\n")
        lines.append(f"[Link]({d.url})  \nSource: {d.source}\n")
        if d.ingredients:
            lines.append("### Ingredients")
            for ing in d.ingredients:
                lines.append(f"- {ing}")
        if d.steps:
            lines.append("### Steps")
            for i, step in enumerate(d.steps, 1):
                lines.append(f"{i}. {step}")
        lines.append("")  # blank line
    out_path = EXPORTS_DIR / f"cookbook_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return json.dumps({"path": str(out_path)})


# =========================
# Agent System Prompt
# =========================
SYSTEM = SystemMessage(content=(
"You are a chef and nutritionist chatbot who assists people worldwide, "
"adapting to their cravings, time, skill, price sensitivity, health goals, and budget. "
"Workflow:\n"
"1) If the user provided audio/video, call transcribe first. If text, continue.\n"
"2) Translate the user request to English for search (but reply in user's original language).\n"
"3) Extract preferences (language, cuisine, part_of_meal, part_of_day, heavy_or_light, time_minutes, "
"difficulty, budget, available_ingredients, servings, allergens, goals, free_text).\n"
"4) Call search_recipes to get top 3.\n"
"   - If request too sparse, still search; prefer healthy, practical, easy-to-source ingredients.\n"
"5) Present results with links and images (YouTube thumbnails if available).\n"
"6) Ask if they want alternatives.\n"
"7) Provide efficiency, health, and price tips.\n"
"8) Offer to guide step-by-step sourcing of ingredients and tools.\n"
"9) Ask for feedback; if provided, call add_feedback.\n"
"10) If asked, call create_cookbook to export selected recipes.\n"
"Keep answers concise and friendly. Use the tools when appropriate."
))

# We’ll keep translation + pref extraction as invisible helper steps (outside tools),
# then the agent uses tools for search/feedback/cookbook.
def build_agent():
    tools = [tool_search_recipes, tool_add_feedback, tool_create_cookbook]
    llm = llm_zero()
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=False,
        handle_parsing_errors=True,
        agent_kwargs={"system_message": SYSTEM},
    )
    return agent


# =========================
# Chat loop
# =========================
MEDIA_RE = re.compile(r"(https?://\S+)", re.I)

def classify_media(text: str) -> Optional[str]:
    m = MEDIA_RE.search(text or "")
    if not m:
        return None
    url = m.group(1)
    if any(x in url for x in ["youtube.com","youtu.be",".mp3",".wav",".m4a",".mp4",".mov",".webm"]):
        return url
    return None

def format_hits(hits: List[Dict[str, Any]], reply_lang: str) -> str:
    # Simple textual rendering; your UI can make this pretty.
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"{i}. {h['title']}")
        lines.append(f"   {h['url']}")
        if h.get("image_url"):
            lines.append(f"   [thumbnail] {h['image_url']}")
    # Add a localized prompt to ask alternatives
    if reply_lang.startswith("en"):
        lines.append("\nWould you like alternatives or a step-by-step plan?")
    else:
        # ask via LLM quick translate
        q = {"en": "Would you like alternatives or a step-by-step plan?"}
        ask = (TRANS_PROMPT | llm_zero()).invoke({"text": q["en"]}).content.strip()
        lines.append("\n" + ask)
    return "\n".join(lines)

def chat_once(agent, user_text: str) -> str:
    # 1) Media?
    media_url = classify_media(user_text)
    transcript_text = None
    if media_url:
        try:
            transcript_text = transcribe_media(media_url)
        except Exception as e:
            transcript_text = None

    # 2) Figure original language & translation for search
    txt_for_understanding = transcript_text or user_text
    user_lang = detect_language(txt_for_understanding)
    text_en = translate_to_english(txt_for_understanding)

    # 3) Extract preferences (no tool call; light LLM helper)
    prefs = extract_prefs(text_en)
    if not prefs.language:
        prefs.language = user_lang

    # 4) Agent search via tool
    tool_input = json.dumps(prefs.model_dump())
    search_res = agent.invoke({"input": f"Search with these preferences: {tool_input}"})
    # The agent will call search_recipes internally; we parse tool output from the final text.

    # Try to pull hits JSON if the agent exposed it; else do a direct call safety-net:
    hits = []
    try:
        # Heuristic: look for a JSON block in the final text
        m = re.search(r"\{[\s\S]*\"hits\"\s*:\s*\[[\s\S]*\}\s*\}", search_res["output"])
        if m:
            hits = json.loads(m.group(0)).get("hits", [])
    except Exception:
        pass
    if not hits:
        # Safety net (direct tool call)
        raw = tool_search_recipes.invoke({"preferences_json": tool_input, "top_k": 3})
        hits = json.loads(raw).get("hits", [])

    # 5) Compose answer in original language
    # Build a short English summary, then translate back.
    eng_summary_lines = []
    if transcript_text:
        eng_summary_lines.append("I transcribed your media and searched accordingly.")
    if not hits:
        eng_summary_lines.append("I couldn't find strong matches; here are some healthy, practical picks:")
        # Try a generic search by health keywords
        fallback = json.loads(tool_search_recipes.invoke(
            {"preferences_json": json.dumps({"free_text":"healthy quick easy"}), "top_k": 3}
        )).get("hits", [])
        hits = fallback

    # Now render hits
    eng_summary_lines.append("Here are the top 3 matches:")
    for i, h in enumerate(hits, 1):
        eng_summary_lines.append(f"{i}) {h['title']} — {h['url']}" + (f" [thumb: {h['image_url']}]" if h.get("image_url") else ""))

    # Extra guidance line
    eng_summary_lines.append(
        "I can suggest alternatives, give efficiency/health/budget tips, guide you step-by-step, "
        "or create a downloadable cookbook with your chosen recipes."
    )
    eng_reply_en = "\n".join(eng_summary_lines)

    # Translate back if needed
    if user_lang != "en":
        back_prompt = ChatPromptTemplate.from_messages([
            ("system", "Translate the following assistant reply into the user's language. Keep links intact."),
            ("human", "{text}")
        ])
        eng_reply_local = (back_prompt | llm_zero()).invoke({"text": eng_reply_en}).content.strip()
        return eng_reply_local
    else:
        return eng_reply_en


# =========================
# Main
# =========================
def load_index(youtube_dir: str, wikibooks_dir: str) -> RecipeIndex:
    docs: List[RecipeDoc] = []
    if youtube_dir and Path(youtube_dir).exists():
        docs += load_youtube_dir(Path(youtube_dir))
    if wikibooks_dir and Path(wikibooks_dir).exists():
        docs += load_wikibooks_dir(Path(wikibooks_dir))
    return RecipeIndex(docs)

def main():
    ap = argparse.ArgumentParser(description="Chef & Nutritionist Chatbot (Agent + Tools)")
    ap.add_argument("--data-youtube", type=str, default="data/recipes")
    ap.add_argument("--data-wikibooks", type=str, default="data/open_wikibooks_toc")
    args = ap.parse_args()

    global INDEX
    INDEX = load_index(args.data_youtube, args.data_wikibooks)
    print(f"Loaded {len(INDEX.docs)} recipes.")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Put it in your .env or environment.")

    agent = build_agent()
    print("\nChef bot ready. Type your request (paste YouTube links to transcribe). Ctrl+C to exit.\n")
    while True:
        try:
            user = input("You: ").strip()
            if not user:
                continue
            answer = chat_once(agent, user)
            print("\nAssistant:\n" + answer + "\n")
        except KeyboardInterrupt:
            print("\nBye!")
            break

if __name__ == "__main__":
    main()

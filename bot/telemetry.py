# bot/telemetry.py
import os

def langsmith_enabled() -> bool:
    return (os.getenv("LANGSMITH_ENABLE", "0") or "").strip() in {"1", "true", "yes", "on"}

def init_langsmith(project: str | None = None):
    """
    Sets env vars programmatically if LANGSMITH_ENABLE is on.
    Safe to call multiple times. Returns a list of LangChain callbacks (or []).
    """
    if not langsmith_enabled():
        return []

    # Ensure required vars are set. If LANGCHAIN_API_KEY isn't present, we just no-op.
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("[langsmith] ENABLED but LANGCHAIN_API_KEY missing — tracing will be skipped.")
        return []

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    if project and not os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = project

    # Optional: return a tracer callback so you can pass it into LLMs/agents/tools
    try:
        from langchain.callbacks.tracers import LangChainTracer
        tracer = LangChainTracer(project_name=os.getenv("LANGCHAIN_PROJECT", project or "KusinaBot"))
        return [tracer]
    except Exception:
        # LangChain ≥0.2 works just by env vars; callbacks are optional.
        return []

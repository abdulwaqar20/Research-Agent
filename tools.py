import os
import inspect
from dotenv import load_dotenv
from ddgs import DDGS
import trafilatura

load_dotenv()  # so this module works even when imported standalone

_client = None


def _get_client():
    """Lazy init — only web_search/fetch_page use this module, and they
    don't need a Groq key at all. Only summarise() needs it."""
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def call_tool(fn, args: dict):
    """Drop any argument the model invented that the function doesn't accept,
    instead of crashing on an unexpected keyword."""
    sig = inspect.signature(fn)
    accepted = {k: v for k, v in args.items() if k in sig.parameters}
    return fn(**accepted)


def web_search(query: str, max_results: int = 5) -> dict:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return {"ok": False, "error": f"no results for: {query}"}
        return {
            "ok": True,
            "results": [
                {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
                for r in results
                if r.get("href")
            ],
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def fetch_page(url: str, max_chars: int = 3000) -> dict:
    try:
        if not url:
            return {"ok": False, "error": "empty url"}
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return {"ok": False, "error": f"could not download {url}"}
        text = trafilatura.extract(downloaded)
        if not text or len(text.strip()) < 50:
            return {"ok": False, "error": f"no usable text extracted from {url}"}
        return {"ok": True, "url": url, "text": text[:max_chars]}
    except Exception as e:
        return {"ok": False, "error": f"fetch failed for {url}: {e}"}


def summarise(url: str, text: str, question: str) -> dict:
    try:
        if not text or len(text.strip()) < 20:
            return {"ok": False, "error": "nothing to summarise"}
        resp = _get_client().chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{
                "role": "user",
                "content": f"Summarise this in 3-5 bullet points, focused on: {question}\n\n{text[:8000]}",
            }],
        )
        return {"ok": True, "summary": resp.choices[0].message.content}
    except Exception as e:
        return {"ok": False, "error": str(e)}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for pages relevant to a query. Returns titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Download a specific URL (found via web_search) and extract its main readable text.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "URL to fetch"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarise",
            "description": "Condense a fetched page's text into bullet points relevant to the research question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL this text came from"},
                    "text": {"type": "string", "description": "raw text to summarise"},
                },
                "required": ["url", "text"],
            },
        },
    },
]

DISPATCH = {"web_search": web_search, "fetch_page": fetch_page, "summarise": summarise}
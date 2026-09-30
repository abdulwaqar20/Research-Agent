import os
import sys
import json
from dotenv import load_dotenv
from groq import Groq

# Windows consoles default to cp1252, which can't encode characters like
# non-breaking hyphens (U+2011) that the model sometimes outputs. Force
# UTF-8 with a safe fallback so printing never crashes the run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

from tools import TOOL_SCHEMAS, DISPATCH, call_tool

MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a research agent. Use web_search, fetch_page, and summarise
to answer the user's question. You must fetch at least one real page before answering.
Every fetch_page result includes a "source_id" like S1, S2 — use exactly that id when
citing that page. Tag every claim with its [S#], and end your answer with a Sources list
mapping each [S#] to its URL. Do not invent a citation id that wasn't given to you.
If a tool fails or an URL was already fetched, do not repeat the identical call — try a
different query or URL instead.

Critical rule: only state facts that are directly supported by the fetched page content
in this conversation. Do not add background knowledge, names, dates, or figures that did
not come from a fetch_page result, even if you're confident they're true. If you don't
have enough fetched material to cover part of the question, say so explicitly in that
part of the answer instead of filling the gap from memory."""

STEP_BUDGET = int(os.environ.get("AGENT_STEP_BUDGET", 6))


def safe_chat(messages, tools=None, tool_choice="auto", retries=1):
    """Groq's gpt-oss-120b occasionally emits output that fails to parse as a
    valid tool call. Retry once without tools forced, then give up gracefully
    instead of crashing the whole agent."""
    kwargs = dict(model=MODEL, messages=messages)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice

    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message, None
        except Exception as e:
            print(f"    [LLM call failed, attempt {attempt + 1}/{retries + 1}: {e}]")
            if attempt < retries:
                continue
            return None, str(e)


def run(question: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps_used = 0
    sources = {}  # url -> "S1", "S2", ...
    fetched_urls = set()

    while steps_used < STEP_BUDGET:
        msg, err = safe_chat(messages, tools=TOOL_SCHEMAS, tool_choice="auto")

        if msg is None:
            # LLM call itself failed twice in a row — stop cleanly instead of crashing
            print(f"\n--- LLM UNAVAILABLE, STOPPING EARLY: {err} ---")
            _print_sources(sources)
            return {"answer": None, "sources": sources, "steps_used": steps_used, "error": err}

        if not msg.tool_calls:
            print("\n--- FINAL ANSWER ---")
            print(msg.content)
            _print_sources(sources)
            return {"answer": msg.content, "sources": sources, "steps_used": steps_used}

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })

        for tc in msg.tool_calls:
            if steps_used >= STEP_BUDGET:
                break

            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            if name == "summarise":
                args["question"] = question

            # dedup check happens BEFORE the real call, so we never waste a
            # fetch on a URL we already have
            if name == "fetch_page" and args.get("url") in fetched_urls:
                result = {
                    "ok": False,
                    "error": "already fetched this URL — reuse the earlier content instead of refetching",
                }
            else:
                fn = DISPATCH.get(name)
                if fn is None:
                    result = {"ok": False, "error": f"unknown tool {name}"}
                else:
                    result = call_tool(fn, args)

            steps_used += 1

            if name == "fetch_page" and result.get("ok"):
                fetched_urls.add(result["url"])
                if result["url"] not in sources:
                    sources[result["url"]] = f"S{len(sources) + 1}"
                result = {**result, "source_id": sources[result["url"]]}

            status = "OK" if result.get("ok") else "FAIL"
            print(f"[{steps_used}/{STEP_BUDGET}] {name}({args}) -> {status}")
            if not result.get("ok"):
                print(f"    error: {result.get('error')}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result)[:2000],  # tight cap, low TPM limit on free tier
            })

    # budget used up before the model volunteered a final answer -> force one
    messages.append({
        "role": "user",
        "content": (
            "Tool budget used up. Using ONLY the content already fetched in this "
            "conversation, answer now, citing [S#] for each claim and listing sources. "
            "Do not add any fact, name, or figure that wasn't in a fetched page. For any "
            "part of the question you can't cover from what was fetched, say plainly "
            "'not covered by the sources gathered' instead of guessing."
        ),
    })
    msg, err = safe_chat(messages, tools=None)
    if msg is None:
        print(f"\n--- LLM UNAVAILABLE FOR FINAL ANSWER: {err} ---")
        _print_sources(sources)
        return {"answer": None, "sources": sources, "steps_used": steps_used, "error": err}

    answer = msg.content
    print("\n--- FINAL ANSWER (forced, step budget exhausted) ---")
    print(answer)
    _print_sources(sources)
    return {"answer": answer, "sources": sources, "steps_used": steps_used}


def _print_sources(sources: dict):
    print("\n--- SOURCES TRACKED ---")
    if not sources:
        print("(none fetched)")
    for url, sid in sources.items():
        print(f"{sid}: {url}")


if __name__ == "__main__":
    args = sys.argv[1:]
    save_path = None
    if "--save" in args:
        idx = args.index("--save")
        save_path = args[idx + 1]
        del args[idx:idx + 2]

    # any remaining args are still treated as the question (for scripting/automation),
    # but if none were given, ask interactively instead of using a hardcoded default
    q = " ".join(args).strip()
    if not q:
        q = input("Enter your research question: ").strip()
    if not q:
        print("No question entered. Exiting.")
        sys.exit(1)

    if save_path:
        # write our own UTF-8 file instead of relying on shell redirection,
        # which on Windows PowerShell mangles non-ASCII characters (mojibake)
        import io
        buf = io.StringIO()
        real_stdout = sys.stdout
        sys.stdout = buf
        try:
            run(q)
        finally:
            sys.stdout = real_stdout
        text = buf.getvalue()
        print(text)  # still show it in the terminal
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n[saved clean UTF-8 output to {save_path}]")
    else:
        run(q)
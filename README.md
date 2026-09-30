# Tool-Using Research Agent

Answers a research question by calling tools (web search, page fetch, summarise) and
deciding itself what to call next, with a hard step budget so it can't loop forever.
Uses Groq (`openai/gpt-oss-120b`) for the LLM calls.

## Architecture

`agent.py` runs a plain while-loop:
1. Send the question + system prompt to the model with 3 tool definitions.
2. If the model requests a tool, run it and feed the result back as a `tool` message.
3. Every successful `fetch_page` gets a source id (S1, S2...) the first time its URL is
   seen. A repeated request for an already-fetched URL is rejected before any real
   network call runs, so it doesn't waste a step.
4. `STEP_BUDGET` (default 6, via `AGENT_STEP_BUDGET`) hard-caps total tool calls — the
   loop counts them itself, so a model that won't stop gets cut off and forced to answer
   with whatever it already fetched.
5. The system prompt forbids adding any fact not present in a fetched page; if the
   budget runs out with gaps still uncovered, the model must say so instead of guessing.
6. Every tool returns `{"ok": bool, ...}` instead of raising. The LLM call itself is
   wrapped in a one-retry `safe_chat()`, since `gpt-oss-120b` occasionally emits a
   malformed or hallucinated tool call — this stops that from crashing the agent.
7. Output is written as UTF-8 directly by the script (`--save path`) rather than via
   shell redirection, which mangles non-ASCII characters on Windows PowerShell.

## Tools
- `web_search(query)` — DuckDuckGo search via `ddgs`
- `fetch_page(url)` — downloads a URL, extracts readable text via `trafilatura`
- `summarise(url, text)` — condenses fetched text with the same Groq model

## Setup
\`\`\`bash
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY
\`\`\`

## Run
\`\`\`bash
python agent.py "your question" --save examples/run.txt
\`\`\`

## Example runs
Full transcripts in `examples/run1.txt`, `run2.txt`, `run3.txt`.

**Run 1 — "What caused the 2010 flash crash?"**
Dedup correctly rejected a repeated fetch of the same URL; the model then tried a
variant URL instead. Step budget forced a final answer citing both fetched pages
(S1, S2), quoting the Wikipedia explanation directly rather than inventing detail.

**Run 2 — "What is the current status of nuclear fusion energy commercialisation?"**
Two fetches failed outright (a dead link, an unparseable PDF). Step budget forced a
final answer built entirely from the one page that succeeded (S1) — no outside facts
added, unlike an earlier draft of this prompt which had hallucinated company names.

**Run 3 — "How does CRISPR gene editing work?"**
Clean run: fetched one page, used `summarise` to condense it, and the model stopped on
its own once satisfied — no forced cutoff needed. Answer cites S1 throughout.

## Known limitations
- PDFs and some JS-heavy pages fail to fetch via `trafilatura` (seen in runs 1 and 2).
- Groq's free tier has a low tokens/minute limit, hence `fetch_page`'s 3000-char cap.
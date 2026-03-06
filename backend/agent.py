"""
agent.py — Recursive LLM agent logic.

RootAgent  → deepseek/deepseek-chat-v3-0324  (reasoning)
SubAgent   → meta-llama/llama-4-scout (summarisation)

Context-as-Variable paradigm: the LLM writes Python code that calls
sql_query(), vector_search(), read_file() rather than receiving raw data.
"""

import os, json, re, asyncio
from typing import Any, Callable, Optional

import httpx
from dotenv import load_dotenv

from backend.sandbox import Sandbox

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

ROOT_MODEL = "deepseek/deepseek-chat-v3-0324"
SUB_MODEL = "meta-llama/llama-4-scout"

# Fallback chains — if the primary model is down, try these in order
ROOT_FALLBACKS = [
    "meta-llama/llama-4-scout",
    "meta-llama/llama-4-maverick",
    "nvidia/llama-3.1-nemotron-nano-8b-v1",
]
SUB_FALLBACKS = [
    "meta-llama/llama-4-maverick",
    "nvidia/llama-3.1-nemotron-nano-8b-v1",
]

MAX_DEPTH = 4          # hard limit on recursion depth
MAX_RETRIES = 3        # self-correction attempts per step
MAX_STEPS = 8          # max sandbox executions per agent call
LLM_RETRY_ATTEMPTS = 3 # retries per model on server errors

SYSTEM_PROMPT = """\
You are a Data Investigator agent. You have access to three tools inside a Python sandbox:

1. sql_query(query: str) -> list[dict]  —  Run a SQL query on a SQLite database with tables:
   • suppliers (id, name, reliability_score, region)
   • shipments (id, supplier_id, item, status, cost, date)
     - status values are UPPERCASE: 'DELAYED', 'DELIVERED', 'CANCELLED'

2. vector_search(query: str, n: int = 3) -> list[dict]  —  Semantic search over supplier agreement PDFs.

3. read_file(path: str = "") -> str  —  Read local shipping log file (default: shipping_logs.txt).

RULES:
- IMPORTANT: Always explore available data FIRST before filtering. For example, run
  sql_query("SELECT * FROM suppliers") to see all supplier names before searching
  for a specific one. Never guess or assume exact names.
- Write Python code to investigate the user's query. Assign your final answer to a variable called `_result_`.
- Cross-reference data across ALL THREE sources (SQL, vector, file) when relevant.
- You can call multiple tools in one code block.
- If a result set is very large (>500 chars), you should call `recursive_agent_call(context, depth)` to spawn a sub-agent to summarize it.
- Respond ONLY with a JSON object:
  {"thought": "your reasoning", "code": "python code to execute"}
- If you have enough information to answer, set _result_ to a string with your final, well-formatted answer.
- Do NOT use any imports or functions outside the provided sandbox.
"""

SUMMARIZE_PROMPT = """\
You are a summarization sub-agent. Summarize the following data concisely, \
focusing on the key findings relevant to the user's original query.

ORIGINAL QUERY: {query}

DATA TO SUMMARIZE:
{data}

Respond with a concise summary as plain text (no JSON wrapping).
"""

FORCE_SUMMARIZE_PROMPT = """\
You have reached the maximum recursion depth. You MUST summarize all \
gathered context and produce a final answer now. Do NOT request any \
more data. Assign your final answer to `_result_`.

CONTEXT SO FAR:
{context}

Respond ONLY with: {{"thought": "final summary", "code": "_result_ = '...your answer...'"}}
"""


async def _call_llm(
    model: str,
    messages: list[dict],
    log_callback: Optional[Callable] = None,
) -> str:
    """Call OpenRouter chat completion with retry + fallback."""
    # Build model chain: primary model + fallbacks
    fallbacks = ROOT_FALLBACKS if model == ROOT_MODEL else SUB_FALLBACKS
    models_to_try = [model] + [fb for fb in fallbacks if fb != model]

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "RLM Data Investigator",
    }

    last_error = None

    for current_model in models_to_try:
        payload = {
            "model": current_model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 2048,
        }

        for attempt in range(LLM_RETRY_ATTEMPTS):
            try:
                if log_callback:
                    retry_tag = f" (retry {attempt + 1})" if attempt > 0 else ""
                    log_callback({
                        "type": "llm_call",
                        "content": f"Calling {current_model}...{retry_tag}",
                        "meta": {"model": current_model, "attempt": attempt},
                    })

                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.post(OPENROUTER_URL, json=payload, headers=headers)

                    if resp.status_code >= 500:
                        # Server error — retry after backoff
                        wait = 2 ** (attempt + 1)
                        if log_callback:
                            log_callback({
                                "type": "system",
                                "content": f"⚠️ {current_model} returned {resp.status_code}, retrying in {wait}s...",
                                "meta": {},
                            })
                        await asyncio.sleep(wait)
                        last_error = f"{resp.status_code} from {current_model}"
                        continue

                    resp.raise_for_status()
                    data = resp.json()

                content = data["choices"][0]["message"]["content"]

                # Strip <think>...</think> tags from deepseek responses
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

                if log_callback:
                    log_callback({
                        "type": "llm_response",
                        "content": content[:500],
                        "meta": {"model": current_model},
                    })

                return content

            except httpx.HTTPStatusError as e:
                last_error = str(e)
                if e.response.status_code >= 500:
                    wait = 2 ** (attempt + 1)
                    if log_callback:
                        log_callback({
                            "type": "system",
                            "content": f"⚠️ {current_model} error: {e.response.status_code}, retrying in {wait}s...",
                            "meta": {},
                        })
                    await asyncio.sleep(wait)
                    continue
                raise  # 4xx errors are not retryable
            except Exception as e:
                last_error = str(e)
                break  # Non-HTTP error, try next model

        # This model failed all retries — try the next fallback
        if log_callback:
            log_callback({
                "type": "system",
                "content": f"🔄 {current_model} unavailable, trying fallback...",
                "meta": {},
            })

    raise RuntimeError(f"All models failed. Last error: {last_error}")



def _parse_agent_response(raw: str) -> dict:
    """Extract JSON {thought, code} from LLM response, tolerating markdown fences."""
    # Try to find JSON block
    json_match = re.search(r'\{[\s\S]*?"thought"[\s\S]*?"code"[\s\S]*?\}', raw)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Try stripping markdown code fences
    cleaned = re.sub(r'```(?:json)?\s*', '', raw)
    cleaned = re.sub(r'```', '', cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Last resort: treat entire response as code
    return {"thought": "Executing response as code", "code": raw}


async def sub_agent_summarize(
    query: str,
    data: str,
    log_callback: Optional[Callable] = None,
) -> str:
    """Spawn a sub-agent to summarize a large data chunk."""
    if log_callback:
        log_callback({
            "type": "system",
            "content": "⚡ Spawning sub-agent for summarization...",
            "meta": {},
        })

    messages = [
        {"role": "user", "content": SUMMARIZE_PROMPT.format(query=query, data=data)},
    ]
    return await _call_llm(SUB_MODEL, messages, log_callback)


async def recursive_agent_call(
    query: str,
    context: str = "",
    depth: int = 0,
    log_callback: Optional[Callable] = None,
    graph_callback: Optional[Callable] = None,
) -> str:
    """
    Recursive agent entry point.

    depth is incremented each call.  At MAX_DEPTH the agent is forced
    to summarize everything and exit.
    """
    if log_callback:
        log_callback({
            "type": "system",
            "content": f"🔄 recursive_agent_call(depth={depth}/{MAX_DEPTH})",
            "meta": {"depth": depth},
        })

    # Emit a recursion-depth graph node
    if graph_callback:
        graph_callback({
            "action": "add_node",
            "node": {
                "id": f"agent_d{depth}",
                "label": f"Agent (d={depth})",
                "source": "agent",
                "data": {"depth": depth},
            },
        })
        if depth > 0:
            graph_callback({
                "action": "add_edge",
                "edge": {
                    "source": f"agent_d{depth - 1}",
                    "target": f"agent_d{depth}",
                    "label": "spawns",
                },
            })

    sandbox = Sandbox(log_callback=log_callback, graph_callback=graph_callback)

    # Inject recursive helper into sandbox namespace
    async def _recursive_helper(ctx: str, d: int):
        return await recursive_agent_call(query, ctx, d, log_callback, graph_callback)

    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context:
        messages.append({"role": "user", "content": f"Previous context:\n{context}"})
    messages.append({"role": "user", "content": query})

    # ── Force summarize at MAX_DEPTH ──
    if depth >= MAX_DEPTH:
        if log_callback:
            log_callback({
                "type": "system",
                "content": f"🛑 MAX_DEPTH ({MAX_DEPTH}) reached — forcing summarize & exit",
                "meta": {},
            })
        messages.append({
            "role": "user",
            "content": FORCE_SUMMARIZE_PROMPT.format(context=context or "No context gathered yet."),
        })

    # ── Agent loop ──
    for step in range(MAX_STEPS):
        raw_response = await _call_llm(ROOT_MODEL, messages, log_callback)
        parsed = _parse_agent_response(raw_response)
        thought = parsed.get("thought", "")
        code = parsed.get("code", "")

        if log_callback:
            log_callback({"type": "thought", "content": thought, "meta": {"step": step}})

        if not code.strip():
            if log_callback:
                log_callback({"type": "system", "content": "No code produced — ending.", "meta": {}})
            break

        # Execute in sandbox with retry on error
        result = sandbox.execute(code)

        if result["error"]:
            # Feed error back to LLM for self-correction
            for retry in range(MAX_RETRIES):
                if log_callback:
                    log_callback({
                        "type": "system",
                        "content": f"⚠️  Error → sending back to LLM (retry {retry + 1}/{MAX_RETRIES})",
                        "meta": {},
                    })
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your code produced an error:\n```\n{result['traceback']}\n```\n"
                        "Please fix the code and try again. Respond with the same JSON format."
                    ),
                })
                raw_response = await _call_llm(ROOT_MODEL, messages, log_callback)
                parsed = _parse_agent_response(raw_response)
                code = parsed.get("code", "")
                if code.strip():
                    result = sandbox.execute(code)
                    if not result["error"]:
                        break
            # If still erroring after retries, continue to next step
            if result["error"]:
                if log_callback:
                    log_callback({"type": "error", "content": f"Failed after {MAX_RETRIES} retries", "meta": {}})

        # Check if _result_ is set (agent is done)
        final = result.get("result") or sandbox._namespace.get("_result_")
        if final is not None:
            # Don't emit final_answer here — main.py does it to avoid duplicates
            return str(final)

        # Check for large output that needs sub-agent summarization
        stdout = result.get("stdout", "")
        if len(stdout) > 500 and depth < MAX_DEPTH:
            summary = await sub_agent_summarize(query, stdout, log_callback)
            messages.append({"role": "assistant", "content": raw_response})
            messages.append({
                "role": "user",
                "content": f"A sub-agent summarized the large output:\n{summary}\n\nContinue investigation or provide final answer.",
            })
        else:
            messages.append({"role": "assistant", "content": raw_response})
            messages.append({
                "role": "user",
                "content": f"Code executed. Output:\n{stdout}\n\nContinue or provide final answer in _result_.",
            })

    # If we exhausted steps without an answer, force a summary
    return sandbox._namespace.get("_result_") or "Investigation complete but no definitive answer was produced."

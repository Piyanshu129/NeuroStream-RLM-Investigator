"""
main.py — FastAPI application with SSE streaming for the RLM agent.

Endpoints:
  POST /api/query    → SSE stream that runs the agent and streams results
  POST /api/setup    → initialize dummy data
  GET  /api/health   → healthcheck
"""

import asyncio, json, uuid, os
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

app = FastAPI(title="RLM Data Investigator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


class SetupResponse(BaseModel):
    status: str
    message: str


# ── Routes ────────────────────────────────────────────────────────────

@app.post("/api/query")
async def start_query(req: QueryRequest):
    """
    Single endpoint: accepts the query and immediately returns an SSE
    stream. The agent runs inside the generator so there is no race
    condition between POST and EventSource.
    """
    queue: asyncio.Queue = asyncio.Queue()

    def log_callback(entry: dict):
        try:
            queue.put_nowait(entry)
        except asyncio.QueueFull:
            pass

    def graph_callback(entry: dict):
        entry["type"] = "graph"
        try:
            queue.put_nowait(entry)
        except asyncio.QueueFull:
            pass

    async def _run_agent():
        from backend.agent import recursive_agent_call

        try:
            result = await recursive_agent_call(
                query=req.query,
                context="",
                depth=0,
                log_callback=log_callback,
                graph_callback=graph_callback,
            )
            log_callback({"type": "final_answer", "content": result, "meta": {}})
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log_callback({"type": "error", "content": f"Agent error: {e}\n{tb}", "meta": {}})
        finally:
            await queue.put(None)  # sentinel

    async def _event_generator():
        # Start agent as background task inside the SSE generator
        agent_task = asyncio.create_task(_run_agent())

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=300)
            except asyncio.TimeoutError:
                yield {"event": "keepalive", "data": "ping"}
                continue

            if event is None:
                break

            event_type = event.get("type", "log")

            if event_type in ("thought", "code", "output", "error", "system",
                              "llm_call", "llm_response"):
                yield {"event": "thought_trace", "data": json.dumps(event)}
            elif event_type == "final_answer":
                yield {"event": "final_answer", "data": json.dumps(event)}
            elif event_type == "graph":
                yield {"event": "graph_update", "data": json.dumps(event)}
            else:
                yield {"event": "thought_trace", "data": json.dumps(event)}

        yield {"event": "done", "data": json.dumps({"status": "complete"})}

    return EventSourceResponse(_event_generator())


@app.post("/api/setup")
async def setup_data():
    """Initialize / reset dummy data sources."""
    try:
        from backend.setup_dummy_data import main as setup_main
        setup_main()
        return SetupResponse(status="ok", message="All data sources initialized successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    return {"status": "ok"}

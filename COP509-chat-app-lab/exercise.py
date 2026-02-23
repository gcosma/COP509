"""
Chatbot Backend — Lab Exercise
===============================
Follow the Exercise Sheet to complete each task.
"""

import os
import json
import pathlib
from datetime import datetime, timezone
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI

from _infrastructure import register_routes

load_dotenv()

app = FastAPI()

# ── TODO 1: see Exercise Sheet, Task 1 ──────
# client = ...


MODELS = [m.strip() for m in os.getenv("MODELS", "").split(",") if m.strip()]


@app.get("/models")
async def list_models():
    return MODELS


SYSTEM_PROMPT = "You are a helpful assistant. Keep your answers concise."


@app.post("/chat/stream")
async def chat_stream(request: Request):
    body = await request.json()
    user_message = body.get("message", "")
    image_data = body.get("image", None)
    history = body.get("history", [])
    model = body.get("model", MODELS[0] if MODELS else None)

    # ── TODO 4 (Extension A): see Exercise Sheet, Task 4 ──

    # ── TODO 2: see Exercise Sheet, Task 2 ──
    # ── TODO 3: see Exercise Sheet, Task 3 ──

    def placeholder():
        yield f"data: {json.dumps({'content': 'TODO 2: Replace me with streaming!'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(placeholder(), media_type="text/event-stream")


# ══════════════════════════════════════════════
# EXTENSIONS
# ══════════════════════════════════════════════

# ── Extension B: Conversation Persistence ─────

CONVERSATIONS_DIR = pathlib.Path(__file__).parent / "conversations"
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)


@app.post("/conversations")
async def save_conversation(request: Request):
    body = await request.json()
    conv_id = body.get("id")
    title = body.get("title", "Untitled")
    messages = body.get("messages", [])
    # ── TODO 5: see Exercise Sheet, Task 5 ──
    return {"status": "ok"}


@app.get("/conversations")
async def list_conversations():
    # ── TODO 6: see Exercise Sheet, Task 6 ──
    pass


@app.get("/conversations/{conv_id}")
async def load_conversation(conv_id: str):
    # ── TODO 6: see Exercise Sheet, Task 6 ──
    pass


# ── Extension C: LLM Benchmarking ────────────
# See Exercise Sheet, Task 7.

def run_evaluation(task_name, model, limit, log_dir):
    """Run an inspect_ai evaluation. Returns (logs, accuracy)."""
    # ── TODO 7: see Exercise Sheet, Task 7 ──
    logs = None
    accuracy = 0.0
    return (logs, accuracy)


# ── Wire everything together ─────────────────
register_routes(app, run_eval_fn=run_evaluation)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

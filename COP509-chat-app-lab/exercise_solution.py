"""
Chatbot Backend — Model Answer
================================
All TODOs completed. This is the instructor reference.
"""

import os
import json
import pathlib
from datetime import datetime, timezone
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from inspect_ai import eval as inspect_eval
import inspect_evals  # noqa: F401 — registers benchmark tasks

from _infrastructure import register_routes

load_dotenv()

app = FastAPI()

# ──────────────────────────────────────────────
# TODO 1: Initialize the OpenAI client ✅
# ──────────────────────────────────────────────
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODELS = [m.strip() for m in os.getenv("MODELS", "").split(",") if m.strip()]

SYSTEM_PROMPT = "You are a helpful assistant. Keep your answers concise."


@app.get("/models")
async def list_models():
    return MODELS


# ──────────────────────────────────────────────
# POST /chat/stream  — Streaming chat (SSE)
# ──────────────────────────────────────────────
@app.post("/chat/stream")
async def chat_stream(request: Request):
    body = await request.json()
    user_message = body.get("message", "")
    image_data = body.get("image", None)
    history = body.get("history", [])
    model = body.get("model", MODELS[0] if MODELS else None)

    # TODO 4: Vision ✅
    if image_data:
        user_content = [
            {"type": "text", "text": user_message},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
        ]
    else:
        user_content = user_message

    # TODOs 2+3: Stream with history ✅
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": user_content}]

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    def generate():
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield f"data: {json.dumps({'content': content})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ──────────────────────────────────────────────
# Extension B: Conversation Persistence
# ──────────────────────────────────────────────
CONVERSATIONS_DIR = pathlib.Path(__file__).parent / "conversations"


# TODO 5: Save a conversation ✅
@app.post("/conversations")
async def save_conversation(request: Request):
    body = await request.json()
    conv_id = body.get("id")
    title = body.get("title", "Untitled")
    messages = body.get("messages", [])

    os.makedirs(CONVERSATIONS_DIR, exist_ok=True)

    data = {
        "id": conv_id,
        "title": title,
        "messages": messages,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(CONVERSATIONS_DIR / f"{conv_id}.json", "w") as f:
        json.dump(data, f, indent=2)

    return {"status": "ok"}


# TODO 6: List & load conversations ✅
@app.get("/conversations")
async def list_conversations():
    if not CONVERSATIONS_DIR.exists():
        return []

    conversations = []
    for filepath in CONVERSATIONS_DIR.glob("*.json"):
        with open(filepath) as f:
            data = json.load(f)
        conversations.append({
            "id": data["id"],
            "title": data.get("title", "Untitled"),
            "updated_at": data.get("updated_at", ""),
        })

    conversations.sort(key=lambda c: c["updated_at"], reverse=True)
    return conversations


@app.get("/conversations/{conv_id}")
async def load_conversation(conv_id: str):
    filepath = CONVERSATIONS_DIR / f"{conv_id}.json"
    if not filepath.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)

    with open(filepath) as f:
        data = json.load(f)
    return data


# ──────────────────────────────────────────────
# Extension C: LLM Benchmarking — TODO 7 ✅
# ──────────────────────────────────────────────
def run_evaluation(task_name, model, limit, log_dir):
    """Run an inspect_ai evaluation. Returns (logs, accuracy)."""
    logs = inspect_eval(
        task_name,
        model=f"openrouter/{model}",
        limit=limit,
        display="none",
        log_dir=log_dir,
        log_buffer=1,
    )
    scores = logs[0].results.scores[0].metrics
    first_metric = next(iter(scores.values()))
    accuracy = round(first_metric.value * 100, 1)
    return (logs, accuracy)


# ── Wire everything together ─────────────────
register_routes(app, run_eval_fn=run_evaluation)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

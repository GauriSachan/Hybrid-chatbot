"""
server.py -- Part 5 backend: exposes POST /chat for app.py (the Streamlit frontend).

Request / response contract (must match app.py's call_real_backend / parse_backend_response):
    POST /chat
    { "question": "...", "session_id": "...", "history": [...] }   (history optional)
    ->
    { "answer": "...",
      "sources": [{"source": "...", "page": 1, "url": "..."}, ...],
      "contexts": ["...retrieved text...", ...] }

Run:
    pip install fastapi uvicorn
    export ANTHROPIC_API_KEY=...
    uvicorn server:app --port 8000
Then in app.py set MOCK_MODE = False (API_URL already defaults to http://localhost:8000/chat).

Upload endpoint (not yet called by app.py -- add a file-upload widget there to use it):
    POST /upload  (multipart file + form field session_id)  -> {"filename": "..."}
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ingestion import delete_session_docs, ingest_user_doc
from pipeline import Session, build_chatbot

app = FastAPI(title="RAG Chatbot Backend")

_lock = threading.Lock()
_sessions: dict[str, Session] = {}
_bot = None 

# Which LLM provider to use: set with  export CHATBOT_PROVIDER=gemini  (default: anthropic).
# Free, no credit card: "gemini" (GEMINI_API_KEY, from aistudio.google.com/apikey) or
# "groq" (GROQ_API_KEY, from console.groq.com). Paid: "anthropic" / "openai".
PROVIDER = os.getenv("CHATBOT_PROVIDER", "gemini")


def get_bot():
    global _bot
    if _bot is None:
        with _lock:
            if _bot is None:                      # re-check: another thread may have built it
                _bot = build_chatbot(provider="gemini", rerank=False)
    return _bot


def get_session(session_id: str) -> Session:
    with _lock:
        if session_id not in _sessions:
            _sessions[session_id] = Session(id=session_id)
        return _sessions[session_id]


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None     # falls back to a shared default if the client omits it
    history: Optional[list[dict]] = None  # accepted but ignored: the server keeps its own history


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    contexts: list[str]
    route: str                            # extra field; app.py's parser ignores unknown keys


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "question must not be empty")
    session = get_session(req.session_id or "default")
    resp = get_bot().chat(req.question, session)
    sources = [{"source": s["source"], "page": s.get("page"), "url": s.get("url")} for s in resp.sources]
    contexts = [s["snippet"] for s in resp.sources]
    return ChatResponse(answer=resp.answer, sources=sources, contexts=contexts, route=resp.route)


@app.post("/upload")
def upload(session_id: str = Form(...), file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        name = ingest_user_doc(tmp_path, session_id, original_name=file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    get_session(session_id).user_docs.append(name)
    return {"filename": name}


@app.post("/session/{session_id}/docs/clear")
def clear_docs(session_id: str):
    """Remove this session's uploaded files WITHOUT ending the chat / clearing history."""
    n = delete_session_docs(session_id)
    s = get_session(session_id)
    s.user_docs.clear()
    return {"status": "cleared", "chunks_removed": n}


@app.post("/session/{session_id}/end")
def end_session(session_id: str):
    delete_session_docs(session_id)
    with _lock:
        _sessions.pop(session_id, None)
    return {"status": "ended"}


@app.get("/health")
def health():
    return {"status": "ok"}

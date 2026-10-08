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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from importlib.util import find_spec

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ingestion import delete_session_docs, ingest_user_doc
from pipeline import Session, build_chatbot

@asynccontextmanager
async def lifespan(_app):
    # Load models at startup so the first /chat call isn't slower than app.py's 30 s timeout.
    if os.getenv("WARMUP", "1") == "1":
        try:
            get_bot()
        except RuntimeError:
            # Keep startup alive; the user can install the missing provider dependency later.
            pass
    yield


app = FastAPI(title="RAG Chatbot Backend", lifespan=lifespan)

_lock = threading.Lock()
_sessions: dict[str, Session] = {}
_bot = None   # built lazily, once, on first request (model loading is slow)


def get_bot():
    global _bot
    if _bot is None:
        with _lock:
            if _bot is None:                      # re-check: another thread may have built it
                provider = os.getenv("CHAT_PROVIDER", "openai")
                try:
                    _bot = build_chatbot(provider=provider, rerank=os.getenv("RERANK", "1") == "1")
                except ModuleNotFoundError as exc:
                    raise RuntimeError(
                        f"The selected provider '{provider}' is not installed. "
                        f"Install it with: pip install {provider} "
                        f"or set CHAT_PROVIDER=openai and install openai."
                    ) from exc
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
    try:
        resp = get_bot().chat(req.question, session)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    sources = [{"source": s["source"], "page": s.get("page"), "url": s.get("url")} for s in resp.sources]
    contexts = [s["snippet"] for s in resp.sources]
    return ChatResponse(answer=resp.answer, sources=sources, contexts=contexts, route=resp.route)


if find_spec("multipart") is not None:
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
else:
    @app.post("/upload")
    def upload():
        raise HTTPException(
            503,
            "File upload requires the optional dependency 'python-multipart'. "
            "Install it with: pip install python-multipart",
        )


@app.post("/session/{session_id}/end")
def end_session(session_id: str):
    delete_session_docs(session_id)
    with _lock:
        _sessions.pop(session_id, None)
    return {"status": "ended"}


@app.get("/health")
def health():
    return {"status": "ok"}

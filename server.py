"""
Voice-agent backend — pure JSON API, auth delegated to Neon Auth.

Every route returns JSON (or NDJSON for /turn). Auth is a short-lived JWT
issued by Neon Auth, sent as `Authorization: Bearer <token>`, verified
against Neon's public JWKS (see neon_auth.py) — no signing secret of ours
to generate or rotate, and no custom users table to maintain, since Neon
Auth syncs signed-in users into Postgres itself.

This stays frontend-agnostic on purpose: the web dashboard (index.html),
a future phone app, or curl can all call these same routes with the same
Bearer token scheme.
"""
import os
import json
import asyncio
import tempfile
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage

from backend_mcp import chatbot, get_thread_state
from retrieval import create_collection_and_ingest
from stt import speech_to_text
from Tts import text_to_speech, AVAILABLE_VOICES
import neon_auth

load_dotenv()

ALLOWED_EMAILS = {
    e.strip().lower() for e in os.getenv("ALLOWED_EMAILS", "").split(",") if e.strip()
}
AGENT_VOICE = os.getenv("AGENT_VOICE", "en-IN-PrabhatNeural")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

SYSTEM_PROMPT = """You are Vikas, a warm, efficient voice + chat assistant.
Speak in natural Hinglish by default. Keep turns short and conversational.
Ask about the user's need, budget, and timeline when relevant. Use your
tools (web search, stock price, document retrieval, wiki, expenses) whenever
they would give a better answer than guessing."""

app = FastAPI(title="Vikas Voice Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # Bearer-token auth, not cookies
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Auth — verify the Neon Auth JWT on every protected route
# ============================================================

async def require_auth(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header.")
    token = authorization.split(" ", 1)[1].strip()

    claims = neon_auth.validate_neon_token(token)
    if claims is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")

    email = (claims.get("email") or "").lower()
    if ALLOWED_EMAILS and email not in ALLOWED_EMAILS:
        raise HTTPException(status_code=403, detail="This account is not allowed to sign in.")

    return {"id": claims["sub"], "email": email, "name": claims.get("name")}


@app.get("/me")
async def me(user=Depends(require_auth)):
    return {"user": user}


@app.get("/config")
async def config():
    """Public, non-secret config the client needs to talk to Neon Auth."""
    return {"neon_auth_base_url": neon_auth.NEON_AUTH_BASE_URL}


def _scoped_thread_id(user: dict, call_id: str) -> str:
    """Namespaces every LangGraph thread by Neon Auth user id, so two users
    can never read or continue each other's conversation."""
    return f"user:{user['id']}:{call_id}"


async def seed_messages(thread_id: str, user_text: str):
    state = await asyncio.to_thread(get_thread_state, thread_id)
    existing = state.values.get("messages", []) if state and state.values else []
    if not existing:
        return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_text)]
    return [HumanMessage(content=user_text)]


# ============================================================
# Text chat
# ============================================================

class ChatRequest(BaseModel):
    call_id: str
    message: str
    speak: bool = False
    voice: Optional[str] = None


@app.post("/chat")
async def chat(body: ChatRequest, user=Depends(require_auth)):
    thread_id = _scoped_thread_id(user, body.call_id)
    messages = await seed_messages(thread_id, body.message)
    config = {"configurable": {"thread_id": thread_id}}

    result = await chatbot.ainvoke({"messages": messages}, config=config)
    reply = result["messages"][-1].content

    audio_b64 = None
    if body.speak:
        voice = body.voice or AGENT_VOICE
        audio_bytes = await asyncio.to_thread(text_to_speech, reply, voice)
        if audio_bytes:
            import base64
            audio_b64 = base64.b64encode(audio_bytes).decode()

    return {"reply": reply, "audio": audio_b64}


# ============================================================
# Voice turn (NDJSON stream: user transcript -> sentence-by-sentence reply+audio)
# ============================================================

def unfinished(text: str) -> bool:
    trailing = text.strip().lower()
    if not trailing:
        return True
    connectors = ("and", "but", "so", "because", "aur", "lekin", "toh", "ki")
    last_word = trailing.split()[-1].strip(",.!?")
    return last_word in connectors


def _split_sentences(text: str):
    import re
    parts = re.split(r"(?<=[.!?।])\s+", text.strip())
    return [p for p in parts if p]


@app.post("/turn")
async def turn(
    call_id: str,
    audio: UploadFile = File(...),
    user=Depends(require_auth),
):
    audio_bytes = await audio.read()
    transcript = await asyncio.to_thread(speech_to_text, audio_bytes, audio.filename or "audio.webm")

    async def stream():
        if not transcript.strip():
            yield json.dumps({"type": "empty"}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
            return

        if unfinished(transcript):
            yield json.dumps({"type": "wait"}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
            return

        yield json.dumps({"type": "user", "text": transcript}) + "\n"

        thread_id = _scoped_thread_id(user, call_id)
        messages = await seed_messages(thread_id, transcript)
        config = {"configurable": {"thread_id": thread_id}}

        result = await chatbot.ainvoke({"messages": messages}, config=config)
        reply = result["messages"][-1].content

        import base64
        for sentence in _split_sentences(reply):
            audio_bytes_out = await asyncio.to_thread(text_to_speech, sentence, AGENT_VOICE)
            audio_b64 = base64.b64encode(audio_bytes_out).decode() if audio_bytes_out else ""
            yield json.dumps({"type": "sentence", "text": sentence, "audio": audio_b64}) + "\n"

        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/interrupt")
async def interrupt(call_id: str, n: int, user=Depends(require_auth)):
    return {"ok": True}


# ============================================================
# Document upload + indexing (RAG)
# ============================================================

@app.post("/ingest")
async def ingest(file: UploadFile = File(...), user=Depends(require_auth)):
    """Upload a PDF and index it into Qdrant so retrieve_relevant_chunks
    can answer questions about it, in this same voice/chat session."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    tmp_path = None
    try:
        data = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        document_id = await asyncio.to_thread(create_collection_and_ingest, tmp_path)
        return {"document_id": document_id, "filename": file.filename}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ============================================================
# Misc
# ============================================================

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/voices")
async def voices():
    return {"voices": AVAILABLE_VOICES}
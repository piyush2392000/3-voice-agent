"""Speech-to-text helper using Groq's Whisper API."""

import os
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
DEFAULT_STT_MODEL = "whisper-large-v3-turbo"


def speech_to_text(audio_bytes: bytes, filename: str = "audio.wav", model: str = DEFAULT_STT_MODEL) -> str:
    if not audio_bytes:
        return ""
    if not GROQ_API_KEY:
        print("[warning] GROQ_API_KEY is not set — cannot transcribe audio.")
        return ""
    try:
        response = requests.post(
            GROQ_TRANSCRIBE_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": (filename, audio_bytes)},
            data={"model": model},
            timeout=30,
        )
        response.raise_for_status()
        return (response.json().get("text") or "").strip()
    except requests.RequestException as exc:
        print(f"[warning] Groq speech-to-text failed: {exc}")
        return ""

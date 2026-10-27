"""Text-to-speech helper built on edge-tts (free, no API key required)."""

import edge_tts

from backend_mcp import run_async

AVAILABLE_VOICES = {
    "Aria (US, female)": "en-US-AriaNeural",
    "Guy (US, male)": "en-US-GuyNeural",
    "Jenny (US, female)": "en-US-JennyNeural",
    "Ryan (UK, male)": "en-GB-RyanNeural",
    "Sonia (UK, female)": "en-GB-SoniaNeural",
    "Swara (Hindi, female)": "hi-IN-SwaraNeural",
    "Madhur (Hindi, male)": "hi-IN-MadhurNeural",
    "Aarohi (Marathi, female)": "mr-IN-AarohiNeural",
    "Manohar (Marathi, male)": "mr-IN-ManoharNeural",
    "Nanami (Japanese, female)": "ja-JP-NanamiNeural",
    "Keita (Japanese, male)": "ja-JP-KeitaNeural",
}
DEFAULT_VOICE = "en-US-AriaNeural"


async def _synthesize(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    audio_bytes = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes.extend(chunk["data"])
    return bytes(audio_bytes)


def text_to_speech(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    text = (text or "").strip()
    if not text:
        return b""
    try:
        return run_async(_synthesize(text, voice))
    except Exception as exc:
        print(f"[warning] edge-tts synthesis failed: {exc}")
        return b""

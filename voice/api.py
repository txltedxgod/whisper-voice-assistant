"""FastAPI endpoints for Voice Assistant transcript NLU and WAV VAD inspection."""

import base64
import binascii
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from voice.pipeline import VoicePipeline

app = FastAPI(title="Whisper Voice Assistant Pipeline", version="1.0.0")
pipe = VoicePipeline()


class SpeechReq(BaseModel):
    """Transcribed utterance payload."""

    transcript: str = Field(..., min_length=1, max_length=4000)


class AudioBytesReq(BaseModel):
    """Base64-encoded PCM WAV audio telemetry request."""

    audio_base64: str = Field(..., min_length=8)
    silence_threshold_dbfs: float = Field(default=-40.0, ge=-96.0, le=0.0)


@app.get("/health", tags=["Monitoring"])
def health() -> dict[str, str]:
    """Liveness check endpoint."""
    return {"status": "healthy"}


@app.post("/api/v1/voice/turn", tags=["Voice"])
def voice_turn(req: SpeechReq) -> dict[str, Any]:
    """Handle one conversational turn and return text response + intent metadata."""
    try:
        result = pipe.interpret_command(req.transcript)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "input": req.transcript,
        "intent": result["intent"],
        "confidence": result["confidence"],
        "slots": result["slots"],
        "response_speech_text": result["response_speech_text"],
    }


@app.post("/api/v1/voice/analyze-audio", tags=["Audio"])
def analyze_audio(req: AudioBytesReq) -> dict[str, Any]:
    """Decode base64 WAV audio and compute VAD energy/duration telemetry."""
    try:
        raw = base64.b64decode(req.audio_base64, validate=True)
        return pipe.analyze_wav_bytes(
            raw, silence_threshold_dbfs=req.silence_threshold_dbfs
        )
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

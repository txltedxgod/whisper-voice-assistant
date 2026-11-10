"""Unit and API tests for WAV PCM VAD analysis and voice command NLU."""

from __future__ import annotations

import base64
import io
import math
import struct
import wave

from fastapi.testclient import TestClient

from voice.api import app
from voice.pipeline import VoicePipeline


def _make_wav_bytes(
    *, amplitude: int, sample_rate: int = 16000, duration_sec: float = 0.2
) -> bytes:
    buf = io.BytesIO()
    n_frames = int(sample_rate * duration_sec)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = [
            int(amplitude * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
            for i in range(n_frames)
        ]
        wf.writeframes(struct.pack(f"<{n_frames}h", *frames))
    return buf.getvalue()


def test_voice_intent_and_slots() -> None:
    p = VoicePipeline()
    res = p.process_audio_transcript("What is the system status?")
    assert "healthy" in res

    parsed = p.interpret_command("Check the weather in Prague tomorrow")
    assert parsed["intent"] == "weather_query"
    assert parsed["slots"]["location"] == "Prague"


def test_wav_vad_distinguishes_speech_and_silence() -> None:
    p = VoicePipeline()
    active_wav = _make_wav_bytes(amplitude=12000)
    active = p.analyze_wav_bytes(active_wav)
    assert active["sample_rate"] == 16000
    assert active["has_speech"] is True
    assert active["speech_ratio"] > 0.8

    silent_wav = _make_wav_bytes(amplitude=0)
    silent = p.analyze_wav_bytes(silent_wav)
    assert silent["has_speech"] is False
    assert silent["rms_dbfs"] <= -40.0


def test_voice_api_endpoints() -> None:
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "healthy"}

    turn = client.post(
        "/api/v1/voice/turn", json={"transcript": "Set a timer for 10 minutes"}
    )
    assert turn.status_code == 200
    assert turn.json()["intent"] == "timer_reminder"
    assert turn.json()["slots"]["duration"] == "10 minutes"

    encoded = base64.b64encode(_make_wav_bytes(amplitude=8000)).decode("ascii")
    audio_resp = client.post(
        "/api/v1/voice/analyze-audio", json={"audio_base64": encoded}
    )
    assert audio_resp.status_code == 200
    assert audio_resp.json()["has_speech"] is True

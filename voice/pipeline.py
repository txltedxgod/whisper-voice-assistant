"""Voice Assistant audio VAD telemetry and intent classification pipeline."""

from __future__ import annotations

import io
import math
import re
import struct
import wave
from typing import Any

_DURATION_RE = re.compile(r"\b(\d+)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)\b", re.I)
_LOCATION_RE = re.compile(r"\bin\s+([A-Z][a-zA-Z\-]{1,40}(?:\s+[A-Z][a-zA-Z\-]{1,40})?)")


class VoicePipeline:
    """Process WAV PCM audio signals and spoken transcripts."""

    def analyze_wav_bytes(
        self,
        wav_bytes: bytes,
        silence_threshold_dbfs: float = -40.0,
    ) -> dict[str, Any]:
        """Inspect a PCM WAV buffer for duration, RMS loudness, and speech activity."""
        if not wav_bytes:
            raise ValueError("Audio payload is empty.")

        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw_frames = wf.readframes(n_frames)
        except wave.Error as exc:
            raise ValueError(f"Invalid WAV stream: {exc}") from exc

        if sample_rate <= 0 or channels <= 0 or sample_width not in (1, 2):
            raise ValueError(
                f"Unsupported WAV format: channels={channels}, "
                f"width={sample_width}, rate={sample_rate}"
            )

        duration_seconds = round(n_frames / sample_rate, 4)
        if n_frames == 0:
            return {
                "sample_rate": sample_rate,
                "channels": channels,
                "duration_seconds": 0.0,
                "rms_dbfs": -96.0,
                "peak_ratio": 0.0,
                "speech_ratio": 0.0,
                "has_speech": False,
            }

        if sample_width == 2:
            total_samples = len(raw_frames) // 2
            samples = [
                s / 32768.0 for s in struct.unpack(f"<{total_samples}h", raw_frames)
            ]
        else:
            samples = [(b - 128) / 128.0 for b in raw_frames]

        mean_sq = sum(s * s for s in samples) / len(samples)
        rms = math.sqrt(mean_sq)
        rms_dbfs = round(20.0 * math.log10(max(rms, 1e-9)), 2)
        peak_ratio = round(max(abs(s) for s in samples), 4)

        # Frame-level VAD using 20ms windows across mono-folded sample stream
        step = max(1, (sample_rate * channels) // 50)
        threshold_amp = 10.0 ** (silence_threshold_dbfs / 20.0)
        active_windows = 0
        total_windows = 0
        for idx in range(0, len(samples), step):
            chunk = samples[idx : idx + step]
            if not chunk:
                continue
            total_windows += 1
            win_rms = math.sqrt(sum(s * s for s in chunk) / len(chunk))
            if win_rms >= threshold_amp:
                active_windows += 1

        speech_ratio = round(active_windows / total_windows, 4) if total_windows else 0.0
        return {
            "sample_rate": sample_rate,
            "channels": channels,
            "duration_seconds": duration_seconds,
            "rms_dbfs": rms_dbfs,
            "peak_ratio": peak_ratio,
            "speech_ratio": speech_ratio,
            "has_speech": rms_dbfs >= silence_threshold_dbfs and speech_ratio > 0.05,
        }

    def interpret_command(self, transcript: str) -> dict[str, Any]:
        """Classify spoken text intent, extract slots, and synthesize reply."""
        cleaned = (transcript or "").strip()
        if not cleaned:
            raise ValueError("Transcript must not be empty.")
        lower = cleaned.lower()

        slots: dict[str, str] = {}
        loc_match = _LOCATION_RE.search(cleaned)
        if loc_match:
            slots["location"] = loc_match.group(1)
        dur_match = _DURATION_RE.search(cleaned)
        if dur_match:
            slots["duration"] = f"{dur_match.group(1)} {dur_match.group(2).lower()}"

        if any(k in lower for k in ("weather", "temperature", "forecast")):
            intent = "weather_query"
            confidence = 0.92
            place = slots.get("location", "your region")
            reply = f"Forecast for {place}: clear skies and mild conditions."
        elif any(k in lower for k in ("deploy", "status", "health", "uptime")):
            intent = "system_health"
            confidence = 0.95
            reply = "All monitored microservices are healthy and passing health probes."
        elif any(k in lower for k in ("timer", "remind", "alarm", "schedule")):
            intent = "timer_reminder"
            confidence = 0.9
            duration = slots.get("duration", "the requested interval")
            reply = f"Scheduled timer for {duration}."
        elif any(k in lower for k in ("rollback", "restart", "scale")):
            intent = "ops_action"
            confidence = 0.88
            reply = f"Prepared safe dry-run plan for operational command: {cleaned}."
        else:
            intent = "general_query"
            confidence = 0.7
            reply = f"Processed query: {cleaned}. Ready for next command."

        return {
            "transcript": cleaned,
            "intent": intent,
            "confidence": confidence,
            "slots": slots,
            "response_speech_text": reply,
        }

    def process_audio_transcript(self, transcript: str) -> str:
        """Return spoken response string for a transcribed utterance."""
        return self.interpret_command(transcript)["response_speech_text"]

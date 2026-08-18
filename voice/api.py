from fastapi import FastAPI
from pydantic import BaseModel
from voice.pipeline import VoicePipeline

app = FastAPI(title="Voice Assistant Pipeline", version="0.1.0")
pipe = VoicePipeline()

class SpeechReq(BaseModel):
    transcript: str

@app.post("/api/v1/voice/turn")
def voice_turn(req: SpeechReq):
    reply = pipe.process_audio_transcript(req.transcript)
    return {"input": req.transcript, "response_speech_text": reply}

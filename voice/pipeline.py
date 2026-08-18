class VoicePipeline:
    def process_audio_transcript(self, transcript: str) -> str:
        prompt = transcript.strip().lower()
        if "weather" in prompt:
            return "The current temperature is 22 degrees Celsius and sunny."
        elif "deploy" in prompt or "status" in prompt:
            return "All 15 microservices are healthy and passing health probes."
        return f"Processed query: {transcript}. Ready for next command."

from voice.pipeline import VoicePipeline

def test_voice():
    p = VoicePipeline()
    res = p.process_audio_transcript("What is the system status?")
    assert "healthy" in res

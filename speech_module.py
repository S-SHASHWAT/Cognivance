import os

import speech_recognition as sr
from pydub import AudioSegment


def transcribe_audio_file(video_path):
    wav_path = "converted.wav"

    try:
        # Keep processing time bounded for production deployment.
        audio = AudioSegment.from_file(video_path)
        if len(audio) > 45000:
            audio = audio[:45000]
        audio.export(wav_path, format="wav")

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)
        return text
    except Exception:
        return "Could not understand audio"
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

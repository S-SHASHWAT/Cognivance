import speech_recognition as sr
from pydub import AudioSegment
import os

def transcribe_audio_file(audio_path):

    # Convert browser audio (webm) → wav using FFmpeg
    webm_audio = AudioSegment.from_file(audio_path, format="webm")
    webm_audio.export("converted.wav", format="wav")

    recognizer = sr.Recognizer()

    with sr.AudioFile("converted.wav") as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
    except:
        text = "Could not understand audio"

    # cleanup temp file
    if os.path.exists("converted.wav"):
        os.remove("converted.wav")

    return text
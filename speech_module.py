import speech_recognition as sr
from pydub import AudioSegment

def transcribe_audio_file(video_path):
    wav_path = "converted.wav"

    try:
        # Convert webm → wav
        audio = AudioSegment.from_file(video_path)
        audio.export(wav_path, format="wav")

        recognizer = sr.Recognizer()

        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        return text

    except:
        return "Could not understand audio"
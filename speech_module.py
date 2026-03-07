import os
import subprocess
import tempfile

import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.utils import make_chunks


def _transcribe_segment(recognizer, segment, temp_dir, idx):
    segment_path = os.path.join(temp_dir, f"segment_{idx}.wav")
    segment.export(segment_path, format="wav")

    with sr.AudioFile(segment_path) as source:
        audio_data = recognizer.record(source)

    return recognizer.recognize_google(audio_data)


def transcribe_audio_file(video_path):
    recognizer = sr.Recognizer()

    with tempfile.TemporaryDirectory(prefix="stt_") as temp_dir:
        wav_path = os.path.join(temp_dir, "full_audio.wav")

        try:
            # Extract complete audio at STT-friendly settings.
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    video_path,
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    wav_path,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            audio = AudioSegment.from_wav(wav_path)

            # First choice: speech-only segments across entire answer.
            silence_threshold = audio.dBFS - 16 if audio.dBFS != float("-inf") else -45
            segments = split_on_silence(
                audio,
                min_silence_len=500,
                silence_thresh=silence_threshold,
                keep_silence=250,
            )

            # Fallback: fixed chunks when silence split is too aggressive.
            if not segments:
                segments = make_chunks(audio, 10000)  # 10-second chunks

            transcripts = []
            for idx, segment in enumerate(segments):
                if len(segment) < 350:
                    continue
                try:
                    part = _transcribe_segment(recognizer, segment, temp_dir, idx)
                    if part and part.strip():
                        transcripts.append(part.strip())
                except Exception:
                    continue

            final_text = " ".join(transcripts).strip()
            return final_text if final_text else "Could not understand audio"
        except Exception:
            return "Could not understand audio"

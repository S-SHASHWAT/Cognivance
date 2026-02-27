from interview_analyzer import analyze_filler_words, analyze_eye_contact
from flask import Flask, render_template, request
import os
from speech_module import transcribe_audio_file

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start_interview", methods=["POST"])
def start_interview():
    print("Headers:", request.headers)
    print("Files:", request.files)

    if "video" not in request.files:
        return "❌ No video file received", 400

    video_file = request.files["video"]

    if video_file.filename == "":
        return "❌ Empty filename", 400

    video_path = os.path.join(UPLOAD_FOLDER, "input.webm")
    video_file.save(video_path)

    print("✅ Video saved")

    # Process
    text = transcribe_audio_file(video_path)
    filler_score = analyze_filler_words(text)
    eye_score = analyze_eye_contact(video_path)

    return f"""
    Transcript: {text} <br>
    Eye Contact Score: {eye_score} <br>
    Filler Words: {filler_score}
    """

if __name__ == "__main__":
    app.run(debug=True)
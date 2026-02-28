import os
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException
from speech_module import transcribe_audio_file
from interview_analyzer import analyze_filler_words, analyze_eye_contact

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.errorhandler(HTTPException)
def handle_http_exception(exc):
    return jsonify({"error": exc.description, "code": exc.code}), exc.code


@app.errorhandler(Exception)
def handle_unexpected_exception(exc):
    app.logger.exception("Unhandled exception")
    return jsonify({"error": str(exc), "code": 500}), 500


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/start_interview", methods=["POST"])
def start_interview():
    video_file = request.files.get("video") or request.files.get("audio")
    if video_file is None:
        return jsonify({"error": "No video file received."}), 400

    if not video_file.filename:
        return jsonify({"error": "Empty filename."}), 400

    video_path = os.path.join(UPLOAD_FOLDER, "input.webm")
    video_file.save(video_path)

    text = transcribe_audio_file(video_path)
    filler_score = analyze_filler_words(text)
    eye_score = analyze_eye_contact(video_path)

    return jsonify(
        {
            "Transcript": text,
            "Eye Contact Score": eye_score,
            "Filler Words": filler_score,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

from flask import Flask, render_template, request, jsonify
from speech_module import transcribe_audio_file
from sentiment_module import analyze_text
from interview_analyzer import analyze_eye_contact, generate_final_score
import os

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/start_interview", methods=["POST"])
def start_interview():
    try:
        uploaded_file = request.files.get("audio") or request.files.get("video")

        if not uploaded_file or uploaded_file.filename == "":
            return jsonify({"error": "No audio/video file received. Expected form field 'audio' or 'video'."}), 400

        media_path = "recorded_audio.webm"
        uploaded_file.save(media_path)

        text = transcribe_audio_file(media_path)
        sentiment, filler_count = analyze_text(text)
        eye_score = analyze_eye_contact(media_path)

        final_score = generate_final_score(eye_score, sentiment, filler_count)

        return jsonify(
            {
                "Transcript": text,
                "Eye Contact Score": eye_score,
                "Filler Words": filler_count,
                "Final Score": final_score,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

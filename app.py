import os
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException
from speech_module import transcribe_audio_file
from interview_analyzer import analyze_filler_words, analyze_eye_contact
from resume_module import (
    extract_resume_text,
    extract_skills,
    generate_skill_questions,
    evaluate_answer_correctness,
)

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEFAULT_GEMINI_API_KEY = "AIzaSyCIpuSQNA0VyGtXfoLx741xkToprdE7Yn0"


def _merge_transcripts(client_text: str, server_text: str) -> tuple[str, str]:
    client = (client_text or "").strip()
    server = (server_text or "").strip()

    if not client and not server:
        return "Could not understand audio", "none"
    if client and not server:
        return client, "browser_live"
    if server and not client:
        return server, "server_stt"

    # Prefer whichever captures more words; append missing part if both differ.
    client_words = len(client.split())
    server_words = len(server.split())

    if client in server:
        return server, "server_stt"
    if server in client:
        return client, "browser_live"

    if client_words >= server_words:
        return f"{client} {server}".strip(), "browser_live+server_stt"
    return f"{server} {client}".strip(), "server_stt+browser_live"


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


@app.route("/generate_resume_questions", methods=["POST"])
def generate_resume_questions():
    resume_file = request.files.get("resume")
    if resume_file is None:
        return jsonify({"error": "No resume file received."}), 400

    if not resume_file.filename:
        return jsonify({"error": "Empty resume filename."}), 400

    ext = os.path.splitext(resume_file.filename)[1].lower()
    if ext not in {".pdf", ".txt", ".md"}:
        return jsonify({"error": "Unsupported format. Upload PDF or TXT."}), 400

    resume_path = os.path.join(UPLOAD_FOLDER, f"resume{ext}")
    resume_file.save(resume_path)

    resume_text = extract_resume_text(resume_path)
    if not resume_text:
        return jsonify({"error": "Could not extract text from resume."}), 400

    skills = extract_skills(resume_text)

    form_key = (request.form.get("gemini_api_key") or "").strip()
    gemini_api_key = form_key or os.environ.get("GEMINI_API_KEY") or DEFAULT_GEMINI_API_KEY
    model = (request.form.get("gemini_model") or "").strip() or None

    questions, source, gemini_error = generate_skill_questions(
        resume_text,
        skills,
        gemini_api_key,
        model=model,
    )

    return jsonify(
        {
            "skills": skills,
            "questions": questions,
            "question_source": source,
            "api_key_configured": bool(gemini_api_key),
            "gemini_error": gemini_error,
            "gemini_model": model or os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview"),
        }
    )


@app.route("/start_interview", methods=["POST"])
def start_interview():
    video_file = request.files.get("video") or request.files.get("audio")
    if video_file is None:
        return jsonify({"error": "No video file received."}), 400

    if not video_file.filename:
        return jsonify({"error": "Empty filename."}), 400

    question = (request.form.get("question") or "").strip()
    client_transcript = (request.form.get("client_transcript") or "").strip()
    form_key = (request.form.get("gemini_api_key") or "").strip()
    model = (request.form.get("gemini_model") or "").strip() or None
    gemini_api_key = form_key or os.environ.get("GEMINI_API_KEY") or DEFAULT_GEMINI_API_KEY

    video_path = os.path.join(UPLOAD_FOLDER, "input.webm")
    video_file.save(video_path)

    server_transcript = transcribe_audio_file(video_path)
    text, transcript_source = _merge_transcripts(client_transcript, server_transcript)

    filler_score = analyze_filler_words(text)
    eye_score = analyze_eye_contact(video_path)
    correctness_score, correctness_feedback, correctness_source, correctness_error = evaluate_answer_correctness(
        question=question,
        answer=text,
        api_key=gemini_api_key,
        model=model,
    )

    return jsonify(
        {
            "Transcript": text,
            "Transcript Source": transcript_source,
            "Eye Contact Score": eye_score,
            "Filler Words": filler_score,
            "Correctness Score": correctness_score,
            "Correctness Feedback": correctness_feedback,
            "Correctness Source": correctness_source,
            "Correctness Error": correctness_error,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

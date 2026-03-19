import os
import sqlite3
from datetime import datetime, timezone
from dotenv import load_dotenv
from flask_cors import CORS
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException
from speech_module import transcribe_audio_file
from interview_analyzer import analyze_filler_words, analyze_eye_contact
from resume_module import (
    INTERVIEW_SKILLS,
    extract_resume_text,
    extract_skills,
    evaluate_answer_with_gemini,
    generate_skill_questions,
    generate_questions_for_selected_skill,
)

load_dotenv()

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")}},
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEFAULT_GEMINI_API_KEY = ""
SKILL_SCORE_DB_PATH = os.environ.get("SKILL_SCORE_DB_PATH", os.path.join(UPLOAD_FOLDER, "skill_scores.db"))


def _get_db_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(SKILL_SCORE_DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def _init_skill_score_db() -> None:
    with _get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS skill_scores (
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                skill TEXT NOT NULL,
                final_score REAL NOT NULL,
                answered_questions INTEGER NOT NULL,
                total_questions INTEGER NOT NULL,
                saved_at TEXT NOT NULL,
                PRIMARY KEY (user_id, session_id)
            )
            """
        )
        connection.commit()


_init_skill_score_db()


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


@app.route("/skill_score", methods=["POST"])
def save_skill_score():
    payload = request.get_json(silent=True) or {}
    user_id = str(payload.get("user_id") or "").strip()
    session_id = str(payload.get("session_id") or "").strip()
    final_score = payload.get("final_score")

    if not user_id or not session_id:
        return jsonify({"error": "user_id and session_id are required."}), 400

    try:
        normalized_score = round(float(final_score), 2)
    except Exception:
        return jsonify({"error": "final_score must be a valid number."}), 400

    record = {
        "user_id": user_id,
        "session_id": session_id,
        "skill": str(payload.get("skill") or "").strip(),
        "final_score": normalized_score,
        "answered_questions": int(payload.get("answered_questions") or 0),
        "total_questions": int(payload.get("total_questions") or 0),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }

    with _get_db_connection() as connection:
        connection.execute(
            """
            INSERT INTO skill_scores (
                user_id, session_id, skill, final_score,
                answered_questions, total_questions, saved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, session_id) DO UPDATE SET
                skill = excluded.skill,
                final_score = excluded.final_score,
                answered_questions = excluded.answered_questions,
                total_questions = excluded.total_questions,
                saved_at = excluded.saved_at
            """,
            (
                record["user_id"],
                record["session_id"],
                record["skill"],
                record["final_score"],
                record["answered_questions"],
                record["total_questions"],
                record["saved_at"],
            ),
        )
        connection.commit()

    return jsonify({"message": "Skill score saved.", "record": record})


@app.route("/skill_score/<user_id>/<session_id>", methods=["GET"])
def get_skill_score(user_id: str, session_id: str):
    with _get_db_connection() as connection:
        row = connection.execute(
            """
            SELECT user_id, session_id, skill, final_score,
                   answered_questions, total_questions, saved_at
            FROM skill_scores
            WHERE user_id = ? AND session_id = ?
            """,
            (user_id.strip(), session_id.strip()),
        ).fetchone()

    if row is None:
        return jsonify({"error": "Skill score not found for the provided user_id and session_id."}), 404
    return jsonify(dict(row))


@app.route("/interview_skills", methods=["GET"])
def interview_skills():
    return jsonify(
        {
            "skills": [
                {"key": key, "label": value["label"]}
                for key, value in INTERVIEW_SKILLS.items()
            ]
        }
    )


@app.route("/generate_skill_questions", methods=["POST"])
def generate_selected_skill_questions():
    selected_skill = (request.form.get("skill") or "").strip().lower()
    if selected_skill not in INTERVIEW_SKILLS:
        return jsonify({"error": "Select one valid skill."}), 400

    form_key = (request.form.get("gemini_api_key") or "").strip()
    gemini_api_key = form_key or os.environ.get("GEMINI_API_KEY") or DEFAULT_GEMINI_API_KEY
    model = (request.form.get("gemini_model") or "").strip() or None

    questions, source, gemini_error = generate_questions_for_selected_skill(
        selected_skill,
        gemini_api_key,
        model=model,
    )

    return jsonify(
        {
            "selected_skill": selected_skill,
            "selected_skill_label": INTERVIEW_SKILLS[selected_skill]["label"],
            "questions": questions,
            "question_source": source,
            "api_key_configured": bool(gemini_api_key),
            "gemini_error": gemini_error,
            "gemini_model": model or os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview"),
        }
    )


@app.route("/generate_resume_questions", methods=["POST"])
def generate_resume_questions():
    resume_file = request.files.get("resume")
    if resume_file is None:
        return jsonify({"error": "No resume file received."}), 400

    if not resume_file.filename:
        return jsonify({"error": "Empty resume filename."}), 400

    ext = os.path.splitext(resume_file.filename)[1].lower()
    if ext not in {".pdf", ".txt", ".md"}:
        return jsonify({"error": "Unsupported format. Upload PDF, TXT, or MD."}), 400

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
    skill_key = (request.form.get("skill") or "").strip().lower()
    client_transcript = (request.form.get("client_transcript") or "").strip()
    form_key = (request.form.get("gemini_api_key") or "").strip()
    model = (request.form.get("gemini_model") or "").strip() or None
    gemini_api_key = form_key or os.environ.get("GEMINI_API_KEY") or DEFAULT_GEMINI_API_KEY

    video_path = os.path.join(UPLOAD_FOLDER, "input.webm")
    video_file.save(video_path)

    # Keep request latency low on deployment: prefer browser transcript,
    # use server STT only as fallback (or when explicitly enabled).
    force_merge = os.environ.get("ENABLE_SERVER_STT_MERGE", "0") == "1"
    if len(client_transcript) >= 20 and not force_merge:
        text = client_transcript
        transcript_source = "browser_live"
    else:
        server_transcript = transcribe_audio_file(video_path)
        text, transcript_source = _merge_transcripts(client_transcript, server_transcript)

    filler_score = analyze_filler_words(text)
    eye_score = analyze_eye_contact(video_path)
    correctness_score, confidence_score, answer_feedback, correctness_source, correctness_error = evaluate_answer_with_gemini(
        question=question,
        answer=text,
        skill_key=skill_key,
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
            "Confidence Score": confidence_score,
            "Correctness Feedback": answer_feedback,
            "Correctness Source": correctness_source,
            "Correctness Error": correctness_error,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

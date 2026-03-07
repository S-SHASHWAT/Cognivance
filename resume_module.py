import json
import os
import re
from typing import List, Tuple

import requests
from PyPDF2 import PdfReader

try:
    from google import genai
except Exception:
    genai = None

SKILL_KEYWORDS = [
    "python",
    "java",
    "c++",
    "javascript",
    "typescript",
    "react",
    "node.js",
    "flask",
    "django",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "aws",
    "docker",
    "kubernetes",
    "machine learning",
    "deep learning",
    "nlp",
    "data analysis",
    "pandas",
    "numpy",
    "opencv",
    "git",
]


def extract_resume_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        reader = PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    if ext in {".txt", ".md"}:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            return file.read().strip()

    raise ValueError("Unsupported resume format. Use PDF or TXT.")


def extract_skills(resume_text: str) -> List[str]:
    text = resume_text.lower()
    found = []

    for skill in SKILL_KEYWORDS:
        pattern = rf"\b{re.escape(skill)}\b"
        if re.search(pattern, text):
            found.append(skill)

    return sorted(set(found))


def _fallback_questions(skills: List[str]) -> List[str]:
    selected = skills[:5] if skills else ["problem solving", "communication"]
    questions = []
    for skill in selected:
        questions.append(f"Explain a real project where you used {skill} and the impact it created.")
        questions.append(f"What is a common challenge in {skill} and how do you solve it?")
    return questions[:10]


def _clean_questions(text: str) -> List[str]:
    lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
    cleaned = []
    for line in lines:
        q = re.sub(r"^\d+[\).]\s*", "", line).strip()
        if q and len(q) > 8:
            cleaned.append(q)
    return cleaned[:8]


def _parse_gemini_rest_text(response_json: dict) -> str:
    candidates = response_json.get("candidates", [])
    if not candidates:
        return ""

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    text_parts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    return "\n".join(text_parts).strip()


def _build_prompt(resume_text: str, skills: List[str]) -> str:
    prompt_skills = ", ".join(skills[:10]) if skills else "general software engineering"
    return (
        "You are an interviewer. Generate exactly 8 concise interview questions tailored to the candidate skills. "
        "Return only a numbered list.\n"
        f"Skills: {prompt_skills}\n"
        "Resume excerpt:\n"
        f"{resume_text[:3000]}"
    )


def _generate_with_sdk(prompt: str, api_key: str, model: str) -> Tuple[List[str], str, str]:
    if genai is None:
        return [], "", "google-genai SDK not installed"

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model, contents=prompt)
        output_text = (response.text or "").strip()
        questions = _clean_questions(output_text)
        if questions:
            return questions, "gemini", ""
        return [], "", "Gemini SDK returned empty/invalid question format"
    except Exception as exc:
        return [], "", f"Gemini SDK error: {exc}"


def _generate_with_rest(prompt: str, api_key: str, model: str) -> Tuple[List[str], str, str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    try:
        response = requests.post(
            url,
            params={"key": api_key},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=30,
        )
        response.raise_for_status()

        output_text = _parse_gemini_rest_text(response.json())
        questions = _clean_questions(output_text)
        if questions:
            return questions, "gemini_rest", ""
        return [], "", "Gemini REST returned empty/invalid question format"
    except Exception as exc:
        return [], "", f"Gemini REST error: {exc}"


def generate_skill_questions(
    resume_text: str,
    skills: List[str],
    api_key: str | None,
    model: str | None = None,
) -> Tuple[List[str], str, str]:
    if not api_key:
        return _fallback_questions(skills), "fallback_no_key", "No Gemini API key provided"

    prompt = _build_prompt(resume_text, skills)
    selected_model = model or os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")

    questions, source, err = _generate_with_sdk(prompt, api_key, selected_model)
    if questions:
        return questions, source, ""

    rest_questions, rest_source, rest_err = _generate_with_rest(prompt, api_key, selected_model)
    if rest_questions:
        return rest_questions, rest_source, ""

    fallback = _fallback_questions(skills)
    combined_error = "; ".join(part for part in [err, rest_err] if part) or "Unknown Gemini failure"
    return fallback, "fallback_error", combined_error


def _parse_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def evaluate_answer_correctness(
    question: str,
    answer: str,
    api_key: str | None,
    model: str | None = None,
) -> Tuple[float, str, str, str]:
    if not answer or answer.strip() == "Could not understand audio":
        return 0.0, "Could not transcribe answer clearly.", "local", ""

    if not api_key:
        return 5.0, "Gemini key missing; used neutral correctness score.", "fallback_no_key", ""

    selected_model = model or os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
    prompt = (
        "You are evaluating an interview answer.\n"
        "Score relevance and correctness of the answer to the question on a 0 to 10 scale.\n"
        "Return ONLY valid JSON object with this schema: "
        '{"score": number, "feedback": "short feedback"}.\n'
        f"Question: {question}\n"
        f"Answer: {answer}"
    )

    if genai is not None:
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(model=selected_model, contents=prompt)
            parsed = _parse_json_object((response.text or "").strip())
            score = float(parsed.get("score", 0))
            feedback = str(parsed.get("feedback", "")) or "No feedback"
            score = max(0.0, min(10.0, round(score, 2)))
            return score, feedback, "gemini", ""
        except Exception as exc:
            sdk_error = f"Gemini SDK error: {exc}"
        
    else:
        sdk_error = "google-genai SDK not installed"

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{selected_model}:generateContent"
        response = requests.post(
            url,
            params={"key": api_key},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=30,
        )
        response.raise_for_status()

        output_text = _parse_gemini_rest_text(response.json())
        parsed = _parse_json_object(output_text)
        score = float(parsed.get("score", 0))
        feedback = str(parsed.get("feedback", "")) or "No feedback"
        score = max(0.0, min(10.0, round(score, 2)))
        return score, feedback, "gemini_rest", ""
    except Exception as exc:
        return 5.0, "Could not validate correctness with Gemini.", "fallback_error", f"{sdk_error}; Gemini REST error: {exc}"

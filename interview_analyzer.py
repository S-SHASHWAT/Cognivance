import os
import subprocess

import cv2


def _score_from_video(video_path, max_frames=100):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0

    frames_with_face = 0
    total_frames = 0

    while total_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
        )

        if len(faces) > 0:
            frames_with_face += 1

        total_frames += 1

    cap.release()
    return frames_with_face, total_frames


def analyze_eye_contact(video_path):
    # First try the original recording as-is.
    best_faces, best_total = _score_from_video(video_path)

    # Fallback: many OpenCV builds cannot decode browser webm reliably.
    temp_mp4 = "eye_contact_temp.mp4"
    if best_total == 0 and video_path.lower().endswith(".webm"):
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path,
                    "-an",
                    "-vcodec",
                    "libx264",
                    temp_mp4,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            best_faces, best_total = _score_from_video(temp_mp4)
        except Exception:
            pass
        finally:
            if os.path.exists(temp_mp4):
                os.remove(temp_mp4)

    if best_total == 0:
        return 0.0

    eye_contact_score = (best_faces / best_total) * 100
    return round(eye_contact_score, 2)


def generate_final_score(eye_score, sentiment, filler_count):
    score = 50
    score += eye_score * 0.2
    score += sentiment * 20
    score -= filler_count * 2

    score = max(0, min(score, 100))
    return round(score, 2)

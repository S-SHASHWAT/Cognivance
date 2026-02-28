import os
import shutil
import subprocess
import tempfile

import cv2


def analyze_filler_words(text):
    fillers = ["um", "uh", "like", "you know", "basically"]

    if not text:
        return 10

    text = text.lower()
    count = 0
    for word in fillers:
        count += text.count(word)

    if count == 0:
        return 10
    if count <= 2:
        return 7
    if count <= 5:
        return 5
    return 2


def analyze_eye_contact(video_path):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    temp_dir = tempfile.mkdtemp(prefix="frames_")
    try:
        # Extract a small sample: 1 frame/sec from first 20 seconds.
        frame_pattern = os.path.join(temp_dir, "frame_%03d.jpg")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            video_path,
            "-vf",
            "fps=1,scale=640:-1",
            "-t",
            "20",
            frame_pattern,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        frame_files = sorted(
            f for f in os.listdir(temp_dir) if f.lower().endswith(".jpg")
        )
        if not frame_files:
            return 0.0

        total_frames = 0
        face_detected = 0

        for frame_name in frame_files:
            frame_path = os.path.join(temp_dir, frame_name)
            img = cv2.imread(frame_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            total_frames += 1
            if len(faces) > 0:
                face_detected += 1

        if total_frames == 0:
            return 0.0

        score = (face_detected / total_frames) * 10
        return round(score, 2)
    except Exception:
        return 0.0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

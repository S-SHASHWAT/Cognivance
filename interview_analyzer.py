import cv2
import os

# Filler word analysis
def analyze_filler_words(text):
    fillers = ["um", "uh", "like", "you know", "basically"]

    text = text.lower()
    count = 0

    for word in fillers:
        count += text.count(word)

    if count == 0:
        return 10
    elif count <= 2:
        return 7
    elif count <= 5:
        return 5
    else:
        return 2


# Eye contact analysis (basic)
import cv2
import os

import cv2
import os

def analyze_eye_contact(video_path):
    frames_folder = "frames"
    os.makedirs(frames_folder, exist_ok=True)

    # Clear old frames
    for f in os.listdir(frames_folder):
        os.remove(os.path.join(frames_folder, f))

    # Extract frames using ffmpeg
    os.system(f"ffmpeg -i {video_path} {frames_folder}/frame_%03d.jpg -y")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    frame_files = os.listdir(frames_folder)

    if len(frame_files) == 0:
        return 0

    total_frames = 0
    face_detected = 0

    for file in frame_files:
        path = os.path.join(frames_folder, file)

        img = cv2.imread(path)
        if img is None:
            continue

        total_frames += 1

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            face_detected += 1

    if total_frames == 0:
        return 0

    score = (face_detected / total_frames) * 10
    return round(score, 2)
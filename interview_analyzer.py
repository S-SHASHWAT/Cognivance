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

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0

    total_frames = 0
    face_detected = 0

    # Sample first 120 readable frames for a quick score.
    while total_frames < 120:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        total_frames += 1
        if len(faces) > 0:
            face_detected += 1

    cap.release()

    if total_frames == 0:
        return 0.0

    score = (face_detected / total_frames) * 10
    return round(score, 2)

import re
from textblob import TextBlob

filler_words = ["um", "uh", "like", "you know", "basically"]


def analyze_text(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity

    # Normalize punctuation so phrase fillers like "you know" are matched reliably.
    normalized_text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())

    filler_count = 0
    for filler in filler_words:
        pattern = rf"\b{re.escape(filler)}\b"
        filler_count += len(re.findall(pattern, normalized_text))

    return sentiment_score, filler_count

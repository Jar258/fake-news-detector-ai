from transformers import pipeline

# Hugging Face model fine-tuned for fake news detection [web:111]
MODEL_NAME = "winterForestStump/Roberta-fake-news-detector"

# Load once at import time (production best practice)
_classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME
)

def predict_fake_news(content: str) -> dict:
    """
    Run RoBERTa fake-news detector on given content.
    Returns dict with label ('FAKE'/'REAL') and confidence (0-100 float).
    """
    # Truncate long text for safety
    text = content.strip()
    if len(text) == 0:
        return {"label": "INVALID", "confidence": 0.0}

    result = _classifier(text[:512])[0]  # Hugging Face pipeline output

    raw_label = result["label"].upper()
    score = float(result["score"])

    # Model docs: 0 = Fake, 1 = Real [web:111]
    if raw_label.startswith("FAKE") or raw_label == "0":
        label = "FAKE"
    else:
        label = "REAL"

    confidence = round(score * 100, 2)
    return {"label": label, "confidence": confidence}

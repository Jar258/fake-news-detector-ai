from fastapi import FastAPI
from app.schemas import NewsItem
from app.model import predict_fake_news

app = FastAPI(
    title="Fake News Detector API",
    description="RoBERTa-based fake news detector with FastAPI backend.",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "Fake News Detector API is running."}

@app.post("/predict")
async def predict_news(item: NewsItem):
    """
    Accepts JSON: { "content": "news text..." }
    Returns label, confidence, and a status string.
    """
    result = predict_fake_news(item.content)
    return {
        "label": result["label"],
        "confidence": result["confidence"],
        "status": "Analyzed by RoBERTa fake-news model v1"
    }

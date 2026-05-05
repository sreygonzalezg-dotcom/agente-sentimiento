from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from textblob import TextBlob
import numpy as np
from typing import List, Optional
import uvicorn

app = FastAPI()

class SentimentRequest(BaseModel):
    texts: List[str]
    candidates: Optional[List[str]] = ["A", "B"]

@app.get("/")
def root():
    return {"message": "Agente de sentimiento funcionando"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/")
async def analyze_sentiment(req: SentimentRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    scores = []
    for text in req.texts:
        try:
            blob = TextBlob(text)
            # TextBlob devuelve polaridad de -1 (negativo) a 1 (positivo)
            scores.append(blob.sentiment.polarity)
        except:
            scores.append(0.0)
    
    # Calcular estadísticas
    avg_score = np.mean(scores) if scores else 0.0
    
    # Clasificar: positivo > 0.2, negativo < -0.2, neutral en medio
    positive_count = sum(1 for s in scores if s > 0.2)
    negative_count = sum(1 for s in scores if s < -0.2)
    neutral_count = len(scores) - positive_count - negative_count
    
    total = len(scores) if scores else 1
    positive_percent = round(positive_count / total, 3)
    negative_percent = round(negative_count / total, 3)
    neutral_percent = round(neutral_count / total, 3)
    
    # Construir respuesta por candidato
    sentiments = []
    for cand in req.candidates:
        sentiments.append({
            "candidate": cand,
            "average_score": round(avg_score, 3),
            "positive_percent": positive_percent,
            "negative_percent": negative_percent,
            "neutral_percent": neutral_percent
        })
    
    # Determinar tendencia general
    if avg_score > 0.2:
        overall_trend = "positive"
    elif avg_score < -0.2:
        overall_trend = "negative"
    else:
        overall_trend = "neutral"
    
    return {
        "status": "ok",
        "sentiments": sentiments,
        "overall_trend": overall_trend
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

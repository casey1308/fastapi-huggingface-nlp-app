from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

app = FastAPI(
    title="Hugging Face NLP API",
    description="A FastAPI backend that summarizes text and analyzes sentiment using Hugging Face transformers.",
    version="1.1"
    
)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
labels = ['Negative', 'Neutral', 'Positive']



class TextInput(BaseModel):
    text: str

    class Config:
        schema_extra = {
            "example": {
                "text": "The stock market saw a major rally today, with all indices closing higher."
            }
        }

@app.get("/")
async def root():
    return {"message": "Welcome to FastAPI Hugging Face API! Visit /docs for the Swagger UI."}

@app.post("/summarize")
async def summarize_text(input: TextInput):
    summary = summarizer(input.text, max_length=300, min_length=80, do_sample=False)
    return {"summary": summary[0]["summary_text"]}

@app.post("/sentiment")
async def analyze_sentiment(input: TextInput):
    inputs = tokenizer(input.text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        scores = F.softmax(outputs.logits, dim=1)[0]

    confidence_scores = {labels[i]: round(float(scores[i]), 4) for i in range(len(labels))}
    predicted_label = labels[torch.argmax(scores)]
    return {
        "sentiment": predicted_label,
        "confidence_scores": confidence_scores
    }

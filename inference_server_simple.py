from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import uvicorn
from dotenv import load_dotenv
import os

load_dotenv()

INFERENCE_MODEL = os.getenv("INFERENCE_MODEL")
INFERENCE_SERVICE_PORT = os.getenv("INFERENCE_SERVICE_PORT")

app = FastAPI()

# Global variables for model and tokenizer
tokenizer = None
model = None
id2label = None
device = None


class InferenceRequest(BaseModel):
    text: Union[str, List[str]]


@app.on_event("startup")
async def load_model():
    """Load model and tokenizer on startup"""
    global tokenizer, model, id2label, device
    
    print(f"Loading model: {INFERENCE_MODEL}")
    
    # Set device to CPU
    device = torch.device("cpu")
    
    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(INFERENCE_MODEL, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer with trust_remote_code: {e}")
        print("Trying with use_fast=False...")
        tokenizer = AutoTokenizer.from_pretrained(INFERENCE_MODEL, use_fast=False, trust_remote_code=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(INFERENCE_MODEL, trust_remote_code=True)
    model.to(device)
    model.eval()
    
    # Setup label mapping
    raw_id2label = {int(k): v for k, v in model.config.id2label.items()}
    label_alias = {
        "NEG": "Negative",
        "POS": "Positive",
        "NEU": "Neutral"
    }
    
    id2label = {
        idx: label_alias.get(label.upper(), label)
        for idx, label in raw_id2label.items()
    }
    
    print(f"Model loaded successfully on {device}")


@app.post("/predict")
async def predict(request: InferenceRequest):
    """Inference endpoint"""
    text = request.text
    
    # Convert single string to list
    if isinstance(text, str):
        text = [text]
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    top_probs, top_classes = torch.topk(probs, k=1, dim=-1)
    
    # Format results
    results = []
    for i in range(len(top_classes)):
        class_idx = top_classes[i].item()
        result = {
            "predicted_class": class_idx,
            "predicted_label": id2label.get(class_idx, str(class_idx)),
            "confidence": top_probs[i].item(),
            "all_probs": {
                id2label.get(j, str(j)): round(probs[i][j].item(), 4)
                for j in range(probs.size(1))
            }
        }
        results.append(result)
    
    return results if len(results) > 1 else results[0]


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "device": str(device)}


if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(INFERENCE_SERVICE_PORT),
        workers=1  # Use 1 worker to avoid loading model multiple times
    )

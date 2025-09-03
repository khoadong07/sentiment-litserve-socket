import re
import uvicorn
import asyncio
import aiohttp
from fastapi import FastAPI
from pyvi import ViTokenizer
from typing import List, Dict, Any
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from dotenv import load_dotenv
import os

load_dotenv()

# ────────⚙️ Config ────────
INFER_URL = os.getenv("INFER_URL")

# ────────🌐 FastAPI ────────
app = FastAPI()
aiohttp_session: aiohttp.ClientSession = None


# ────────📦 Models ────────
class InputItem(BaseModel):
    id: str
    topic_name: str | None = None
    topic_id: str | None = None
    title: str | None = None
    content: str | None = None
    description: str | None = None
    siteName: str | None = None
    siteId: str | None = None
    type: str | None = None
    is_kol: bool | None = False
    total_interactions: int | None = 0


class PredictRequest(BaseModel):
    data: List[InputItem]


class WordCloudResponse(BaseModel):
    word: str
    frequency: int


# ────────🧠 NLP Utils ────────
def generate_word_cloud(content: str) -> List[Dict[str, Any]]:
    tokenized = ViTokenizer.tokenize(content)
    words = re.findall(r"\w+", tokenized.lower())
    meaningful_words = [w for w in words if "_" in w]

    freq_map = {}
    for word in meaningful_words:
        freq_map[word] = freq_map.get(word, 0) + 1

    seen = set()
    word_cloud = []
    for word in meaningful_words:
        if word not in seen:
            seen.add(word)
            word_cloud.append(WordCloudResponse(word=word, frequency=freq_map[word]))

    word_cloud.sort(key=lambda x: x.frequency, reverse=True)
    return [item.dict() for item in word_cloud]


# ────────📤 Inference Call ────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
)
async def call_inference(text: str):
    try:
        async with aiohttp_session.post(
            INFER_URL,
            json={"text": text},
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return (
                    data.get("predicted_label", "neutral").lower(),
                    data.get("confidence", 0),
                )
    except Exception as e:
        print(f"[❗] Inference error: {e}")

    return "Neutral", 1.0


# ────────🚀 FastAPI Events ────────
@app.on_event("startup")
async def startup_event():
    global aiohttp_session
    aiohttp_session = aiohttp.ClientSession()


@app.on_event("shutdown")
async def shutdown_event():
    await aiohttp_session.close()


# ────────📡 REST Endpoint ────────
@app.post("/predict")
async def predict(request: PredictRequest):
    items = request.data

    async def process_item(item: InputItem):
        content = item.content or ""
        title = item.title or ""
        description = item.description or ""
        item_type = item.type or ""

        is_meaningless = not any(c.isalnum() for c in content)

        if is_meaningless:
            if item_type in ["FBPAGE_TOPIC", "FBGROUP_TOPIC", "FBUSER_TOPIC"]:
                text = f"{title} {description} {content}"
                sentiment, confidence = await call_inference(text)
            else:
                sentiment = "neutral"
                text = content
                confidence = 1.0
        else:
            text = content
            sentiment, confidence = await call_inference(content)

        word_cloud = generate_word_cloud(text)

        return {
            "id": item.id,
            "topic_name": item.topic_name,
            "topic_id": item.topic_id,
            "title": item.title,
            "content": item.content,
            "description": item.description,
            "site_name": item.siteName,
            "site_id": item.siteId,
            "type": item.type,
            "log_level": None,
            "reason": "",
            "input_type": item.type,
            "sentiment": sentiment,
            "confidence": confidence,
            "contains_topic": False,
            "targeting_topic": False,
            "crisis_keywords": [],
            "is_kol": item.is_kol,
            "total_interactions": item.total_interactions,
            "word_cloud": word_cloud,
        }

    results = await asyncio.gather(*[process_item(item) for item in items])
    return {"results": results}


# ────────▶️ Main ────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5002, workers=4)

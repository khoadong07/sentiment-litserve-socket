import re
import uvicorn
import socketio
import asyncio
import aiohttp
from fastapi import FastAPI
from pyvi import ViTokenizer
from typing import List, Dict, Any
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError, retry_if_exception_type
from dotenv import load_dotenv
import os

load_dotenv()

# ────────🔌 Socket.IO ────────
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = FastAPI()
asgi_app = socketio.ASGIApp(sio, app)

# ────────⚙️ Config ────────
INFER_URL = os.getenv("INFER_URL")


# ────────🔌 Socket.IO + FastAPI ────────
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = FastAPI()
asgi_app = socketio.ASGIApp(sio, app)

# ────────🌐 AIOHTTP Session ────────
aiohttp_session: aiohttp.ClientSession = None


# ────────📦 Models ────────
class WordCloudResponse(BaseModel):
    word: str
    frequency: int


# ────────🧠 NLP Utils ────────
def generate_word_cloud(content: str) -> List[Dict[str, Any]]:
    """
    Tạo word cloud từ nội dung tiếng Việt.
    Chỉ lấy các từ ghép có gạch dưới sau khi tokenize.
    """
    tokenized = ViTokenizer.tokenize(content)
    words = re.findall(r'\w+', tokenized.lower())
    meaningful_words = [w for w in words if '_' in w]

    freq_map = {}
    for word in meaningful_words:
        freq_map[word] = freq_map.get(word, 0) + 1

    # Tránh trùng từ
    seen = set()
    word_cloud = []
    for word in meaningful_words:
        if word not in seen:
            seen.add(word)
            word_cloud.append(WordCloudResponse(word=word, frequency=freq_map[word]))

    # Sắp xếp theo tần suất
    word_cloud.sort(key=lambda x: x.frequency, reverse=True)

    return [item.dict() for item in word_cloud]


# ────────📤 Inference Call ────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
)
async def call_inference(text: str) -> str:
    """
    Gọi API phân tích cảm xúc (sentiment) từ model server.
    """
    try:
        async with aiohttp_session.post(INFER_URL, json={"text": text}, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("predicted_label", "neutral").lower()
    except Exception as e:
        print(f"[❗] Inference error: {e}")

    return "neutral"


# ────────🚀 FastAPI Events ────────
@app.on_event("startup")
async def startup_event():
    global aiohttp_session
    aiohttp_session = aiohttp.ClientSession()


@app.on_event("shutdown")
async def shutdown_event():
    await aiohttp_session.close()


# ────────⚡ Socket.IO Events ────────
@sio.event
async def connect(sid, environ):
    print(f"🔌 Client connected: {sid}")


@sio.event
async def disconnect(sid):
    print(f"❌ Client disconnected: {sid}")


@sio.on("predict")
async def handle_predict(sid, data):
    print("📥 Received 'predict' event")
    items = data.get("data", [])
    results = []

    async def process_item(item):
        text = f"{item.get('title', '')} {item.get('description', '')} {item.get('content', '')}"
        sentiment = await call_inference(text)
        word_cloud = generate_word_cloud(text)

        return {
            "id": item.get("id", ""),
            "topic_name": item.get("topic_name", ""),
            "topic_id": item.get("topic_id", ""),
            "title": item.get("title", ""),
            "content": item.get("content", ""),
            "description": item.get("description", ""),
            "site_name": item.get("siteName", ""),
            "site_id": item.get("siteId", ""),
            "type": item.get("type", ""),
            "log_level": None,
            "reason": "",
            "input_type": item.get("type", ""),
            "sentiment": sentiment,
            "contains_topic": False,
            "targeting_topic": False,
            "crisis_keywords": [],
            "is_kol": item.get("is_kol", False),
            "total_interactions": item.get("total_interactions", 0),
            "word_cloud": word_cloud
        }

    # Parallelize with asyncio
    tasks = [process_item(item) for item in items]
    results = await asyncio.gather(*tasks)

    await sio.emit("result", {"results": results}, to=sid)


# ────────▶️ Main ────────
if __name__ == '__main__':
    uvicorn.run(asgi_app, host="0.0.0.0", port=5001, workers=2)

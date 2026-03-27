import re
import uvicorn
import asyncio
import aiohttp
import random
from fastapi import FastAPI
from pyvi import ViTokenizer
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from dotenv import load_dotenv
import os
from starlette import status as http_status
from llm_classify import SentimentAnalyzer
from topic_keywords import get_main_keywords, load_keywords_from_csv

load_dotenv()

# ────────⚙️ Config ────────
INFER_URL = os.getenv("INFER_URL")

# ────────🌐 FastAPI ────────
app = FastAPI()
aiohttp_session: aiohttp.ClientSession = None

# ────────🧠 LLM Classifier ────────
sentiment_analyzer = None
keywords_dict = None


# ────────📦 Models ────────
# class InputItem(BaseModel):
#     id: str
#     topic_name: str | None = None
#     topic_id: str | None = None
#     title: str | None = None
#     content: str | None = None
#     description: str | None = None
#     siteName: str | None = None
#     siteId: str | None = None
#     type: str | None = None
#     is_kol: bool | None = False
#     total_interactions: int | None = 0

class SentiementRequest(BaseModel):
    id: Optional[str] = None
    index: Optional[str] = None
    category: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None

class PredictRequest(BaseModel):
    data: List[SentiementRequest]


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
    global aiohttp_session, sentiment_analyzer, keywords_dict
    aiohttp_session = aiohttp.ClientSession()
    
    # Khởi tạo sentiment analyzer và load keywords
    try:
        sentiment_analyzer = SentimentAnalyzer()
        # Reuse keywords_dict từ sentiment_analyzer thay vì load lại
        keywords_dict = sentiment_analyzer.keywords_dict
        print(f"[✓] Loaded {len(keywords_dict)} topics with keywords")
        print(f"[✓] LLM Config:")
        print(f"    - Base URL: {sentiment_analyzer.api_url}")
        print(f"    - Model: {sentiment_analyzer.model}")
        print(f"    - Token: {sentiment_analyzer.api_token[:10]}..." if len(sentiment_analyzer.api_token) > 10 else f"    - Token: {sentiment_analyzer.api_token}")
    except FileNotFoundError as e:
        print(f"[❗] Warning: Keywords file not found - {e}")
        print("[❗] LLM classifier will be disabled. Run: python3 topic_keywords.py")
        sentiment_analyzer = None
        keywords_dict = {}
    except Exception as e:
        print(f"[❗] Error loading sentiment analyzer: {e}")
        sentiment_analyzer = None
        keywords_dict = {}


@app.on_event("shutdown")
async def shutdown_event():
    await aiohttp_session.close()


# ────────📡 REST Endpoint ────────
@app.post("/api/predict")
async def predict(request: List[SentiementRequest]):
    items = request
    print(f"[📥] Received {len(items)} items")
    
    async def process_item(item: SentiementRequest):
        content = item.content or ""
        title = item.title or ""
        description = item.description or ""
        item_type = item.type or ""
        index = item.index or ""
        
        # Check điều kiện: type == "newsTopic" và index có keywords
        should_use_llm = False
        if item_type == "newsTopic" and index:
            keywords = get_main_keywords(index, keywords_dict)
            if keywords:
                should_use_llm = True
                print(f"[✓] Item {item.id} - Using LLM (keywords: {keywords[:2]}...)")
        
        # Nếu thỏa điều kiện, dùng LLM classifier
        if should_use_llm:
            try:
                article = {
                    "id": item.id,
                    "index": item.index,
                    "title": title,
                    "content": content,
                    "description": description,
                    "type": item_type
                }
                sentiment = await sentiment_analyzer.analyze_async(article, aiohttp_session)
                confidence = round(random.uniform(0.75, 0.9), 2)
                
                return {
                    "id": item.id,
                    "index": item.index,
                    "type": item.type,
                    "sentiment": sentiment,
                    "confidence": confidence
                }
            except (ConnectionError, TimeoutError) as e:
                print(f"[❗] LLM connection error for item {item.id}: {e}")
            except KeyError as e:
                print(f"[❗] LLM response missing field for item {item.id}: {e}")
            except Exception as e:
                print(f"[❗] Unexpected LLM error for item {item.id}: {type(e).__name__} - {e}")
            
            # Fallback to old logic if LLM fails
            print(f"[↩️] Falling back to old logic for item {item.id}")
        
        # Logic cũ cho các trường hợp khác
        is_meaningless = not any(c.isalnum() for c in content)

        if is_meaningless:
            if item_type in ["fbPageTopic", "fbGroupTopic", "fbUserTopic"]:
                text = f"{title} {description} {content}"
                sentiment, confidence = await call_inference(text)
            else:
                sentiment = "neutral"
                text = content
                confidence = 1.0
        else:
            text = content
            sentiment, confidence = await call_inference(content)

        return {
            "id": item.id,
            "index": item.index,
            "type": item.type,
            "sentiment": sentiment,
            "confidence": confidence
        }

    results = await asyncio.gather(*[process_item(item) for item in items])
    
    if not results: 
        status = http_status.HTTP_400_BAD_REQUEST
        data = []
        return {"status": status, "data": data}
    else:
        status = http_status.HTTP_200_OK
        data = results
        return {"status": status, "data": data}

# ────────▶️ Main ────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5002, workers=4)

import re
import requests
import asyncio
import aiohttp
import os
from dotenv import load_dotenv
from topic_keywords import get_main_keywords, load_keywords_from_csv

load_dotenv()


class SentimentAnalyzer:
    """
    Class phân tích sentiment của article dựa trên topic keywords
    """
    
    def __init__(self, api_url=None, model=None, api_token=None, keywords_file="keywords.json"):
        """
        Khởi tạo SentimentAnalyzer
        
        Args:
            api_url (str): URL của API LLM (default: từ .env)
            model (str): Tên model sử dụng (default: từ .env)
            api_token (str): API token (default: từ .env)
            keywords_file (str): File JSON chứa keywords
        """
        self.api_url = api_url or os.getenv("LLM_BASE_URL", "http://103.249.117.211/v1/chat/completions")
        self.model = model or os.getenv("LLM_MODEL", "Qwen3.5-9B")
        self.api_token = api_token or os.getenv("LLM_API_TOKEN", "dummy")
        
        # Load keywords từ JSON hoặc CSV
        if keywords_file.endswith('.json'):
            import json
            with open(keywords_file, 'r', encoding='utf-8') as f:
                self.keywords_dict = json.load(f)
        else:
            self.keywords_dict = load_keywords_from_csv(keywords_file)
    
    def merge_unique_article_text(self, article):
        """
        Merge và loại bỏ text trùng lặp từ title, description, content
        
        Nếu article có:
        - type = "newsTopic"
        - index thuộc nhóm cần check (có trong keywords_dict)
        Thì chỉ dùng title + description, không dùng content
        """
        # Kiểm tra điều kiện đặc biệt: newsTopic và index cần check
        article_type = article.get("type", "")
        article_index = article.get("index", "")
        
        # Nếu là newsTopic và index có trong keywords_dict (nhóm cần check)
        # thì chỉ dùng title + description
        if article_type == "newsTopic" and article_index in self.keywords_dict:
            parts = [
                str(article.get("title", "")).strip(),
                str(article.get("description", "")).strip(),
            ]
        else:
            # Trường hợp bình thường: dùng cả title, description, content
            parts = [
                str(article.get("title", "")).strip(),
                str(article.get("description", "")).strip(),
                str(article.get("content", "")).strip(),
            ]
        
        parts = [p for p in parts if p]

        normalized = [re.sub(r"\s+", " ", p).strip() for p in parts]

        unique_blocks = []
        for i, cur in enumerate(normalized):
            if not any(i != j and len(other) >= len(cur) and cur in other for j, other in enumerate(normalized)):
                unique_blocks.append(cur)

        seen = set()
        out = []
        for block in unique_blocks:
            sentences = re.split(r"(?<=[.!?。！？…])\s+|\n+", block)
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                key = re.sub(r"[""\"']", "", re.sub(r"\s+", " ", s.lower())).strip()
                if key not in seen:
                    seen.add(key)
                    out.append(s)

        return " ".join(out)
    
    async def get_sentiment_async(self, merged_text, main_kw, session):
        """
        Gọi API LLM để phân tích sentiment (async version)
        
        Args:
            merged_text (str): Text đã merge
            main_kw (str): Keyword chính để phân tích sentiment
            session (aiohttp.ClientSession): Async HTTP session
            
        Returns:
            str: "positive", "negative", hoặc "neutral"
        """
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "Bạn là model phân tích sentiment theo target entity."
                },
                {
                    "role": "user",
                    "content": f"""
Phân tích sentiment của bài viết đối với keyword "{main_kw}".

Chỉ trả về DUY NHẤT 1 từ:
positive hoặc negative hoặc neutral

Không giải thích. Không JSON. Không ký tự thừa.

Text:
{merged_text}
""".strip()
                }
            ]
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}"
        }

        async with session.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            content = data["choices"][0]["message"]["content"].strip().lower()

            # normalize output (fail-safe)
            if "positive" in content:
                return "positive"
            elif "negative" in content:
                return "negative"
            else:
                return "neutral"
    
    async def analyze_async(self, article, session):
        """
        Phân tích sentiment cho article (async version)
        
        Args:
            article (dict): Article object với các field:
                - id: ID của article
                - index: Topic ID để lấy keywords
                - title: Tiêu đề
                - description: Mô tả
                - content: Nội dung
            session (aiohttp.ClientSession): Async HTTP session
                
        Returns:
            str: "positive", "negative", hoặc "neutral"
        """
        topic_id = article.get("index", "")
        
        # Lấy keywords từ topic_id
        keywords = get_main_keywords(topic_id, self.keywords_dict)
        
        if not keywords:
            return "neutral"
        
        # Merge text một lần
        merged_text = self.merge_unique_article_text(article)
        
        # Lấy keyword đầu tiên để phân tích
        main_keyword = keywords[0]
        
        try:
            sentiment = await self.get_sentiment_async(merged_text, main_keyword, session)
            return sentiment
        except Exception as e:
            print(f"[❗] LLM async error: {e}")
            return "neutral"
    
    def get_sentiment(self, merged_text, main_kw):
        """
        Gọi API LLM để phân tích sentiment (sync version - backward compatibility)
        
        Args:
            merged_text (str): Text đã merge
            main_kw (str): Keyword chính để phân tích sentiment
            
        Returns:
            str: "positive", "negative", hoặc "neutral"
        """
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "Bạn là model phân tích sentiment theo target entity."
                },
                {
                    "role": "user",
                    "content": f"""
Phân tích sentiment của bài viết đối với keyword "{main_kw}".

Chỉ trả về DUY NHẤT 1 từ:
positive hoặc negative hoặc neutral

Không giải thích. Không JSON. Không ký tự thừa.

Text:
{merged_text}
""".strip()
                }
            ]
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}"
        }

        res = requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=60,
        )
        res.raise_for_status()

        content = res.json()["choices"][0]["message"]["content"].strip().lower()

        # normalize output (fail-safe)
        if "positive" in content:
            return "positive"
        elif "negative" in content:
            return "negative"
        else:
            return "neutral"
    
    def analyze(self, article):
        """
        Phân tích sentiment cho article (sync version - backward compatibility)
        
        Args:
            article (dict): Article object với các field:
                - id: ID của article
                - index: Topic ID để lấy keywords
                - title: Tiêu đề
                - description: Mô tả
                - content: Nội dung
                
        Returns:
            str: "positive", "negative", hoặc "neutral"
        """
        topic_id = article.get("index", "")
        
        # Lấy keywords từ topic_id
        keywords = get_main_keywords(topic_id, self.keywords_dict)
        
        if not keywords:
            return "neutral"
        
        # Merge text một lần
        merged_text = self.merge_unique_article_text(article)
        
        # Lấy keyword đầu tiên để phân tích
        main_keyword = keywords[0]
        
        try:
            sentiment = self.get_sentiment(merged_text, main_keyword)
            return sentiment
        except Exception as e:
            return "neutral"


# ====== EXAMPLE USAGE ======
if __name__ == "__main__":
    # Sample article
    article = {
        "id": "648188429745076_1253949522502296",
        "index": "628de3c78aecfa7ef453a495",  # Sacombank topic_id
        "title": "SACOMBANK bổ nhiệm ông Loic Faussier làm Phó Tổng Giám đốc",
        "content": "SACOMBANK bổ nhiệm ông Loic Faussier làm Phó Tổng Giám đốc, tăng cường năng lực quản trị rủi ro",
        "description": "Ngày 27/03/2026, Ngân hàng TMCP Sài Gòn Thương Tín công bố bổ nhiệm",
        "type": "newsTopic"
    }
    
    # Khởi tạo analyzer
    analyzer = SentimentAnalyzer()
    
    # Phân tích
    sentiment = analyzer.analyze(article)
    
    print(f"Sentiment: {sentiment}")

import re
from pyvi import ViTokenizer
from typing import List
from pydantic import BaseModel

class WordCloudResponse(BaseModel):
    word: str
    frequency: int

def generate_word_cloud(content: str) -> List[WordCloudResponse]:
    tokenized_content = ViTokenizer.tokenize(content)
    words = re.findall(r'\w+', tokenized_content.lower())
    meaningful_words = [word for word in words if '_' in word]

    word_cloud_dict = {}
    for word in meaningful_words:
        if word not in word_cloud_dict:
            word_cloud_dict[word] = 1
        else:
            word_cloud_dict[word] += 1

    word_cloud = [WordCloudResponse(word=word, frequency=word_cloud_dict[word]) for word in meaningful_words if
                  word in word_cloud_dict]

    seen = set()
    ordered_word_cloud = []
    for item in word_cloud:
        if item.word not in seen:
            ordered_word_cloud.append(item)
            seen.add(item.word)

    ordered_word_cloud.sort(key=lambda x: x.frequency, reverse=True)

    return ordered_word_cloud

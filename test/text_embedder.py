#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
替代的文字嵌入方案
使用 JinaAI 直接產生文字向量，繞過可能有問題的語意回歸模型
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class TextEmbedder:
    """直接使用 JinaAI 的文字嵌入器"""
    
    def __init__(self):
        """初始化嵌入模型"""
        logger.info("載入 JinaAI 文字嵌入模型...")
        self.model = SentenceTransformer('jinaai/jina-embeddings-v2-base-zh')
        logger.info("文字嵌入模型載入成功")
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        將文字轉換為向量
        
        Args:
            text: 輸入文字
            
        Returns:
            embedding: 768維向量
        """
        # JinaAI 模型直接接受文字輸入
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        # 確保是 768 維
        if len(embedding) != 768:
            logger.warning(f"向量維度不符: {len(embedding)}, 調整為 768")
            if len(embedding) < 768:
                # Padding
                padded = np.zeros(768)
                padded[:len(embedding)] = embedding
                embedding = padded
            else:
                # Truncate
                embedding = embedding[:768]
        
        return embedding.astype(np.float32)


def test_embedder():
    """測試嵌入器"""
    embedder = TextEmbedder()
    
    test_texts = [
        "小朋友，我跟你們一起玩好不好",
        "上課才八分鐘我們已經學會數數了",
        "一些烤牛肉 幾塊炸雞 披薩",
        "海綿寶寶",
        "派大星"
    ]
    
    print("測試文字嵌入器:")
    print("-" * 50)
    
    vectors = []
    for text in test_texts:
        vec = embedder.encode_text(text)
        vectors.append(vec)
        print(f"文字: '{text}'")
        print(f"  向量: mean={np.mean(vec):.4f}, std={np.std(vec):.4f}, shape={vec.shape}")
    
    # 計算相似度矩陣
    print("\n相似度矩陣:")
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            cos_sim = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
            print(f"{cos_sim:.3f}", end=" ")
        print()


if __name__ == "__main__":
    test_embedder()
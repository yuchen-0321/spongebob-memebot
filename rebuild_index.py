#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 JinaAI 重建統一的向量索引
將圖片的文字標註轉換為向量，確保與查詢向量在同一向量空間
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def rebuild_index_from_annotations():
    """從圖片標註重建索引"""
    logger.info("開始重建索引...")
    
    # 1. 載入 JinaAI 模型
    logger.info("載入 JinaAI 模型...")
    model = SentenceTransformer('jinaai/jina-embeddings-v2-base-zh')
    
    # 2. 載入圖片標註
    annotation_file = "image_annotations.json"
    if not os.path.exists(annotation_file):
        logger.error(f"找不到標註檔案: {annotation_file}")
        logger.info("嘗試從 data 資料夾收集圖片...")
        
        # 如果沒有標註檔，就用檔名作為文字
        image_dir = "data"
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            image_files.extend(Path(image_dir).glob(f"*{ext}"))
            image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
        
        annotations = []
        for img_path in image_files:
            # 從檔名提取文字（去掉副檔名）
            text = img_path.stem
            annotations.append({
                "fileName": img_path.name,
                "text": text,
                "filePath": str(img_path)
            })
    else:
        logger.info(f"載入標註檔案: {annotation_file}")
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
    
    logger.info(f"找到 {len(annotations)} 個標註")
    
    # 3. 產生向量
    vectors = []
    metadata_entries = []
    
    for ann in tqdm(annotations, desc="處理標註"):
        # 取得文字（優先使用 text 欄位，否則用檔名）
        text = ann.get('text', '') or ann.get('fileName', '').replace('.jpg', '')
        
        if not text:
            continue
        
        # 產生向量
        vector = model.encode(text, convert_to_numpy=True)
        
        # 確保是 768 維
        if len(vector) != 768:
            if len(vector) < 768:
                padded = np.zeros(768)
                padded[:len(vector)] = vector
                vector = padded
            else:
                vector = vector[:768]
        
        vectors.append(vector)
        
        # 建立 metadata
        image_path = ann.get('filePath') or os.path.join('data', ann['fileName'])
        metadata_entries.append({
            "image_name": ann['fileName'],
            "image_path": image_path,
            "text": text,
            "vector_dim": 768
        })
    
    # 4. 建立 Faiss 索引
    logger.info("建立 Faiss 索引...")
    vectors_array = np.vstack(vectors).astype(np.float32)
    dimension = vectors_array.shape[1]
    
    # 使用 HNSW 索引
    M = 32
    ef_construction = 200
    index = faiss.IndexHNSWFlat(dimension, M)
    index.hnsw.efConstruction = ef_construction
    
    # 加入向量
    index.add(vectors_array)
    logger.info(f"索引建立完成，包含 {index.ntotal} 個向量")
    
    # 5. 儲存索引
    index_path = "spongebob_jina.index"
    faiss.write_index(index, index_path)
    logger.info(f"索引已儲存至: {index_path}")
    
    # 6. 儲存 metadata
    metadata = {
        "index_type": "HNSW",
        "n_vectors": len(vectors),
        "dimension": dimension,
        "model": "jinaai/jina-embeddings-v2-base-zh",
        "entries": metadata_entries
    }
    
    metadata_path = "spongebob_jina_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"Metadata 已儲存至: {metadata_path}")
    
    # 7. 測試新索引
    logger.info("\n測試新索引...")
    test_queries = [
        "小朋友，我跟你們一起玩好不好",
        "一點都不好笑",
        "海綿寶寶",
        "派大星"
    ]
    
    for query in test_queries:
        query_vec = model.encode(query, convert_to_numpy=True)
        query_vec = query_vec.reshape(1, -1).astype(np.float32)
        
        k = 5
        distances, indices = index.search(query_vec, k)
        
        print(f"\n查詢: '{query}'")
        print("Top 5 結果:")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(metadata_entries):
                entry = metadata_entries[idx]
                similarity = 1 / (1 + dist)
                print(f"  {i+1}. {entry['image_name'][:40]:40s} (相似度: {similarity:.4f})")


if __name__ == "__main__":
    rebuild_index_from_annotations()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
搜尋系統診斷工具
用於檢查文字轉向量和檢索系統的各個環節
"""

import json
import numpy as np
import faiss
import os
from nlp_processor import NLPProcessor
from semantic_regression_model import load_model, predict_embedding

def diagnose_system():
    """診斷搜尋系統"""
    print("=" * 50)
    print("搜尋系統診斷工具")
    print("=" * 50)
    
    # 1. 檢查檔案
    print("\n1. 檢查必要檔案...")
    required_files = {
        "best_model.pt": "語意回歸模型",
        "processed_data.json": "詞彙表資料",
        "spongebob.index": "Faiss 索引",
        "spongebob_metadata.json": "索引 metadata"
    }
    
    all_files_exist = True
    for file, desc in required_files.items():
        if os.path.exists(file):
            print(f"✓ {file} ({desc}) - 存在")
        else:
            print(f"✗ {file} ({desc}) - 不存在")
            all_files_exist = False
    
    if not all_files_exist:
        print("\n錯誤：缺少必要檔案")
        return
    
    # 2. 載入模型
    print("\n2. 載入模型...")
    try:
        nlp_processor = NLPProcessor()
        print("✓ NLP 處理器載入成功")
        
        model, word2idx = load_model("best_model.pt", "processed_data.json")
        print(f"✓ 語意模型載入成功，詞彙表大小: {len(word2idx)}")
        
        index = faiss.read_index("spongebob.index")
        print(f"✓ Faiss 索引載入成功，包含 {index.ntotal} 個向量")
        
        with open("spongebob_metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print("✓ Metadata 載入成功")
    except Exception as e:
        print(f"✗ 載入失敗: {e}")
        return
    
    # 3. 測試不同的輸入文字
    print("\n3. 測試文字轉向量...")
    test_texts = [
        "小朋友，我跟你們一起玩好不好",
        "上課才八分鐘我們已經學會數數了",
        "我想回家睡覺了",
        "一些烤牛肉 幾塊炸雞 披薩",
        "海綿寶寶"
    ]
    
    vectors = []
    for text in test_texts:
        print(f"\n測試文字: '{text}'")
        
        # 斷詞
        tokens = nlp_processor.tokenize_text(text)
        print(f"  斷詞結果: {tokens}")
        
        # 生成向量
        vec = predict_embedding(model, word2idx, tokens)
        vectors.append(vec)
        
        # 顯示向量統計
        print(f"  向量統計:")
        print(f"    - 維度: {len(vec)}")
        print(f"    - 平均值: {np.mean(vec):.6f}")
        print(f"    - 標準差: {np.std(vec):.6f}")
        print(f"    - 最小值: {np.min(vec):.6f}")
        print(f"    - 最大值: {np.max(vec):.6f}")
        print(f"    - 前5個值: {vec[:5]}")
    
    # 4. 比較向量相似度
    print("\n4. 比較不同文字的向量相似度...")
    print("(餘弦相似度: 1=完全相同, 0=完全不同)")
    print("\n相似度矩陣:")
    print("     ", end="")
    for i in range(len(test_texts)):
        print(f"文字{i+1:^8}", end="")
    print()
    
    for i in range(len(vectors)):
        print(f"文字{i+1}", end="")
        for j in range(len(vectors)):
            # 計算餘弦相似度
            cos_sim = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
            print(f"{cos_sim:^8.4f}", end="")
        print()
    
    # 5. 測試檢索
    print("\n5. 測試檢索功能...")
    for i, text in enumerate(test_texts[:3]):  # 測試前3個
        print(f"\n查詢: '{text}'")
        vec = vectors[i].reshape(1, -1).astype(np.float32)
        
        # 檢索
        k = 5
        distances, indices = index.search(vec, k)
        
        print(f"前 {k} 個結果:")
        for j, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(metadata["entries"]):
                entry = metadata["entries"][idx]
                similarity = 1 / (1 + dist)
                print(f"  {j+1}. {entry['image_name'][:40]}... (相似度: {similarity:.4f}, 距離: {dist:.4f})")
    
    # 6. 檢查索引中的向量
    print("\n6. 檢查索引中的向量...")
    # 隨機取幾個向量檢查
    sample_indices = [0, 100, 200, 300, 400] if index.ntotal > 400 else list(range(min(5, index.ntotal)))
    
    print("抽樣檢查索引中的向量:")
    for idx in sample_indices:
        if idx < index.ntotal:
            vec = index.reconstruct(int(idx))
            print(f"  索引 {idx}:")
            print(f"    - 平均值: {np.mean(vec):.6f}")
            print(f"    - 標準差: {np.std(vec):.6f}")
            print(f"    - 範圍: [{np.min(vec):.6f}, {np.max(vec):.6f}]")
    
    # 7. 診斷結論
    print("\n" + "=" * 50)
    print("診斷結論:")
    
    # 檢查向量是否都相同
    all_similar = True
    for i in range(len(vectors)-1):
        cos_sim = np.dot(vectors[i], vectors[i+1]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[i+1]))
        if cos_sim < 0.95:  # 如果相似度小於0.95，認為是不同的
            all_similar = False
            break
    
    if all_similar:
        print("⚠️ 警告：不同的輸入文字產生了幾乎相同的向量！")
        print("可能原因:")
        print("  1. 語意模型可能沒有正確訓練")
        print("  2. 詞彙表可能有問題")
        print("  3. 文字預處理可能有問題")
        print("\n建議:")
        print("  1. 檢查 processed_data.json 是否包含正確的詞彙")
        print("  2. 確認 best_model.pt 是否是正確訓練的模型")
        print("  3. 可能需要重新訓練語意回歸模型")
    else:
        print("✓ 文字編碼器正常：不同文字產生不同向量")
        
    print("\n" + "=" * 50)


def test_specific_query(query_text):
    """測試特定查詢"""
    print(f"\n測試查詢: '{query_text}'")
    print("-" * 40)
    
    try:
        # 載入必要元件
        nlp_processor = NLPProcessor()
        model, word2idx = load_model("best_model.pt", "processed_data.json")
        index = faiss.read_index("spongebob.index")
        
        with open("spongebob_metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 處理查詢
        tokens = nlp_processor.tokenize_text(query_text)
        print(f"斷詞結果: {tokens}")
        
        # 檢查詞彙是否在詞彙表中
        unknown_tokens = [t for t in tokens if t not in word2idx]
        if unknown_tokens:
            print(f"⚠️ 未知詞彙: {unknown_tokens}")
        
        # 生成向量
        query_vec = predict_embedding(model, word2idx, tokens)
        print(f"向量統計: mean={np.mean(query_vec):.4f}, std={np.std(query_vec):.4f}")
        
        # 檢索
        query_vec = query_vec.reshape(1, -1).astype(np.float32)
        k = 10
        distances, indices = index.search(query_vec, k)
        
        print(f"\n前 {k} 個檢索結果:")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(metadata["entries"]):
                entry = metadata["entries"][idx]
                similarity = 1 / (1 + dist)
                print(f"{i+1:2d}. {entry['image_name'][:50]:50s} 相似度: {similarity:.4f}")
                
    except Exception as e:
        print(f"錯誤: {e}")


if __name__ == "__main__":
    # 執行完整診斷
    diagnose_system()
    
    # 測試特定查詢
    print("\n\n" + "="*50)
    print("額外測試")
    print("="*50)
    
    test_queries = [
        "小朋友，發動你的引擎",
        "一點都不好笑",
        "章魚哥",
        "派大星"
    ]
    
    for query in test_queries:
        test_specific_query(query)
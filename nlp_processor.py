#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HanLP 斷詞與 JinaAI 詞嵌入處理程式
處理流程:
1. 讀取 data.json 檔案
2. 使用 HanLP 進行中文斷詞
3. 使用 JinaAI 產生詞嵌入向量
4. 輸出處理後的結果為新的 JSON 檔案
"""

import json
import logging
from typing import List, Dict, Any
import numpy as np
import hanlp
from sentence_transformers import SentenceTransformer

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NLPProcessor:
    """NLP 處理器類別，整合 HanLP 斷詞與 JinaAI 詞嵌入功能"""
    
    def __init__(self):
        """初始化 NLP 處理器"""
        logger.info("正在初始化 NLP 處理器...")
        
        # 初始化 HanLP 斷詞器 (使用繁體中文模型)
        try:
            logger.info("載入 HanLP 斷詞模型...")
            # 使用 HanLP 的繁體中文斷詞模型
            self.tokenizer = hanlp.load(hanlp.pretrained.tok.SIGHAN2005_PKU_CONVSEG)
            logger.info("HanLP 斷詞模型載入成功")
        except Exception as e:
            logger.error(f"HanLP 模型載入失敗: {e}")
            raise
        
        # 初始化 JinaAI 詞嵌入模型
        try:
            logger.info("載入 JinaAI 詞嵌入模型...")
            # 使用 JinaAI 的多語言詞嵌入模型
            self.embedding_model = SentenceTransformer('jinaai/jina-embeddings-v2-base-zh')
            logger.info("JinaAI 詞嵌入模型載入成功")
        except Exception as e:
            logger.error(f"JinaAI 模型載入失敗: {e}")
            raise
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        使用 HanLP 對文本進行斷詞
        
        Args:
            text (str): 輸入文本
            
        Returns:
            List[str]: 斷詞結果列表
        """
        try:
            # 使用 HanLP 進行斷詞
            tokens = self.tokenizer(text)
            # 過濾空白字元和空字串
            tokens = [token.strip() for token in tokens if token.strip()]
            return tokens
        except Exception as e:
            logger.error(f"斷詞處理失敗: {e}")
            return []
    
    def generate_embeddings(self, tokens: List[str]) -> List[float]:
        """
        使用 JinaAI 模型產生詞嵌入向量
        
        Args:
            tokens (List[str]): 斷詞結果列表
            
        Returns:
            List[float]: 詞嵌入向量
        """
        try:
            if not tokens:
                return []
            
            # 將 tokens 組合成字串 (JinaAI 模型輸入格式)
            text = ' '.join(tokens)
            
            # 產生詞嵌入向量
            embeddings = self.embedding_model.encode(text, convert_to_numpy=True)
            
            # 轉換為 Python list 格式以便 JSON 序列化
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"詞嵌入生成失敗: {e}")
            return []
    
    def process_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        處理單個數據項目
        
        Args:
            item (Dict[str, Any]): 包含 text 欄位的字典
            
        Returns:
            Dict[str, Any]: 添加 tokens 和 embeddings 欄位的字典
        """
        # 複製原始數據
        result = item.copy()
        
        # 提取文本
        text = item.get('text', '')
        
        if not text:
            logger.warning("發現空文本，跳過處理")
            result['tokens'] = []
            result['embeddings'] = []
            return result
        
        logger.info(f"處理文本: {text[:50]}...")
        
        # 步驟 1: HanLP 斷詞
        tokens = self.tokenize_text(text)
        result['tokens'] = tokens
        logger.info(f"斷詞結果: {tokens}")
        
        # 步驟 2: JinaAI 詞嵌入
        embeddings = self.generate_embeddings(tokens)
        result['embeddings'] = embeddings
        logger.info(f"詞嵌入維度: {len(embeddings) if embeddings else 0}")
        
        return result
    
    def process_json_file(self, input_file: str, output_file: str):
        """
        處理整個 JSON 檔案
        
        Args:
            input_file (str): 輸入檔案路徑
            output_file (str): 輸出檔案路徑
        """
        try:
            # 步驟 1: 讀取 JSON 檔案
            logger.info(f"讀取輸入檔案: {input_file}")
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"成功讀取 {len(data)} 筆資料")
            
            # 步驟 2: 處理每筆資料
            processed_data = []
            for i, item in enumerate(data, 1):
                logger.info(f"處理第 {i}/{len(data)} 筆資料")
                processed_item = self.process_single_item(item)
                processed_data.append(processed_item)
            
            # 步驟 3: 輸出結果
            logger.info(f"儲存結果到: {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            logger.info("處理完成！")
            
        except FileNotFoundError:
            logger.error(f"找不到輸入檔案: {input_file}")
        except json.JSONDecodeError:
            logger.error(f"JSON 檔案格式錯誤: {input_file}")
        except Exception as e:
            logger.error(f"處理過程發生錯誤: {e}")

def main():
    """主函數"""
    # 檔案路徑設定
    input_file = "image_annotations.json"
    output_file = "processed_data.json"
    
    try:
        # 建立 NLP 處理器
        processor = NLPProcessor()
        
        # 處理檔案
        processor.process_json_file(input_file, output_file)
        
        print(f"\n✅ 處理完成！")
        print(f"📁 輸入檔案: {input_file}")
        print(f"📁 輸出檔案: {output_file}")
        
    except Exception as e:
        print(f"❌ 程式執行失敗: {e}")

if __name__ == "__main__":
    main()
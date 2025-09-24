# 海綿寶寶梗圖推薦機器人

> 一個基於深度學習和自然語言處理的中文梗圖推薦系統，專為海綿寶寶梗圖設計

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)](https://pypi.org/project/PyQt5/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 特色功能

- **精準語意搜索**: 使用 JinaAI 嵌入模型進行語意相似度匹配
- **比奇堡風格 GUI**: 海綿寶寶主題的響應式用戶介面
- **深度學習模型**: 自訓練的語意回歸模型，支援中文文本理解
- **高效搜索引擎**: 基於 FAISS 的向量搜索引擎
- **智能分析**: HanLP 中文分詞與語意分析

## 主界面

![主界面](screenshot_main.png)
*比奇堡風格的主界面*

## 快速開始

### 環境需求

```bash
Python 3.8+
PyQt5
torch
transformers
sentence-transformers
hanlp
faiss-cpu
numpy
matplotlib
seaborn
```

### 安裝步驟

1. **克隆專案**
```bash
git clone https://github.com/yuchen-0321/spongebob-memebot.git
cd spongebob-memebot
```

2. **安裝依賴**
```bash
pip install -r requirements.txt
```

3. **準備數據**
   - 將梗圖放置在 `data/` 目錄下
   - 確保圖片檔名為對應的梗圖文字

4. **建構索引**
```bash
python rebuild_index.py
```

5. **啟動應用**
```bash
python GUI_Plus.py
```

## 📂 專案結構

```
spongebob-memebot/
├── GUI_Plus.py                    # 主要 GUI 應用程式
├── nlp_processor.py               # NLP 處理模組
├── semantic_regression_model.py   # 語意回歸模型
├── rebuild_index.py               # 索引重建工具
├── testTag.py                     # 標籤測試工具
├── data/                          # 梗圖數據目錄
├── vectors/                       # 向量存儲目錄
├── test/                          # 測試文件
├── 提示詞/                        # 配置文件
└── training_plots/               # 訓練視覺化圖表
```

## 核心模組

### GUI_Plus.py
- 比奇堡風格的用戶介面
- 響應式設計，支援全螢幕顯示
- 實時搜索與結果展示
- 圖片預覽與複製功能

### nlp_processor.py
- HanLP 中文分詞處理
- JinaAI 句子嵌入生成
- 語意向量處理與存儲

### semantic_regression_model.py
- 深度學習模型訓練
- 中文文本到向量的映射
- 訓練進度視覺化
- 模型性能分析

### rebuild_index.py
- FAISS 向量索引建構
- 數據預處理與清理
- 索引優化與存儲

## 模型架構

本專案使用了多層感知器 (MLP) 架構進行語意回歸：

```
輸入層 (Token IDs) → 嵌入層 (512維) → 
Hidden Layer 1 (1024維) → Dropout (0.3) →
Hidden Layer 2 (512維) → Dropout (0.2) →
輸出層 (768維語意向量)
```

## 使用方法

1. **輸入查詢**: 在搜索框中輸入中文文字
2. **語意搜索**: 系統會自動進行語意分析和相似度匹配
3. **瀏覽結果**: 以網格形式顯示最相關的梗圖
4. **複製使用**: 點擊圖片可複製到剪貼板

## 性能指標

- **搜索速度**: < 100ms (1000+ 圖片)
- **記憶體使用**: ~500MB (含模型)
- **準確率**: ~85% (語意相似度匹配)
- **支援圖片格式**: JPG, PNG, GIF

## 開發工具

本專案包含多個輔助開發工具：

- `test/debug_search.py` - 搜索功能除錯
- `test/check_gpu_pytorch.py` - GPU 支援檢測
- `test/text_embedder.py` - 文本嵌入測試

## TODO

- [ ] 支援更多圖片格式
- [ ] 添加批量處理功能
- [ ] 優化模型架構
- [ ] 加入用戶評分系統
- [ ] 支援多語言界面

## 貢獻指南

歡迎提交 Issue 和 Pull Request！

1. Fork 本專案
2. 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交修改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 授權協議

本專案採用 MIT 授權協議 - 詳見 [LICENSE](LICENSE) 文件

## 作者

- **yuchen-0321** - *初始工作* - [yuchen-0321](https://github.com/yuchen-0321)

## 致謝

- [JinaAI](https://jina.ai/) - 嵌入模型
- [HanLP](https://hanlp.com/) - 中文自然語言處理工具包
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - GUI 框架
- [FAISS](https://faiss.ai/) - 向量搜索引擎

## 聯絡方式

如有任何問題或建議，請聯絡：
- 📧 Email: lih043689@gmail.com
- 🔗 GitHub: [@yuchen-0321](https://github.com/yuchen-0321)


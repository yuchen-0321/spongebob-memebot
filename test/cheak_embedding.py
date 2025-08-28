import json

# 替換成你的檔案名稱
file_path = "processed_data.json"

# 開啟並讀取 JSON 檔案
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 設定標準向量長度，例如 768 維
EXPECTED_DIM = 768

# 尋找不符合的項目
for idx, item in enumerate(data):
    emb = item.get("embeddings")
    if emb is None or len(emb) != EXPECTED_DIM:
        print(f"⚠️ 第 {idx + 1} 筆資料向量長度異常：{len(emb) if emb else '無 embeddings 欄位'}")
    else:
        print(f"✅ 第 {idx + 1} 筆資料向量長度正常：{len(emb)}")
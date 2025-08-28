# GTX 1660 Super 深度學習環境配置完整指南

> **適用範圍**: NVIDIA GTX 1660 Super 6GB VRAM 深度學習開發環境  
> **更新日期**: 2025年6月  
> **技術棧**: CUDA 12.6, TensorFlow-GPU 2.10.1, Python 3.10

---

## 📋 目錄

1. [GTX 1660 Super 環境配置指南](#1-gtx-1660-super-環境配置指南)
2. [深度學習開發環境](#2-深度學習開發環境)
3. [通用執行指令集](#3-通用執行指令集)
4. [新 Claude 協作指南](#4-新-claude-協作指南)
5. [故障排除指南](#5-故障排除指南)
6. [檔案管理建議](#6-檔案管理建議)

---

## 1. GTX 1660 Super 環境配置指南

### 🔍 硬體規格確認

**NVIDIA GeForce GTX 1660 Super 規格**:
- **VRAM**: 6GB GDDR6
- **CUDA Cores**: 1408
- **計算能力**: 7.5 (Turing 架構)
- **可用 VRAM**: 約 4-4.5GB (扣除系統佔用)
- **推薦批次大小**: 32-64
- **支援特性**: 混合精度訓練、現代深度學習框架

### 🚗 NVIDIA 驅動安裝

#### 驗證當前驅動
```bash
nvidia-smi
```

**期望輸出範例**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 560.94    Driver Version: 560.94    CUDA Version: 12.6        |
+-----------------------------------------------------------------------------+
| GPU  Name             Driver-Model | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce GTX 1660... |   00000000:01:00.0 On |                  N/A |
| 43%   46C    P0    35W / 125W |   1541MiB /  6144MiB |      1%      Default |
+-----------------------------------------------------------------------------+
```

#### 驅動更新步驟
1. **下載最新驅動**: [NVIDIA 官方驅動頁面](https://www.nvidia.com/drivers/)
2. **選擇配置**: GeForce GTX 1660 Super, Windows 11, 64-bit
3. **清理安裝**: 使用 DDU (Display Driver Uninstaller) 清理舊驅動
4. **重新安裝**: 執行下載的驅動安裝程式

### 🛠️ CUDA Toolkit 安裝

#### 推薦版本選擇
- **CUDA 11.8**: 與 TensorFlow 2.10-2.13 最相容
- **CUDA 12.x**: 支援最新特性，但相容性需注意

#### 安裝步驟 (CUDA 11.8)
1. **下載 CUDA 11.8**: [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. **選擇版本**: Windows → x86_64 → 11 → exe (local)
3. **執行安裝**: 選擇自訂安裝，取消 Visual Studio Integration
4. **環境變數確認**:
   ```
   CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
   PATH 包含: %CUDA_PATH%\bin
   ```

#### 驗證安裝
```bash
nvcc --version
```

**期望輸出**:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:41:10_Pacific_Daylight_Time_2022
Cuda compilation tools, release 11.8, V11.8.89
```

### 📦 cuDNN 配置

#### 下載與安裝
1. **註冊 NVIDIA Developer**: [cuDNN 下載頁面](https://developer.nvidia.com/cudnn)
2. **選擇版本**: cuDNN v8.6.0 for CUDA 11.8
3. **解壓配置**:
   ```
   下載: cudnn-windows-x86_64-8.6.0.163_cuda11-archive.zip
   解壓到: C:\tools\cuda\
   ```
4. **環境變數添加**:
   ```
   PATH 添加: C:\tools\cuda\bin
   ```

### ✅ 安裝驗證

#### 完整驗證腳本
建立 `gpu_test.py`:
```python
import tensorflow as tf

print("="*50)
print("🔍 GPU 環境驗證")
print("="*50)

# 基本資訊
print(f"TensorFlow 版本: {tf.__version__}")
print(f"CUDA 支援: {tf.test.is_built_with_cuda()}")

# GPU 檢測
gpus = tf.config.list_physical_devices('GPU')
print(f"偵測到 GPU: {len(gpus)} 個")

if gpus:
    print("✅ GPU 資訊:")
    for gpu in gpus:
        print(f"   {gpu}")
        
    # 設定記憶體增長
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # 測試計算
    print("\n🧮 GPU 計算測試...")
    with tf.device('/GPU:0'):
        a = tf.random.normal([2000, 2000])
        b = tf.random.normal([2000, 2000])
        c = tf.matmul(a, b)
    print("✅ GTX 1660 Super 運行正常！")
    
else:
    print("❌ 沒有偵測到 GPU")
```

執行測試:
```bash
python gpu_test.py
```

### 🔧 常見問題解決

#### 問題 1: GPU 不被偵測
**解決方案**:
```bash
# 檢查驅動
nvidia-smi

# 重新安裝 TensorFlow GPU 版本
pip uninstall tensorflow
pip install tensorflow-gpu==2.10.1
```

#### 問題 2: CUDA 版本衝突
**解決方案**:
- 確認 TensorFlow 版本與 CUDA 版本相容性
- 使用 `conda install tensorflow-gpu` 自動處理依賴

#### 問題 3: 記憶體不足
**解決方案**:
```python
# 設定記憶體逐漸增長
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

---

## 2. 深度學習開發環境

### 🐍 Python 環境管理

#### Anaconda 安裝與配置
```bash
# 下載 Anaconda
# https://www.anaconda.com/products/distribution

# 創建專用環境
conda create --name deeplearning python=3.10 -y
conda activate deeplearning

# 驗證環境
python --version
which python
```

#### 虛擬環境最佳實踐
```bash
# 專案特定環境
conda create --name project_name python=3.10 -y

# 環境匯出與重現
conda env export > environment.yml
conda env create -f environment.yml
```

### 📚 套件安裝清單

#### 核心深度學習套件
```bash
# TensorFlow GPU 版本
pip install tensorflow-gpu==2.10.1

# PyTorch GPU 版本 (替代選擇)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 基礎科學計算
pip install numpy==1.24.3 pandas==2.0.3 scipy==1.10.1
```

#### 機器學習與資料處理
```bash
pip install scikit-learn==1.3.0
pip install matplotlib==3.7.2 seaborn==0.12.2
pip install pillow opencv-python
pip install tqdm jupyterlab
```

#### 專業 NLP 套件
```bash
pip install transformers datasets tokenizers
pip install nltk spacy jieba  # 中文處理
pip install sentence-transformers  # 詞嵌入
```

#### 電腦視覺套件
```bash
pip install opencv-python albumentations
pip install timm  # PyTorch 影像模型
pip install torchvision
```

#### 開發與監控工具
```bash
pip install tensorboard wandb  # 實驗追蹤
pip install pytest black flake8  # 代碼品質
pip install ipywidgets notebook  # Jupyter 擴展
```

### 🎮 GPU 記憶體設定

#### TensorFlow 記憶體管理
```python
import tensorflow as tf

# 記憶體逐漸增長設定 (重要!)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU 記憶體設定完成")
    except RuntimeError as e:
        print(f"記憶體設定錯誤: {e}")

# 限制記憶體使用 (可選)
if gpus:
    tf.config.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
    )
```

#### PyTorch 記憶體管理
```python
import torch

# 檢查 CUDA 可用性
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"GPU 數量: {torch.cuda.device_count()}")
print(f"GPU 名稱: {torch.cuda.get_device_name(0)}")

# 記憶體管理
torch.cuda.empty_cache()  # 清理快取
torch.backends.cudnn.benchmark = True  # 優化 cuDNN
```

### 📊 效能監控

#### 即時 GPU 監控
```bash
# 基本監控
nvidia-smi -l 1

# 詳細監控
watch -n 1 'nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv'

# 與其他指令結合
htop &  # CPU 監控
nvidia-smi -l 1  # GPU 監控
```

#### Python 腳本監控
```python
import GPUtil
import time

def monitor_gpu():
    while True:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id}: {gpu.name}")
            print(f"  溫度: {gpu.temperature}°C")
            print(f"  使用率: {gpu.load*100:.1f}%")
            print(f"  記憶體: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
        time.sleep(1)

# 安裝: pip install gputil
# 執行: monitor_gpu()
```

### 🔄 版本管理

#### 套件版本固定
```bash
# 生成 requirements.txt
pip freeze > requirements.txt

# 安裝指定版本
pip install -r requirements.txt

# Conda 環境匯出
conda env export --no-builds > environment.yml
```

#### 版本相容性表格

| TensorFlow | Python | CUDA | cuDNN | 推薦用途 |
|------------|--------|------|-------|----------|
| 2.10.1     | 3.8-3.11 | 11.2 | 8.1   | 穩定生產 |
| 2.13.0     | 3.8-3.11 | 11.8 | 8.6   | 平衡選擇 |
| 2.15.0     | 3.9-3.12 | 12.2 | 8.9   | 最新特性 |

---

## 3. 通用執行指令集

### 🔍 環境檢查指令

#### GPU 狀態檢查
```bash
# 基本 GPU 資訊
nvidia-smi

# 詳細 GPU 資訊
nvidia-smi -q

# GPU 程序清單
nvidia-smi pmon

# CUDA 版本
nvcc --version
cat /usr/local/cuda/version.txt  # Linux
```

#### Python 環境檢查
```bash
# Python 版本與路徑
python --version
which python

# 套件版本檢查
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# GPU 可用性檢查
python -c "import tensorflow as tf; print('GPU可用:', len(tf.config.list_physical_devices('GPU')) > 0)"
python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
```

#### 套件列表與搜索
```bash
# 已安裝套件
pip list
conda list

# 搜索特定套件
pip list | grep tensorflow
conda list | grep torch

# 檢查過期套件
pip list --outdated
```

### 📊 即時監控指令

#### GPU 效能監控
```bash
# 每秒更新 GPU 狀態
nvidia-smi -l 1

# 自訂監控格式
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv -l 1

# 監控特定 GPU
nvidia-smi -i 0 -l 1

# 結合 watch 指令
watch -n 1 nvidia-smi
```

#### 系統資源監控
```bash
# CPU 與記憶體
htop
top

# 磁碟使用
df -h
du -sh *

# 網路監控
iftop
```

#### 監控腳本組合
```bash
# 建立 monitor.sh
#!/bin/bash
echo "開始系統監控..."
trap 'kill $(jobs -p)' EXIT

# 背景執行各種監控
htop &
nvidia-smi -l 1 &

# 等待使用者中斷
wait
```

### 🚀 通用訓練指令格式

#### 基本訓練指令
```bash
# 基本執行
python train.py

# 指定 GPU
CUDA_VISIBLE_DEVICES=0 python train.py

# 背景執行
nohup python train.py > training.log 2>&1 &

# 即時查看日誌
tail -f training.log
```

#### 參數化執行
```bash
# 標準參數格式
python train.py \
    --gpu 0 \
    --batch-size 64 \
    --epochs 100 \
    --learning-rate 0.001 \
    --model-dir ./models \
    --data-dir ./data

# 環境變數控制
export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=2
python train.py --config config.yaml
```

#### 分布式訓練 (多 GPU)
```bash
# TensorFlow 分布式
python -m tensorflow.distribute.strategy train.py

# PyTorch 分布式
python -m torch.distributed.launch --nproc_per_node=2 train.py
```

### 🛠️ 除錯指令

#### 記憶體問題除錯
```bash
# 清理 GPU 記憶體
nvidia-smi --gpu-reset

# 檢查程序佔用
nvidia-smi pmon -c 1

# 強制結束 GPU 程序
sudo kill -9 <PID>
```

#### Python 程序管理
```bash
# 查看 Python 程序
ps aux | grep python

# 結束特定程序
pkill -f "python train.py"

# 查看資源佔用
pidstat -r -p <PID>
```

#### 日誌與除錯
```bash
# 即時日誌查看
tail -f training.log

# 錯誤日誌篩選
grep -i error training.log
grep -i "out of memory" training.log

# 系統日誌檢查
dmesg | grep -i nvidia  # Linux
```

---

## 4. 新 Claude 協作指南

### 💻 硬體環境描述模板

複製以下模板，填入你的具體資訊給新的 Claude：

```markdown
## 我的深度學習環境

**硬體配置:**
- GPU: NVIDIA GeForce GTX 1660 Super
- VRAM: 6GB (可用約 4-4.5GB)
- 計算能力: 7.5 (Turing 架構)
- 主機記憶體: [填入你的 RAM 大小]
- CPU: [填入你的 CPU 型號]

**軟體環境:**
- 作業系統: Windows 11
- NVIDIA 驅動: 560.94
- CUDA 版本: 12.6
- Python: 3.10.x
- 主要框架: TensorFlow-GPU 2.10.1 (已確認正常運行)

**環境狀態:**
- ✅ GPU 驗證通過 (nvidia-smi 正常)
- ✅ TensorFlow GPU 支援確認
- ✅ 記憶體增長設定完成
- ✅ 基本計算測試通過

**效能基準:**
- 矩陣運算 (2000x2000): [填入測試時間]
- 可用批次大小: 32-64 (依模型而定)
- 典型訓練速度: [如已知]
```

### 🎯 問題描述模板

根據你要解決的問題類型，選擇對應模板：

#### 訓練問題模板
```markdown
## 訓練問題描述

**任務類型:** [例如: 圖像分類/文字分類/目標檢測/其他]

**資料規模:**
- 訓練樣本數: [數量]
- 驗證樣本數: [數量]
- 資料維度: [例如: 224x224x3 圖片 / 512 序列長度文字]

**目前狀況:**
- 模型架構: [簡述使用的模型]
- 遇到的問題: [具體描述]
- 當前效能: [準確率/損失值/其他指標]
- 期望目標: [想達到的效果]

**已嘗試的方法:**
- [列出已經試過的技術/參數]
- [效果如何]

**資源限制:**
- 訓練時間預算: [例如: 希望在 X 小時內完成]
- 記憶體限制: 4GB VRAM
- 其他限制: [如有]
```

#### 環境問題模板
```markdown
## 環境問題描述

**問題類型:** [例如: 安裝問題/相容性問題/效能問題]

**具體症狀:**
- 錯誤訊息: [完整貼上錯誤訊息]
- 出現時機: [什麼時候發生]
- 環境狀態: [相關環境資訊]

**復現步驟:**
1. [步驟一]
2. [步驟二]
3. [問題出現]

**已嘗試的解決方法:**
- [方法一及結果]
- [方法二及結果]
```

### 📁 檔案提供清單

#### 必要檔案
- **訓練資料**: 完整資料集或代表性樣本
- **當前程式碼**: 主要的訓練腳本
- **錯誤日誌**: 如果有問題，提供完整錯誤訊息

#### 可選檔案 (有助於更好理解)
- **資料集描述**: README 或資料說明文件
- **之前的實驗結果**: 訓練曲線、混淆矩陣等
- **模型檔案**: 如果有預訓練或部分訓練的模型
- **配置檔案**: config.yaml 或 settings.py

#### 檔案格式說明
```markdown
**資料集格式範例:**
- 圖像分類: ImageFolder 結構或 CSV 索引檔
- 文字分類: JSON/CSV 格式，包含 text 和 label 欄位
- 其他格式: [具體說明]

**檔案大小限制:**
- 單檔最大: [查看 Claude 當前限制]
- 總大小建議: [壓縮大檔案或提供樣本]
```

### 🔧 技術背景說明

#### 經驗水平描述
```markdown
**我的技術背景:**
- 深度學習經驗: [初學者/中級/進階]
- 熟悉的框架: [TensorFlow/PyTorch/其他]
- 偏好的學習方式: [代碼範例/理論解釋/步驟指導]

**希望的協助方式:**
- ✅ 提供完整可執行的程式碼
- ✅ 包含詳細註解說明
- ✅ 提供除錯和監控方法
- ✅ 說明關鍵參數的調整方向
```

#### 專案目標說明
```markdown
**專案目標:**
- 主要目標: [例如: 達到 X% 準確率]
- 次要目標: [例如: 訓練時間控制在 X 小時內]
- 部署需求: [是否需要考慮部署環境]

**時間安排:**
- 開發時限: [如有]
- 里程碑: [階段性目標]
```

---

## 5. 故障排除指南

### ❌ 常見錯誤與解決

#### CUDA Out of Memory 錯誤
**症狀**: 
```
RuntimeError: CUDA out of memory. Tried to allocate XXX MiB
```

**解決方案**:
```python
# 方法 1: 減少批次大小
batch_size = 16  # 從 64 降到 16

# 方法 2: 設定記憶體增長
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# 方法 3: 清理記憶體
import gc
tf.keras.backend.clear_session()
gc.collect()

# 方法 4: 使用混合精度訓練
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

#### GPU Not Detected 錯誤
**症狀**:
```python
tf.config.list_physical_devices('GPU')  # 返回空列表
```

**檢查步驟**:
```bash
# 1. 檢查驅動
nvidia-smi

# 2. 檢查 CUDA
nvcc --version

# 3. 檢查 TensorFlow 版本
python -c "import tensorflow as tf; print(tf.__version__)"

# 4. 重新安裝 GPU 版本
pip uninstall tensorflow
pip install tensorflow-gpu==2.10.1
```

#### 版本不相容錯誤
**症狀**:
```
Could not load dynamic library 'libcudnn.so.X'
```

**解決方案**:
```bash
# 檢查版本相容性
python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"

# 降級到穩定版本
pip install tensorflow-gpu==2.10.1

# 或使用 conda 自動處理依賴
conda install tensorflow-gpu -c conda-forge
```

### 🔧 VRAM 不足處理

#### 模型輕量化策略
```python
# 1. 使用更小的模型
model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3))

# 2. 減少層數或神經元數量
model = Sequential([
    Dense(128, activation='relu'),  # 從 512 減少到 128
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# 3. 使用 DepthwiseConv2D
tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same')
```

#### 批次大小優化
```python
# 動態批次大小調整
def find_optimal_batch_size(model, X_sample, y_sample, max_batch=128):
    batch_size = max_batch
    while batch_size >= 1:
        try:
            model.fit(X_sample[:batch_size], y_sample[:batch_size], epochs=1, verbose=0)
            print(f"最佳批次大小: {batch_size}")
            return batch_size
        except Exception as e:
            if "out of memory" in str(e).lower():
                batch_size //= 2
                print(f"嘗試批次大小: {batch_size}")
            else:
                raise e
    return 1
```

#### 混合精度訓練
```python
# TensorFlow 混合精度設定
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# 確保輸出層使用 float32
model.add(Dense(num_classes, activation='softmax', dtype='float32'))

# 損失縮放
optimizer = tf.keras.optimizers.Adam()
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
```

#### 梯度累積技術
```python
# 模擬更大批次大小的梯度累積
def train_with_gradient_accumulation(model, dataset, accumulation_steps=4):
    optimizer = tf.keras.optimizers.Adam()
    
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = loss_fn(y, predictions) / accumulation_steps
        
        gradients = tape.gradient(loss, model.trainable_variables)
        return loss, gradients
    
    accumulated_gradients = [tf.Variable(tf.zeros_like(var)) 
                           for var in model.trainable_variables]
    
    for step, (x, y) in enumerate(dataset):
        loss, gradients = train_step(x, y)
        
        # 累積梯度
        for i, grad in enumerate(gradients):
            accumulated_gradients[i].assign_add(grad)
        
        # 每 accumulation_steps 步更新一次
        if (step + 1) % accumulation_steps == 0:
            optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))
            # 重置累積梯度
            for acc_grad in accumulated_gradients:
                acc_grad.assign(tf.zeros_like(acc_grad))
```

### 🚨 緊急恢復方案

#### 環境完全重建
```bash
# 1. 備份重要檔案
cp -r important_files/ backup/

# 2. 移除舊環境
conda remove --name deeplearning --all -y

# 3. 重新創建環境
conda create --name deeplearning python=3.10 -y
conda activate deeplearning

# 4. 重新安裝套件
pip install tensorflow-gpu==2.10.1
pip install -r requirements.txt

# 5. 驗證環境
python gpu_test.py
```

#### 套件降級方案
```bash
# TensorFlow 版本降級
pip install tensorflow-gpu==2.8.0  # 更穩定的版本

# CUDA 工具包降級
# 下載並安裝 CUDA 11.2

# 整體降級到穩定組合
pip install tensorflow-gpu==2.8.0 numpy==1.21.6 pandas==1.3.5
```

#### 系統還原點 (Windows)
```powershell
# 創建還原點
Checkpoint-Computer -Description "Deep Learning Environment Backup"

# 查看還原點
Get-ComputerRestorePoint

# 恢復到特定還原點
Restore-Computer -RestorePoint X
```

---

## 6. 檔案管理建議

### 📂 專案結構模板

#### 標準深度學習專案結構
```
project_name/
├── data/                   # 資料檔案
│   ├── raw/               # 原始資料
│   ├── processed/         # 處理後資料
│   ├── external/          # 外部資料集
│   └── interim/           # 中間處理結果
├── models/                # 訓練好的模型
│   ├── checkpoints/       # 訓練檢查點
│   ├── final/            # 最終模型
│   └── experiments/       # 實驗模型
├── notebooks/             # Jupyter notebooks
│   ├── exploratory/       # 資料探索
│   ├── modeling/         # 模型開發
│   └── evaluation/       # 結果評估
├── src/                   # 原始碼
│   ├── data/             # 資料處理腳本
│   ├── features/         # 特徵工程
│   ├── models/           # 模型定義
│   ├── visualization/    # 視覺化工具
│   └── utils/            # 工具函數
├── outputs/               # 結果輸出
│   ├── figures/          # 圖表
│   ├── reports/          # 報告
│   └── predictions/      # 預測結果
├── logs/                  # 訓練日誌
│   ├── tensorboard/      # TensorBoard 日誌
│   ├── training/         # 訓練日誌
│   └── error/            # 錯誤日誌
├── configs/               # 配置檔案
│   ├── model_config.yaml # 模型配置
│   ├── data_config.yaml  # 資料配置
│   └── training_config.yaml # 訓練配置
├── tests/                 # 測試檔案
├── requirements.txt       # Python 依賴
├── environment.yml        # Conda 環境
├── README.md             # 專案說明
├── .gitignore            # Git 忽略檔案
└── setup.py              # 套件安裝腳本
```

#### 快速建立專案結構
```bash
# 建立專案目錄腳本 create_project.sh
#!/bin/bash
PROJECT_NAME=$1

mkdir -p $PROJECT_NAME/{data/{raw,processed,external,interim},models/{checkpoints,final,experiments},notebooks/{exploratory,modeling,evaluation},src/{data,features,models,visualization,utils},outputs/{figures,reports,predictions},logs/{tensorboard,training,error},configs,tests}

cd $PROJECT_NAME

# 建立基本檔案
touch README.md .gitignore requirements.txt environment.yml setup.py
touch configs/{model_config.yaml,data_config.yaml,training_config.yaml}

echo "專案 $PROJECT_NAME 結構建立完成！"
```

### 📝 命名規範

#### 檔案命名規範
```bash
# 模型檔案命名
model_architecture_dataset_version_metric.h5
# 範例: resnet50_cifar10_v1_acc92.h5

# 實驗記錄命名
experiment_YYYYMMDD_HHMM_description.json
# 範例: experiment_20250607_1430_lstm_emotion_classification.json

# 資料檔案命名
dataset_preprocessing_version.csv
# 範例: emotions_cleaned_v2.csv

# 日誌檔案命名
training_YYYYMMDD_HHMM.log
# 範例: training_20250607_1430.log
```

#### 版本管理規範
```bash
# 模型版本號規則
v[major].[minor].[patch]
# v1.0.0 - 第一個穩定版本
# v1.1.0 - 新增功能
# v1.1.1 - 錯誤修復

# Git 標籤規範
git tag -a v1.0.0 -m "First stable model release"
git tag -a exp-20250607 -m "Emotion classification experiment"
```

#### 配置檔案命名
```yaml
# config/model_config.yaml
model:
  name: "lstm_classifier"
  version: "v1.0.0"
  architecture: "bidirectional_lstm"
  
# config/data_config.yaml  
data:
  dataset: "emotion_classification"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

# config/training_config.yaml
training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.001
  optimizer: "adam"
```

### 💾 備份策略

#### 本地備份方案
```bash
# 每日自動備份腳本
#!/bin/bash
BACKUP_DIR="/backup/deeplearning"
PROJECT_DIR="/path/to/project"
DATE=$(date +%Y%m%d)

# 建立備份目錄
mkdir -p $BACKUP_DIR/$DATE

# 備份重要檔案
rsync -av --exclude='*.log' --exclude='__pycache__' \
  $PROJECT_DIR/ $BACKUP_DIR/$DATE/

# 壓縮備份
tar -czf $BACKUP_DIR/$DATE.tar.gz $BACKUP_DIR/$DATE/
rm -rf $BACKUP_DIR/$DATE/

# 保留最近 7 天的備份
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "備份完成: $BACKUP_DIR/$DATE.tar.gz"
```

#### 雲端備份方案
```bash
# Google Drive 同步 (rclone)
rclone sync /path/to/project googledrive:backup/deeplearning

# GitHub 自動推送
git add .
git commit -m "Auto backup $(date)"
git push origin main

# 模型檔案 (使用 DVC)
dvc add models/final/
git add models/final/.gitignore models/final/.dvc
git commit -m "Update model files"
dvc push
```

#### 重要檔案優先級
```bash
# 高優先級 (每日備份)
- 原始碼 (src/)
- 配置檔案 (configs/)
- 訓練腳本
- README.md

# 中優先級 (每週備份)  
- 處理後資料 (data/processed/)
- 最終模型 (models/final/)
- 重要結果 (outputs/reports/)

# 低優先級 (按需備份)
- 原始資料 (通常來源穩定)
- 實驗模型 (models/experiments/)
- 日誌檔案 (可重新生成)
```

### 🗂️ 大檔案處理

#### 大資料集管理
```bash
# 使用 Git LFS 管理大檔案
git lfs install
git lfs track "*.h5"
git lfs track "*.pkl"
git lfs track "data/raw/*"

# .gitattributes 檔案內容
*.h5 filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
data/raw/* filter=lfs diff=lfs merge=lfs -text
```

#### 檔案壓縮策略
```bash
# 資料檔案壓縮
# JSON 檔案
gzip large_dataset.json  # 通常可減少 70-80%

# 圖片資料集
tar -czf images.tar.gz images/  # 視內容而定

# NumPy 陣列
python -c "
import numpy as np
data = np.load('large_array.npy')
np.savez_compressed('large_array.npz', data=data)
"

# 模型檔案
# TensorFlow SavedModel 格式通常已最佳化
# 可考慮量化減少大小
```

#### 分割與合併
```bash
# 大檔案分割
split -b 100M large_file.h5 large_file.h5.part_

# 檔案合併
cat large_file.h5.part_* > large_file.h5

# Python 實現
def split_large_file(file_path, chunk_size=100*1024*1024):  # 100MB
    with open(file_path, 'rb') as f:
        chunk_num = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            with open(f"{file_path}.part_{chunk_num:03d}", 'wb') as chunk_file:
                chunk_file.write(chunk)
            chunk_num += 1
```

#### 版本控制最佳實踐
```bash
# .gitignore 範例
# 資料檔案
data/raw/
data/interim/
*.csv
*.h5
*.pkl
*.npy

# 模型檔案
models/checkpoints/
models/experiments/
*.ckpt
*.pb

# 日誌檔案
logs/
*.log
tensorboard/

# 系統檔案
__pycache__/
*.pyc
.DS_Store
Thumbs.db

# IDE 檔案
.vscode/
.idea/
*.swp

# 環境檔案
.env
```

#### 文件化最佳實踐
```markdown
# README.md 模板
# 專案名稱

## 簡介
[專案簡述]

## 環境需求
- Python 3.10+
- NVIDIA GPU (建議 GTX 1660 Super 或更高)
- CUDA 11.8+
- 記憶體: 16GB+ 建議

## 安裝步驟
```bash
conda create --name project_name python=3.10
conda activate project_name
pip install -r requirements.txt
```

## 使用方法
```bash
# 資料準備
python src/data/prepare_data.py

# 模型訓練
python src/models/train.py --config configs/model_config.yaml

# 模型評估
python src/models/evaluate.py --model models/final/best_model.h5
```

## 專案結構
[檔案結構說明]

## 結果
[實驗結果與效能指標]

## 授權
[授權資訊]
```

---

## 📚 附錄

### 🔗 有用的連結

#### 官方文件
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [TensorFlow GPU 支援指南](https://www.tensorflow.org/install/gpu)
- [PyTorch 安裝指南](https://pytorch.org/get-started/locally/)

#### 社群資源
- [TensorFlow GitHub](https://github.com/tensorflow/tensorflow)
- [PyTorch Forums](https://discuss.pytorch.org/)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

#### 效能優化
- [TensorFlow 效能指南](https://www.tensorflow.org/guide/gpu_performance_analysis)
- [GPU 記憶體優化](https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth)

### 📞 支援管道

#### 常見問題
1. **GPU 無法被檢測**：檢查驅動和 CUDA 版本相容性
2. **記憶體不足**：減少批次大小或使用混合精度
3. **訓練速度慢**：檢查 GPU 使用率和 I/O 瓶頸
4. **模型收斂困難**：調整學習率和正則化參數

#### 故障排除流程
1. 確認硬體狀態 (`nvidia-smi`)
2. 檢查軟體版本相容性
3. 查看錯誤日誌和堆疊追蹤
4. 搜尋相關的 GitHub Issues
5. 在相關論壇提問

---

## 📋 檢查清單

### ✅ 環境設定檢查清單

- [ ] NVIDIA 驅動已安裝 (版本 460+)
- [ ] CUDA Toolkit 已安裝並配置環境變數
- [ ] cuDNN 已下載並配置
- [ ] Python 虛擬環境已建立
- [ ] TensorFlow-GPU 或 PyTorch 已安裝
- [ ] GPU 檢測測試通過
- [ ] 記憶體增長設定已配置
- [ ] 基本運算測試通過

### ✅ 專案開發檢查清單

- [ ] 專案結構已建立
- [ ] requirements.txt 已建立
- [ ] .gitignore 已配置
- [ ] README.md 已撰寫
- [ ] 備份策略已實施
- [ ] 監控腳本已準備
- [ ] 測試資料已準備
- [ ] 基準測試已執行

### ✅ 協作準備檢查清單

- [ ] 環境資訊已整理
- [ ] 問題描述已準備
- [ ] 相關檔案已備妥
- [ ] 技術背景已說明
- [ ] 期望目標已明確
- [ ] 時間預算已評估

---

*本指南最後更新：2025年6月*  
*適用於 NVIDIA GTX 1660 Super 及類似規格的 GPU*
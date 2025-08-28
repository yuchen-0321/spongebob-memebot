#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch GPU 環境檢查腳本
檢查 PyTorch 是否正確安裝 GPU 支援
"""

import torch
import sys
import subprocess

print("=" * 60)
print("🔍 PyTorch GPU 環境診斷")
print("=" * 60)

# 1. 檢查 PyTorch 版本
print("\n📦 PyTorch 版本資訊：")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"編譯的 CUDA 版本: {torch.version.cuda}")
print(f"cuDNN 版本: {torch.backends.cudnn.version()}")

# 2. 檢查 CUDA 設備
if torch.cuda.is_available():
    print("\n✅ GPU 偵測成功！")
    print(f"GPU 數量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  名稱: {torch.cuda.get_device_name(i)}")
        print(f"  計算能力: {torch.cuda.get_device_capability(i)}")
        print(f"  總記憶體: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # 檢查當前記憶體使用
        if torch.cuda.is_available():
            print(f"  已使用記憶體: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  快取記憶體: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    
    # 3. 測試 GPU 運算
    print("\n🧮 GPU 運算測試...")
    try:
        # 建立測試張量
        device = torch.device('cuda:0')
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        
        # 執行矩陣乘法
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        z = torch.matmul(x, y)
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        
        print(f"✅ GPU 運算成功！")
        print(f"矩陣乘法 (1000x1000) 耗時: {elapsed_time:.2f} ms")
        
    except Exception as e:
        print(f"❌ GPU 運算失敗: {e}")
        
else:
    print("\n❌ 無法偵測到 GPU！")
    print("\n可能的原因：")
    print("1. PyTorch 安裝的是 CPU 版本")
    print("2. NVIDIA 驅動未正確安裝")
    print("3. CUDA 版本不相容")
    
    # 檢查系統 NVIDIA 驅動
    print("\n🔍 檢查 NVIDIA 驅動...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA 驅動已安裝")
            # 顯示簡短的 nvidia-smi 輸出
            lines = result.stdout.split('\n')
            for line in lines[:7]:  # 只顯示前幾行
                print(line)
        else:
            print("❌ 無法執行 nvidia-smi，請確認 NVIDIA 驅動是否安裝")
    except FileNotFoundError:
        print("❌ 找不到 nvidia-smi 指令，請確認 NVIDIA 驅動是否安裝")
    
    # 提供解決方案
    print("\n💡 解決方案：")
    print("\n1. 如果 nvidia-smi 正常但 PyTorch 無法偵測 GPU，請重新安裝 PyTorch GPU 版本：")
    print("   pip uninstall torch torchvision torchaudio")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n2. 如果使用 conda，請使用：")
    print("   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
    
    print("\n3. 檢查您的 CUDA 版本是否與 PyTorch 相容")

# 4. 顯示建議的安裝指令
print("\n" + "=" * 60)
print("📌 根據您的環境，建議使用以下指令安裝 PyTorch：")

if sys.platform.startswith('win'):
    print("\nWindows 系統：")
    print("# CUDA 11.8")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\n# CUDA 12.1")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
else:
    print("\nLinux/Mac 系統：")
    print("# CUDA 11.8")
    print("pip install torch torchvision torchaudio")

print("\n" + "=" * 60)

# 5. 環境變數檢查
print("\n🔍 環境變數檢查：")
import os
cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
if cuda_home:
    print(f"CUDA_HOME/CUDA_PATH: {cuda_home}")
else:
    print("⚠️ CUDA_HOME 或 CUDA_PATH 環境變數未設定")

path = os.environ.get('PATH', '')
if 'cuda' in path.lower():
    print("✅ PATH 包含 CUDA 路徑")
else:
    print("⚠️ PATH 可能未包含 CUDA 路徑")

print("\n" + "=" * 60)
print("診斷完成！")
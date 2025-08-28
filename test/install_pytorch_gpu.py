#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch GPU 版本安裝腳本
自動檢測環境並安裝適合的 PyTorch GPU 版本
"""

import subprocess
import sys
import os

print("=" * 60)
print("🚀 PyTorch GPU 版本安裝助手")
print("=" * 60)

def run_command(cmd, shell=True):
    """執行命令並返回結果"""
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

# 1. 檢查 nvidia-smi
print("\n1. 檢查 NVIDIA 驅動...")
success, stdout, stderr = run_command("nvidia-smi")
if not success:
    print("❌ 無法執行 nvidia-smi，請先安裝 NVIDIA 驅動！")
    print("請訪問: https://www.nvidia.com/drivers/")
    sys.exit(1)

# 解析 CUDA 版本
cuda_version = None
for line in stdout.split('\n'):
    if 'CUDA Version:' in line:
        # 提取 CUDA 版本號
        cuda_str = line.split('CUDA Version:')[1].strip()
        cuda_version = cuda_str.split()[0]
        print(f"✅ 偵測到 CUDA 版本: {cuda_version}")
        break

if not cuda_version:
    print("⚠️ 無法偵測 CUDA 版本，將使用預設設定")
    cuda_version = "11.8"

# 2. 確定要安裝的 PyTorch 版本
cuda_major = float(cuda_version.split('.')[0] + '.' + cuda_version.split('.')[1])

if cuda_major >= 12.0:
    pytorch_cuda = "cu121"
    print(f"將安裝支援 CUDA 12.1 的 PyTorch")
elif cuda_major >= 11.8:
    pytorch_cuda = "cu118"
    print(f"將安裝支援 CUDA 11.8 的 PyTorch")
else:
    print(f"⚠️ CUDA 版本 {cuda_version} 較舊，建議升級到 CUDA 11.8 或更新版本")
    pytorch_cuda = "cu118"

# 3. 檢查當前 PyTorch 安裝
print("\n2. 檢查當前 PyTorch 安裝...")
try:
    import torch
    current_version = torch.__version__
    cuda_available = torch.cuda.is_available()
    print(f"當前 PyTorch 版本: {current_version}")
    print(f"CUDA 支援: {'是' if cuda_available else '否'}")
    
    if cuda_available:
        response = input("\n已安裝支援 GPU 的 PyTorch，是否要重新安裝？(y/N): ")
        if response.lower() != 'y':
            print("取消安裝")
            sys.exit(0)
except ImportError:
    print("未安裝 PyTorch")

# 4. 卸載現有版本
print("\n3. 卸載現有 PyTorch...")
uninstall_cmd = f"{sys.executable} -m pip uninstall -y torch torchvision torchaudio"
print(f"執行: {uninstall_cmd}")
success, stdout, stderr = run_command(uninstall_cmd)
if success:
    print("✅ 卸載完成")
else:
    print("⚠️ 卸載可能失敗，繼續安裝...")

# 5. 安裝新版本
print(f"\n4. 安裝 PyTorch GPU 版本 (CUDA {pytorch_cuda})...")
install_cmd = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{pytorch_cuda}"
print(f"執行: {install_cmd}")
print("這可能需要幾分鐘時間，請耐心等待...\n")

success, stdout, stderr = run_command(install_cmd)
if success:
    print("✅ 安裝完成！")
else:
    print("❌ 安裝失敗")
    print(f"錯誤: {stderr}")
    sys.exit(1)

# 6. 驗證安裝
print("\n5. 驗證安裝...")
verify_script = """
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 名稱: {torch.cuda.get_device_name(0)}")
    print("✅ PyTorch GPU 版本安裝成功！")
    
    # 簡單測試
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = torch.matmul(x, y)
    print("✅ GPU 運算測試通過！")
else:
    print("❌ GPU 仍然無法使用，請檢查 CUDA 安裝")
"""

success, stdout, stderr = run_command(f'{sys.executable} -c "{verify_script}"')
print(stdout)
if stderr:
    print(f"錯誤: {stderr}")

print("\n" + "=" * 60)
print("安裝程序完成！")
print("\n如果仍有問題，請嘗試：")
print("1. 重新啟動電腦")
print("2. 確認 CUDA 路徑在系統 PATH 中")
print("3. 在新的終端機中測試")
print("\n您現在可以執行 check_gpu_pytorch.py 來驗證安裝")
import tensorflow as tf

print("="*50)
print("🔍 最終 GPU 測試")
print("="*50)

print(f"TensorFlow 版本: {tf.__version__}")
print(f"CUDA 支援: {tf.test.is_built_with_cuda()}")

# 檢查 GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"偵測到 GPU: {len(gpus)} 個")

if gpus:
    print("✅ GPU 資訊:")
    for gpu in gpus:
        print(f"   {gpu}")
        
    # 設定記憶體增長
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU 記憶體設定成功")
    except:
        print("⚠️ 記憶體設定警告（但不影響使用）")
    
    # 測試計算
    print("\n🧮 GPU 計算測試...")
    try:
        with tf.device('/GPU:0'):
            a = tf.random.normal([2000, 2000])
            b = tf.random.normal([2000, 2000])
            c = tf.matmul(a, b)
        print("🎉 你的 GTX 1660 Super 可以使用了！")
        print("🚀 準備好進行 LSTM 訓練！")
    except Exception as e:
        print(f"❌ GPU 計算失敗: {e}")
        
else:
    print("❌ 仍未偵測到 GPU")

print("="*50)
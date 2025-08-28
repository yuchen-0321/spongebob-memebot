#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
語意向量回歸模型訓練程式
目的：訓練一個模型，將輸入的中文文字句子轉換為對應的語意向量
輸入：中文斷詞後的 token list
輸出：768 維的語意向量
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import List, Tuple, Dict, Any
import logging
from tqdm import tqdm
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    """自訂資料集類別"""
    
    def __init__(self, data_path: str, max_length: int = 50):
        """
        初始化資料集
        
        Args:
            data_path: JSON 檔案路徑
            max_length: 最大序列長度
        """
        self.max_length = max_length
        self.data = []
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}  # 特殊標記
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        
        # 載入資料
        logger.info(f"載入資料: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 建立詞彙表
        self._build_vocabulary(raw_data)
        
        # 處理資料
        self._process_data(raw_data)
        
        logger.info(f"資料集大小: {len(self.data)}")
        logger.info(f"詞彙表大小: {len(self.word2idx)}")
    
    def _build_vocabulary(self, raw_data: List[Dict]):
        """建立詞彙表"""
        logger.info("建立詞彙表...")
        
        for item in raw_data:
            if 'tokens' in item and item['tokens']:
                for token in item['tokens']:
                    if token not in self.word2idx:
                        idx = len(self.word2idx)
                        self.word2idx[token] = idx
                        self.idx2word[idx] = token
    
    def _process_data(self, raw_data: List[Dict]):
        """處理資料並轉換為訓練格式"""
        logger.info("處理資料...")
        
        for item in raw_data:
            # 檢查必要欄位
            if 'tokens' not in item or 'embeddings' not in item:
                continue
            
            tokens = item['tokens']
            embeddings = item['embeddings']
            
            # 跳過空資料
            if not tokens or not embeddings:
                continue
            
            # 確保 embeddings 是 768 維
            if len(embeddings) != 768:
                logger.warning(f"跳過維度不正確的資料: {len(embeddings)} != 768")
                continue
            
            # 將 tokens 轉換為 indices
            token_ids = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
            
            # 截斷或補零到固定長度
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            else:
                token_ids = token_ids + [self.word2idx['<PAD>']] * (self.max_length - len(token_ids))
            
            self.data.append((token_ids, embeddings))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        token_ids, embeddings = self.data[idx]
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(embeddings, dtype=torch.float32)
        )


class DotProductAttention(nn.Module):
    """Dot-Product Attention 機制"""
    
    def __init__(self, hidden_size: int):
        super(DotProductAttention, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size * 2, hidden_size * 2)  # 因為是雙向 LSTM
    
    def forward(self, lstm_output: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            lstm_output: (batch_size, seq_len, hidden_size * 2)
            mask: (batch_size, seq_len) padding mask
        
        Returns:
            attended_output: (batch_size, hidden_size * 2)
            attention_weights: (batch_size, seq_len)
        """
        # 計算 attention scores
        scores = self.W(lstm_output)  # (batch_size, seq_len, hidden_size * 2)
        scores = torch.sum(scores * lstm_output, dim=-1)  # (batch_size, seq_len)
        
        # 應用 mask (如果有)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 計算 attention weights
        attention_weights = torch.softmax(scores, dim=-1)  # (batch_size, seq_len)
        
        # 加權平均
        attended_output = torch.bmm(
            attention_weights.unsqueeze(1), 
            lstm_output
        ).squeeze(1)  # (batch_size, hidden_size * 2)
        
        return attended_output, attention_weights


class SemanticRegressionModel(nn.Module):
    """語意向量回歸模型"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 hidden_dim: int = 128, output_dim: int = 768,
                 num_layers: int = 2, dropout: float = 0.3):
        """
        Args:
            vocab_size: 詞彙表大小
            embedding_dim: 詞嵌入維度
            hidden_dim: LSTM 隱藏層維度
            output_dim: 輸出向量維度 (768)
            num_layers: LSTM 層數
            dropout: Dropout 機率
        """
        super(SemanticRegressionModel, self).__init__()
        
        # Embedding 層
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 雙向 BiLSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention 機制
        self.attention = DotProductAttention(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 輸出層 (全連接層)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, output_dim)
        
        # 激活函數
        self.relu = nn.ReLU()
        
        # 初始化權重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型權重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor):
        """
        前向傳播
        
        Args:
            x: (batch_size, seq_len) token indices
            
        Returns:
            output: (batch_size, 768) 語意向量
        """
        # 建立 padding mask
        mask = (x != 0).float()  # (batch_size, seq_len)
        
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # BiLSTM
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Attention
        attended, _ = self.attention(lstm_out, mask)  # (batch_size, hidden_dim * 2)
        
        # 全連接層
        out = self.dropout(attended)
        out = self.fc1(out)  # (batch_size, hidden_dim * 4)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)  # (batch_size, 768)
        
        # 最終輸出層為 linear（不使用 softmax）
        return out


def train_epoch(model: nn.Module, dataloader: DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer,
                device: torch.device) -> Tuple[float, List[float], List[float]]:
    """訓練一個 epoch"""
    model.train()
    total_loss = 0.0
    batch_losses = []
    gradient_norms = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        # 將資料移到 GPU
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 前向傳播
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # 計算 loss
        loss = criterion(outputs, targets)
        
        # 反向傳播
        loss.backward()
        
        # 計算梯度範數
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        gradient_norms.append(grad_norm)
        
        optimizer.step()
        
        # 累計 loss
        batch_losses.append(loss.item())
        total_loss += loss.item()
        
        # 更新進度條
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader), batch_losses, gradient_norms


def validate(model: nn.Module, dataloader: DataLoader, 
             criterion: nn.Module, device: torch.device) -> Tuple[float, np.ndarray, np.ndarray, List[float]]:
    """驗證模型"""
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    cosine_similarities = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 儲存輸出和目標
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            # 計算餘弦相似度
            cos_sim = nn.functional.cosine_similarity(outputs, targets)
            cosine_similarities.extend(cos_sim.cpu().numpy().tolist())
            
            total_loss += loss.item()
    
    # 合併所有批次
    all_outputs = np.vstack(all_outputs)
    all_targets = np.vstack(all_targets)
    
    return total_loss / len(dataloader), all_outputs, all_targets, cosine_similarities


def cosine_embedding_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Cosine Embedding Loss 實作
    
    Args:
        output: 預測的語意向量 (batch_size, 768)
        target: 真實的語意向量 (batch_size, 768)
    
    Returns:
        loss: Cosine similarity loss
    """
    # 正規化向量
    output_norm = torch.nn.functional.normalize(output, p=2, dim=1)
    target_norm = torch.nn.functional.normalize(target, p=2, dim=1)
    
    # 計算 cosine similarity
    cosine_sim = torch.sum(output_norm * target_norm, dim=1)
    
    # 轉換為 loss (1 - cosine_similarity)
    loss = 1 - cosine_sim
    
    return loss.mean()


def create_visualizations(training_history: Dict, save_dir: str = "training_plots"):
    """生成並儲存所有訓練視覺化圖表"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 設定繪圖風格
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    
    # 1. Training vs Validation Loss Curves
    logger.info("生成 Loss Curves...")
    plt.figure(figsize=(12, 6))
    
    epochs = range(1, len(training_history['train_losses']) + 1)
    plt.plot(epochs, training_history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, training_history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    
    # 標記最佳驗證損失
    best_epoch = np.argmin(training_history['val_losses']) + 1
    best_val_loss = min(training_history['val_losses'])
    plt.scatter(best_epoch, best_val_loss, color='green', s=200, marker='*', 
                label=f'Best Val Loss (Epoch {best_epoch})', zorder=5)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Learning Rate Schedule
    if 'learning_rates' in training_history:
        logger.info("生成 Learning Rate Schedule...")
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, training_history['learning_rates'], 'g-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'learning_rate_schedule.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Cosine Similarity Distribution (Best Epoch)
    if 'best_epoch_cosine_similarities' in training_history:
        logger.info("生成 Cosine Similarity Distribution (Best Epoch)...")
        plt.figure(figsize=(10, 6))
        cos_sims = training_history['best_epoch_cosine_similarities']
        best_epoch = training_history.get('best_epoch', '?')
        
        plt.hist(cos_sims, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(np.mean(cos_sims), color='red', linestyle='--', 
                label=f'Mean: {np.mean(cos_sims):.3f}')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.title(f'Cosine Similarity Distribution (Best Epoch: {best_epoch})')  # 標題顯示最佳 epoch
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'cosine_similarity_distribution_best_epoch.png'), dpi=300, bbox_inches='tight')
        plt.close()
    elif 'final_cosine_similarities' in training_history:
        # 如果沒有最佳 epoch 資訊，使用最終 epoch 的資料（向後相容）
        logger.info("生成 Cosine Similarity Distribution (Final Epoch)...")
        plt.figure(figsize=(10, 6))
        cos_sims = training_history['final_cosine_similarities']
        
        plt.hist(cos_sims, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(np.mean(cos_sims), color='red', linestyle='--', 
                label=f'Mean: {np.mean(cos_sims):.3f}')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.title('Cosine Similarity Distribution (Final Epoch)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'cosine_similarity_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Gradient Norm Evolution
    if 'gradient_norms' in training_history:
        logger.info("生成 Gradient Norm Evolution...")
        plt.figure(figsize=(12, 6))
        
        # 每個 epoch 的平均梯度範數
        avg_grad_norms = []
        for epoch_norms in training_history['gradient_norms']:
            if epoch_norms:  # 確保不是空列表
                avg_grad_norms.append(np.mean(epoch_norms))
        
        if avg_grad_norms:
            plt.plot(range(1, len(avg_grad_norms) + 1), avg_grad_norms, 'purple', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Average Gradient Norm')
            plt.title('Gradient Norm Evolution')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'gradient_norm_evolution.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 5. Loss Distribution Heatmap
    if 'batch_losses_history' in training_history:
        logger.info("生成 Loss Distribution Heatmap...")
        
        # 準備數據矩陣
        max_batches = max(len(losses) for losses in training_history['batch_losses_history'])
        loss_matrix = np.full((len(training_history['batch_losses_history']), max_batches), np.nan)
        
        for i, epoch_losses in enumerate(training_history['batch_losses_history']):
            loss_matrix[i, :len(epoch_losses)] = epoch_losses
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(loss_matrix, cmap='YlOrRd', cbar_kws={'label': 'Loss'}, 
                   xticklabels=False, yticklabels=5)
        plt.xlabel('Batch')
        plt.ylabel('Epoch')
        plt.title('Batch Loss Distribution Across Epochs')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'loss_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Training Efficiency Metrics
    if 'epoch_times' in training_history:
        logger.info("生成 Training Efficiency Metrics...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # 訓練時間
        ax1.plot(epochs, training_history['epoch_times'], 'b-', linewidth=2, marker='o')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Training Time per Epoch')
        ax1.grid(True, alpha=0.3)
        
        # GPU 記憶體使用（如果有記錄）
        if 'gpu_memory_usage' in training_history:
            ax2.plot(epochs, training_history['gpu_memory_usage'], 'r-', linewidth=2, marker='s')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('GPU Memory (MB)')
            ax2.set_title('GPU Memory Usage')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_efficiency.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 7. Vector Space Visualization (t-SNE)
    if 'final_outputs' in training_history and 'final_targets' in training_history:
        logger.info("生成 Vector Space Visualization...")
        
        outputs = training_history['final_outputs']
        targets = training_history['final_targets']
        
        # 限制樣本數量以加速 t-SNE
        n_samples = min(1000, len(outputs))
        sample_indices = np.random.choice(len(outputs), n_samples, replace=False)
        
        # 合併預測和真實向量
        all_vectors = np.vstack([outputs[sample_indices], targets[sample_indices]])
        labels = ['Predicted'] * n_samples + ['Ground Truth'] * n_samples
        
        # t-SNE 降維
        logger.info("執行 t-SNE 降維...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        vectors_2d = tsne.fit_transform(all_vectors)
        
        plt.figure(figsize=(12, 8))
        colors = ['blue', 'red']
        for i, label in enumerate(['Predicted', 'Ground Truth']):
            mask = np.array(labels) == label
            plt.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1], 
                       c=colors[i], label=label, alpha=0.6, s=30)
        
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('t-SNE Visualization of Semantic Vectors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'tsne_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 8. Prediction Error Analysis
    if 'final_outputs' in training_history and 'final_targets' in training_history:
        logger.info("生成 Prediction Error Analysis...")
        
        outputs = training_history['final_outputs']
        targets = training_history['final_targets']
        
        # 計算每個維度的平均絕對誤差
        dimension_errors = np.mean(np.abs(outputs - targets), axis=0)
        
        plt.figure(figsize=(14, 6))
        plt.bar(range(len(dimension_errors)), dimension_errors, width=1.0)
        plt.xlabel('Vector Dimension')
        plt.ylabel('Mean Absolute Error')
        plt.title('Prediction Error by Vector Dimension')
        plt.grid(True, alpha=0.3, axis='y')
        
        # 標記誤差最大的前10個維度
        top_10_dims = np.argsort(dimension_errors)[-10:]
        for dim in top_10_dims:
            plt.text(dim, dimension_errors[dim], str(dim), 
                    ha='center', va='bottom', fontsize=8, color='red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'dimension_error_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 9. Summary Report
    logger.info("生成 Summary Report...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss curves (簡化版)
    ax1.plot(epochs, training_history['train_losses'], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, training_history['val_losses'], 'r-', label='Val', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Final metrics
    final_train_loss = training_history['train_losses'][-1]
    final_val_loss = training_history['val_losses'][-1]
    best_val_loss = min(training_history['val_losses'])
    
    metrics_text = f"""Final Training Loss: {final_train_loss:.6f}
Final Validation Loss: {final_val_loss:.6f}
Best Validation Loss: {best_val_loss:.6f}
Best Epoch: {np.argmin(training_history['val_losses']) + 1}
Total Epochs: {len(training_history['train_losses'])}"""
    
    if 'final_cosine_similarities' in training_history:
        mean_cos_sim = np.mean(training_history['final_cosine_similarities'])
        metrics_text += f"\nMean Cosine Similarity: {mean_cos_sim:.4f}"
    
    ax2.text(0.1, 0.5, metrics_text, transform=ax2.transAxes, 
            fontsize=14, verticalalignment='center', fontfamily='monospace')
    ax2.axis('off')
    ax2.set_title('Training Summary')
    
    # Cosine similarity histogram (if available)
    if 'best_epoch_cosine_similarities' in training_history:
        cos_sims = training_history['best_epoch_cosine_similarities']
        best_epoch = training_history.get('best_epoch', '?')
        ax3.hist(cos_sims, bins=30, edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Cosine Similarity')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Cosine Similarity Distribution (Best Epoch: {best_epoch})')
        ax3.grid(True, alpha=0.3)
    elif 'final_cosine_similarities' in training_history:
        cos_sims = training_history['final_cosine_similarities']
        ax3.hist(cos_sims, bins=30, edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Cosine Similarity')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Cosine Similarity Distribution (Final Epoch)')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.axis('off')
    
    plt.suptitle('Training Summary Report', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary_report.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"所有圖表已儲存至: {save_dir}/")
    
    # 儲存訓練歷史為 JSON
    history_to_save = {
        'train_losses': training_history['train_losses'],
        'val_losses': training_history['val_losses'],
        'best_epoch': int(np.argmin(training_history['val_losses']) + 1),
        'best_val_loss': float(min(training_history['val_losses'])),
    }
    
    if 'learning_rates' in training_history:
        history_to_save['learning_rates'] = training_history['learning_rates']
    
    if 'final_cosine_similarities' in training_history:
        history_to_save['mean_cosine_similarity'] = float(np.mean(training_history['final_cosine_similarities']))
    
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history_to_save, f, indent=2)
    
    logger.info("訓練歷史已儲存至 training_history.json")


# ============================================
# 新增推論相關函式
# ============================================

def load_model(
    model_path: str,
    data_json: str,
    device: str = "cpu"
) -> Tuple[SemanticRegressionModel, Dict[str, int]]:
    """
    載入訓練好的模型和詞彙表
    
    Args:
        model_path: 模型權重檔案路徑 (如 best_model.pt)
        data_json: 用於重建詞彙表的 JSON 檔案路徑 (如 processed_data.json)
        device: 運算設備 ("cpu" 或 "cuda")
    
    Returns:
        model: 載入權重的 SemanticRegressionModel
        word2idx: 詞彙到索引的映射字典
    """
    logger.info(f"載入模型和詞彙表...")
    
    # 讀取 JSON 資料以重建詞彙表
    logger.info(f"從 {data_json} 重建詞彙表...")
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    
    with open(data_json, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 重建詞彙表
    for item in raw_data:
        if 'tokens' in item and item['tokens']:
            for token in item['tokens']:
                if token not in word2idx:
                    word2idx[token] = len(word2idx)
    
    logger.info(f"詞彙表大小: {len(word2idx)}")
    
    # 初始化模型
    model = SemanticRegressionModel(
        vocab_size=len(word2idx),
        embedding_dim=128,
        hidden_dim=128,
        output_dim=768,
        num_layers=2,
        dropout=0.3
    )
    
    # 載入模型權重
    logger.info(f"載入模型權重: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # 設定為評估模式
    model.eval()
    
    # 移動到指定設備
    model = model.to(device)
    
    logger.info("模型載入完成")
    return model, word2idx


def predict_embedding(
    model: SemanticRegressionModel,
    word2idx: Dict[str, int],
    tokens: List[str],
    max_length: int = 50,
    device: str = "cpu"
) -> np.ndarray:
    """
    使用模型預測語意向量
    
    Args:
        model: 訓練好的 SemanticRegressionModel
        word2idx: 詞彙到索引的映射字典
        tokens: 斷詞後的 token 列表
        max_length: 最大序列長度
        device: 運算設備 ("cpu" 或 "cuda")
    
    Returns:
        embedding: 預測的語意向量，shape=(768,)
    """
    # 將 tokens 轉換為 indices
    token_ids = [word2idx.get(token, word2idx['<UNK>']) for token in tokens]
    
    # 截斷或補零到固定長度
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    else:
        token_ids = token_ids + [word2idx['<PAD>']] * (max_length - len(token_ids))
    
    # 轉換為 tensor
    input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
    
    # 使用模型預測
    with torch.no_grad():
        output = model(input_tensor)
    
    # 轉換為 numpy array
    embedding = output.squeeze(0).detach().cpu().numpy()
    
    return embedding


# ============================================
# 主程式（訓練流程）
# ============================================

if __name__ == "__main__":
    # 強制檢查 GPU 可用性
    if not torch.cuda.is_available():
        logger.error("❌ 無法偵測到 GPU！請檢查以下項目：")
        logger.error("1. NVIDIA 驅動是否正確安裝 (執行 nvidia-smi)")
        logger.error("2. PyTorch 是否為 GPU 版本")
        logger.error("3. CUDA 是否正確安裝")
        print("\n建議執行以下指令檢查：")
        print("1. nvidia-smi")
        print("2. python -c \"import torch; print(torch.cuda.is_available())\"")
        print("3. python -c \"import torch; print(torch.version.cuda)\"")
        
        # 提供安裝建議
        print("\n如果 PyTorch 不是 GPU 版本，請重新安裝：")
        print("pip uninstall torch torchvision torchaudio")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
        sys.exit(1)

    # 設定 GPU 設備
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    # 顯示 GPU 資訊
    logger.info(f"✅ 使用設備: {device}")
    logger.info(f"GPU 型號: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    logger.info(f"PyTorch CUDA 版本: {torch.version.cuda}")
    logger.info(f"cuDNN 版本: {torch.backends.cudnn.version()}")

    # 設定 cuDNN 優化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    """主函數：執行訓練流程"""
    # 設定參數
    DATA_PATH = "processed_data.json"
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 128
    OUTPUT_DIM = 768
    MAX_LENGTH = 50
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    # 檢查資料檔案是否存在
    if not os.path.exists(DATA_PATH):
        logger.error(f"找不到資料檔案: {DATA_PATH}")
        sys.exit(1)
    
    # 建立資料集
    logger.info("建立資料集...")
    dataset = CustomDataset(DATA_PATH, max_length=MAX_LENGTH)
    
    # 分割訓練集和驗證集 (9:1)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # 建立 DataLoader - 使用 GPU 優化設定
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,  # Windows 上設為 0 避免多進程問題
        pin_memory=True,  # 加速 GPU 資料傳輸
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0,  # Windows 上設為 0 避免多進程問題
        pin_memory=True,  # 加速 GPU 資料傳輸
        persistent_workers=False
    )
    
    logger.info(f"訓練集大小: {len(train_dataset)}")
    logger.info(f"驗證集大小: {len(val_dataset)}")
    
    # 建立模型
    logger.info("建立模型...")
    model = SemanticRegressionModel(
        vocab_size=len(dataset.word2idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    # 顯示模型資訊
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"總參數量: {total_params:,}")
    logger.info(f"可訓練參數量: {trainable_params:,}")
    
    # 定義 loss function 和 optimizer
    criterion = nn.MSELoss().to(device)  # 使用 MSE Loss
    # 如果要使用 Cosine Embedding Loss，可以替換為：
    # criterion = cosine_embedding_loss
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 學習率調度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 訓練模型
    logger.info("開始訓練...")
    best_val_loss = float('inf')
    
    # 訓練歷史記錄
    training_history = {
        'train_losses': [],
        'val_losses': [],
        'learning_rates': [],
        'batch_losses_history': [],
        'gradient_norms': [],
        'epoch_times': [],
        'gpu_memory_usage': []
    }
    
    import time
    
    for epoch in range(NUM_EPOCHS):
        logger.info(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        
        # 記錄 epoch 開始時間
        epoch_start_time = time.time()
        
        # 記錄當前學習率
        current_lr = optimizer.param_groups[0]['lr']
        training_history['learning_rates'].append(current_lr)
        
        # 訓練
        train_loss, batch_losses, grad_norms = train_epoch(model, train_loader, criterion, optimizer, device)
        training_history['train_losses'].append(train_loss)
        training_history['batch_losses_history'].append(batch_losses)
        training_history['gradient_norms'].append(grad_norms)
        
        # 驗證
        val_loss, outputs, targets, cosine_sims = validate(model, val_loader, criterion, device)
        training_history['val_losses'].append(val_loss)
        
        # 記錄 GPU 記憶體使用
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated(device) / 1024**2  # MB
            training_history['gpu_memory_usage'].append(gpu_memory)
        
        # 記錄 epoch 時間
        epoch_time = time.time() - epoch_start_time
        training_history['epoch_times'].append(epoch_time)
        
        # 調整學習率
        scheduler.step(val_loss)
        
        # 顯示結果
        logger.info(f"訓練 Loss: {train_loss:.4f}")
        logger.info(f"驗證 Loss: {val_loss:.4f}")
        logger.info(f"平均餘弦相似度: {np.mean(cosine_sims):.4f}")
        logger.info(f"Epoch 時間: {epoch_time:.2f} 秒")
        
        # 儲存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"儲存最佳模型 (驗證 Loss: {val_loss:.4f})")
            torch.save(model.state_dict(), 'best_model.pt')
            
            # 儲存最佳 epoch 的詳細資訊
            training_history['final_outputs'] = outputs
            training_history['final_targets'] = targets
            training_history['final_cosine_similarities'] = cosine_sims
            
            # 新增：記錄最佳 epoch 的資訊
            training_history['best_epoch'] = epoch + 1
            training_history['best_epoch_cosine_similarities'] = cosine_sims.copy()  # 儲存最佳 epoch 的相似度
    
    logger.info(f"\n訓練完成！最佳驗證 Loss: {best_val_loss:.4f}")
    logger.info("模型已儲存至 best_model.pt")
    
    # 儲存詞彙表
    vocab_info = {
        'word2idx': dataset.word2idx,
        'idx2word': dataset.idx2word,
        'vocab_size': len(dataset.word2idx),
        'max_length': MAX_LENGTH
    }
    
    with open('vocab_info.json', 'w', encoding='utf-8') as f:
        json.dump(vocab_info, f, ensure_ascii=False, indent=2)
    
    logger.info("詞彙表資訊已儲存至 vocab_info.json")
    
    # 生成所有視覺化圖表
    logger.info("\n生成訓練視覺化圖表...")
    create_visualizations(training_history)
    
    # 清理 GPU 記憶體
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================
# 使用範例
# ============================================

# Example:
# from semantic_regression_model import load_model, predict_embedding
# model, word2idx = load_model("best_model.pt", "processed_data.json")
# emb = predict_embedding(model, word2idx, ["一些", "烤牛肉", "幾", "塊", "炸雞", "披", "薩"])
# print(emb.shape)  # (768,)
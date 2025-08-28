#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
海綿寶寶梗圖推薦機器人 GUI
採用比奇堡風格設計，響應式全螢幕支援
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import faiss
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QScrollArea, QGridLayout, QLabel,
    QDialog, QMessageBox, QProgressBar, QStatusBar, QSizePolicy,
    QGraphicsDropShadowEffect, QFrame, QToolButton
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QPropertyAnimation, QRect, QEasingCurve, pyqtSlot
from PyQt5.QtGui import (
    QPixmap, QFont, QPalette, QColor, QFontDatabase, QIcon, QPainter, 
    QBrush, QPainterPath, QLinearGradient, QImage, QClipboard
)

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpongeBobButton(QPushButton):
    """海綿寶寶風格按鈕"""
    def __init__(self, text, primary=True):
        super().__init__(text)
        self.primary = primary
        self.base_font_size = 20  # 加大基礎字體
        self.setup_style()
        
    def setup_style(self):
        if self.primary:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #FF8FA3, stop:1 #FFB6C1);
                    color: white;
                    border: none;
                    padding: 16px 40px;
                    border-radius: 30px;
                    font-size: {self.base_font_size}px;
                    font-weight: bold;
                    min-width: 160px;
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #FF7A8F, stop:1 #FFA0B0);
                }}
                QPushButton:pressed {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #FF6A7F, stop:1 #FF90A0);
                }}
                QPushButton:disabled {{
                    background: #4A5568;
                    color: #718096;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: #4ECDC4;
                    color: white;
                    border: none;
                    padding: 14px 32px;
                    border-radius: 25px;
                    font-size: {self.base_font_size - 2}px;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background-color: #3EBDB4;
                }}
            """)


class SpongeBobLineEdit(QLineEdit):
    """海綿寶寶風格輸入框"""
    def __init__(self, placeholder=""):
        super().__init__()
        self.setPlaceholderText(placeholder)
        self.base_font_size = 20  # 加大基礎字體
        self.setStyleSheet(f"""
            QLineEdit {{
                background-color: #162B45;
                border: 3px solid #1E3A5F;
                border-radius: 30px;
                padding: 16px 32px;
                font-size: {self.base_font_size}px;
                color: #F5F7FA;
                selection-background-color: #F4D03F;
                selection-color: #0A1628;
            }}
            QLineEdit:focus {{
                border-color: #F4D03F;
                background-color: #1E3A5F;
            }}
            QLineEdit::placeholder {{
                color: #7A8CA0;
            }}
        """)


class ImageCard(QFrame):
    """海綿寶寶風格圖片卡片"""
    clicked = pyqtSignal(str, str, float, int)
    
    def __init__(self, image_path: str, image_name: str, score: float, rank: int, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.image_name = image_name
        self.score = score
        self.rank = rank
        self.base_size = 320
        self.setup_ui()
        
    def setup_ui(self):
        self.setFixedSize(self.base_size, self.base_size + 80)
        self.setCursor(Qt.PointingHandCursor)
        
        # 設定卡片樣式
        self.setStyleSheet("""
            QFrame {
                background-color: #162B45;
                border-radius: 20px;
                border: 3px solid #1E3A5F;
            }
            QFrame:hover {
                background-color: #1E3A5F;
                border-color: #F4D03F;
            }
        """)
        
        # 陰影效果
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(25)
        shadow.setXOffset(0)
        shadow.setYOffset(5)
        shadow.setColor(QColor(0, 0, 0, 60))
        self.setGraphicsEffect(shadow)
        
        # 布局
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        self.setLayout(layout)
        
        # 排名標籤
        rank_label = QLabel(f"TOP {self.rank}")
        rank_label.setStyleSheet("""
            QLabel {
                background-color: #F4D03F;
                color: #0A1628;
                padding: 6px 16px;
                border-radius: 15px;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        rank_label.setAlignment(Qt.AlignCenter)
        rank_label.setMaximumHeight(36)
        layout.addWidget(rank_label)
        
        # 圖片容器
        image_container = QLabel()
        image_container.setFixedSize(self.base_size - 30, self.base_size - 120)
        image_container.setStyleSheet("""
            QLabel {
                background-color: #0A1628;
                border-radius: 15px;
                padding: 8px;
            }
        """)
        image_container.setAlignment(Qt.AlignCenter)
        
        # 載入圖片
        pixmap = QPixmap(self.image_path)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(
                self.base_size - 50, self.base_size - 140,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            image_container.setPixmap(scaled_pixmap)
        else:
            image_container.setText("無法載入圖片")
            image_container.setStyleSheet("""
                QLabel {
                    background-color: #0A1628;
                    border-radius: 15px;
                    color: #7A8CA0;
                    font-size: 18px;
                }
            """)
        
        layout.addWidget(image_container)
        
        # 檔名標籤
        name_label = QLabel(self.image_name[:25] + "..." if len(self.image_name) > 25 else self.image_name)
        # 先拆解出不含副檔名的檔名
        base_name, _ = os.path.splitext(self.image_name)
        # 再做長度截斷
        display_name = (base_name[:25] + "...") if len(base_name) > 25 else base_name
        name_label.setText(display_name)
        name_label.setStyleSheet("""
            QLabel {
                color: #F5F7FA;
                font-size: 22px;
                font-weight: 500;
                border: none;               /* 取消邊框 */
                background-color: transparent;  /* 透明底色 */
            }
        """)
        name_label.setWordWrap(True)
        name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(name_label)
        
        # 相似度標籤
        score_label = QLabel(f"相似度: {self.score:.3f}")
        score_label.setStyleSheet("""
            QLabel {
                color: #FF8FA3;
                font-size: 16px;
                font-weight: 600;
                border: none;               /* 取消邊框 */
                background-color: transparent;  /* 透明底色 */
            }
        """)
        score_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(score_label)
        
        layout.addStretch()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.image_path, self.image_name, self.score, self.rank)


class SpongeBobImageDialog(QDialog):
    """海綿寶寶風格圖片詳情對話框"""
    
    def __init__(self, image_path: str, image_name: str, score: float, rank: int, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.setWindowTitle(f"🧽 圖片詳情 - TOP {rank}")
        self.setModal(True)
        self.resize(1000, 800)
        
        # 設定對話框樣式
        self.setStyleSheet("""
            QDialog {
                background-color: #0A1628;
            }
        """)
        
        # 主布局
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(25)
        self.setLayout(layout)
        
        # 標題區域
        title_layout = QHBoxLayout()
        
        title_label = QLabel(image_name)
        title_label.setStyleSheet("""
            QLabel {
                color: #F4D03F;
                font-size: 32px;
                font-weight: bold;
            }
        """)
        title_layout.addWidget(title_label)
        
        # 排名和分數標籤
        info_layout = QHBoxLayout()
        
        rank_badge = QLabel(f"TOP {rank}")
        rank_badge.setStyleSheet("""
            QLabel {
                background-color: #F4D03F;
                color: #0A1628;
                padding: 8px 20px;
                border-radius: 20px;
                font-size: 20px;
                font-weight: bold;
            }
        """)
        rank_badge.setFixedHeight(40)
        info_layout.addWidget(rank_badge)
        
        score_badge = QLabel(f"相似度 {score:.3f}")
        score_badge.setStyleSheet("""
            QLabel {
                background-color: #FF8FA3;
                color: white;
                padding: 8px 20px;
                border-radius: 20px;
                font-size: 20px;
                font-weight: 600;
            }
        """)
        score_badge.setFixedHeight(40)
        info_layout.addWidget(score_badge)
        
        info_layout.addStretch()
        title_layout.addLayout(info_layout)
        
        layout.addLayout(title_layout)
        
        # 圖片顯示
        image_container = QLabel()
        image_container.setAlignment(Qt.AlignCenter)
        image_container.setStyleSheet("""
            QLabel {
                background-color: #162B45;
                border-radius: 20px;
                padding: 30px;
                border: 3px solid #1E3A5F;
            }
        """)
        
        pixmap = QPixmap(self.image_path)
        if not pixmap.isNull():
            # 保存原始圖片供複製使用
            self.original_pixmap = pixmap
            
            # 計算縮放尺寸
            max_width = 900
            max_height = 500
            scaled_pixmap = pixmap.scaled(
                max_width, max_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            image_container.setPixmap(scaled_pixmap)
        else:
            image_container.setText("無法載入圖片")
            image_container.setStyleSheet("""
                QLabel {
                    background-color: #162B45;
                    border-radius: 20px;
                    color: #7A8CA0;
                    font-size: 24px;
                }
            """)
            self.original_pixmap = None
        
        layout.addWidget(image_container)
        
        # 路徑資訊
        path_label = QLabel(f"📁 檔案路徑: {image_path}")
        path_label.setWordWrap(True)
        path_label.setStyleSheet("""
            QLabel {
                color: #B8C5D6;
                font-size: 18px;
                padding: 15px;
                background-color: #162B45;
                border-radius: 12px;
            }
        """)
        layout.addWidget(path_label)
        
        # 按鈕區域
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        # 複製圖片按鈕
        copy_btn = SpongeBobButton("📋 複製圖片", primary=False)
        copy_btn.clicked.connect(self.copy_image)
        btn_layout.addWidget(copy_btn)
        
        # 關閉按鈕
        close_btn = SpongeBobButton("關閉", primary=True)
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
    
    def copy_image(self):
        """複製圖片到剪貼簿"""
        if self.original_pixmap:
            clipboard = QApplication.clipboard()
            clipboard.setPixmap(self.original_pixmap)
            
            # 顯示成功訊息
            msg = QMessageBox(self)
            msg.setWindowTitle("成功！")
            msg.setText("圖片已複製到剪貼簿！\n準備好做梗圖了嗎？😄")
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: #162B45;
                    color: #FFFFFF;
                }
                QMessageBox QLabel {
                    color: #FFFFFF;
                }
                QMessageBox QPushButton {
                    background-color: #4ECDC4;
                    color: white;
                    padding: 8px 20px;
                    border-radius: 15px;
                    font-size: 16px;
                }
            """)
            msg.exec_()


class SearchWorker(QThread):
    """背景執行搜尋的工作執行緒 - GPU優化版本"""
    progress = pyqtSignal(str)
    result_ready = pyqtSignal(list)
    error = pyqtSignal(str)
    
    def __init__(self, text: str, index, metadata):
        super().__init__()
        self.text = text
        self.index = index
        self.metadata = metadata
        self.embedder = None
    
    def run(self):
        try:
            # 初始化文字嵌入器 - 明確指定使用 GPU
            self.progress.emit("正在連接到比奇堡伺服器...")
            from sentence_transformers import SentenceTransformer
            import torch
            
            # 檢查 GPU 可用性
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.progress.emit(f"使用設備: {device}")
            logger.info(f"SentenceTransformer 使用設備: {device}")
            
            # 載入模型並指定設備
            self.embedder = SentenceTransformer('jinaai/jina-embeddings-v2-base-zh', device=device)
            
            # 產生向量
            self.progress.emit("派大星正在分析你的文字...")
            query_vec = self.embedder.encode(self.text, convert_to_numpy=True)
            
            # 確保是 768 維
            if len(query_vec) != 768:
                if len(query_vec) < 768:
                    padded = np.zeros(768)
                    padded[:len(query_vec)] = query_vec
                    query_vec = padded
                else:
                    query_vec = query_vec[:768]
            
            query_vec = query_vec.reshape(1, -1).astype(np.float32)
            
            # 向量檢索 - 只取前9個結果
            self.progress.emit("海綿寶寶正在尋找最適合的梗圖...")
            k = min(9, self.index.ntotal)
            distances, indices = self.index.search(query_vec, k)
            
            # 整理結果
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.metadata["entries"]):
                    entry = self.metadata["entries"][idx]
                    similarity = 1 / (1 + dist)
                    results.append({
                        "path": entry.get("image_path", ""),
                        "name": entry.get("image_name", ""),
                        "score": float(similarity),
                        "distance": float(dist),
                        "rank": i + 1
                    })
            
            self.result_ready.emit(results)
            
        except Exception as e:
            logger.error(f"搜尋執行緒錯誤: {e}")
            self.error.emit(str(e))

class SpongeBobMainWindow(QMainWindow):
    """海綿寶寶梗圖推薦機器人主視窗"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧽 海綿寶寶梗圖推薦機器人")
        
        # 設定初始大小和最小尺寸
        self.resize(1400, 900)
        self.setMinimumSize(1000, 700)
        
        # 設定背景樣式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0A1628;
            }
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #162B45;
                width: 16px;
                border-radius: 8px;
            }
            QScrollBar::handle:vertical {
                background-color: #1E3A5F;
                border-radius: 8px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #2A4A6F;
            }
            QStatusBar {
                background-color: #162B45;
                color: #F5F7FA;
                font-size: 18px;
                padding: 10px;
                border-top: 2px solid #1E3A5F;
            }
        """)
        
        # 初始化變數
        self.index = None
        self.metadata = None
        self.search_worker = None
        
        # 設定 UI
        self.setup_ui()
        
        # 載入資源
        self.load_resources()
    
    def setup_ui(self):
        """設定使用者介面"""
        # 中央小部件
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: transparent;")
        self.setCentralWidget(central_widget)
        
        # 主垂直布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(50, 40, 50, 30)
        main_layout.setSpacing(35)
        central_widget.setLayout(main_layout)
        
        # 標題區域
        title_container = QWidget()
        title_layout = QVBoxLayout()
        title_container.setLayout(title_layout)
        
        # 主標題
        title_label = QLabel("🧽 海綿寶寶梗圖推薦機器人")
        title_label.setStyleSheet("""
            QLabel {
                color: #F4D03F;
                font-size: 48px;
                font-weight: bold;
                margin-bottom: 15px;
            }
        """)
        title_label.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title_label)
        
        # 副標題
        subtitle_label = QLabel("歡迎來到比奇堡！輸入文字找出最適合的梗圖")
        subtitle_label.setStyleSheet("""
            QLabel {
                color: #B8C5D6;
                font-size: 24px;
                margin-bottom: 30px;
            }
        """)
        subtitle_label.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(subtitle_label)
        
        main_layout.addWidget(title_container)
        
        # 搜尋區域容器
        search_container = QWidget()
        search_container.setMaximumWidth(900)
        search_layout = QHBoxLayout()
        search_layout.setContentsMargins(0, 0, 0, 0)
        search_layout.setSpacing(20)
        search_container.setLayout(search_layout)
        
        # 搜尋輸入框
        self.search_input = SpongeBobLineEdit("輸入你想要的梗圖關鍵字...")
        self.search_input.returnPressed.connect(self.on_search_clicked)
        search_layout.addWidget(self.search_input)
        
        # 搜尋按鈕
        self.search_btn = SpongeBobButton("搜尋梗圖 🔍")
        self.search_btn.clicked.connect(self.on_search_clicked)
        search_layout.addWidget(self.search_btn)
        
        # 將搜尋容器置中
        search_wrapper = QHBoxLayout()
        search_wrapper.addStretch()
        search_wrapper.addWidget(search_container)
        search_wrapper.addStretch()
        main_layout.addLayout(search_wrapper)
        
        # 進度條
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #162B45;
                border-radius: 8px;
                height: 12px;
                text-align: center;
                border: 2px solid #1E3A5F;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #F4D03F, stop:1 #FFD700);
                border-radius: 6px;
            }
        """)
        main_layout.addWidget(self.progress_bar)
        
        # 結果顯示區域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        main_layout.addWidget(self.scroll_area)
        
        # 結果容器
        self.results_widget = QWidget()
        self.results_layout = QGridLayout()
        self.results_layout.setSpacing(30)
        self.results_layout.setContentsMargins(30, 30, 30, 30)
        self.results_widget.setLayout(self.results_layout)
        self.scroll_area.setWidget(self.results_widget)
        
        # 狀態列
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 海綿寶寶經典台詞
        quotes = [
            "我準備好了！我準備好了！ 🧽",
            "今天是美好的一天！ ☀️",
            "想像力比知識更重要！ 🌈",
            "F代表朋友，U代表你和我！ 🎵",
            "我的褲子是方形的！ 📦"
        ]
        import random
        self.status_bar.showMessage(random.choice(quotes))
        
        # 初始歡迎訊息
        self.show_welcome_message()
    
    def show_welcome_message(self):
        """顯示歡迎訊息"""
        welcome_widget = QWidget()
        welcome_layout = QVBoxLayout()
        welcome_layout.setAlignment(Qt.AlignCenter)
        welcome_widget.setLayout(welcome_layout)
        
        # 海綿寶寶表情
        # icon_label = QLabel("🧽")
        # icon_label.setStyleSheet("font-size: 120px;")
        # icon_label.setAlignment(Qt.AlignCenter)
        # welcome_layout.addWidget(icon_label)
        
        # 歡迎文字
        welcome_label = QLabel("哈囉！我是海綿寶寶！")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("""
            QLabel {
                color: #F4D03F;
                font-size: 36px;
                font-weight: bold;
                margin: 30px 0;
            }
        """)
        welcome_layout.addWidget(welcome_label)
        
        # 提示文字
        hint_label = QLabel("告訴我你想要什麼樣的梗圖，我會幫你找到最棒的！")
        hint_label.setAlignment(Qt.AlignCenter)
        hint_label.setStyleSheet("""
            QLabel {
                color: #B8C5D6;
                font-size: 24px;
                margin-bottom: 20px;
            }
        """)
        welcome_layout.addWidget(hint_label)
        
        # 範例提示
        example_label = QLabel("💡 試試看：「我準備好了」、「章魚哥」、「派大星」")
        example_label.setAlignment(Qt.AlignCenter)
        example_label.setStyleSheet("""
            QLabel {
                color: #7A8CA0;
                font-size: 20px;
                font-style: italic;
            }
        """)
        welcome_layout.addWidget(example_label)
        
        self.results_layout.addWidget(welcome_widget, 0, 0, 1, 3, Qt.AlignCenter)
    
    def clear_results(self):
        """清除搜尋結果"""
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def load_resources(self):
        """載入模型和索引"""
        try:
            self.status_bar.showMessage("正在進入比奇堡... 🏝️")
            
            # 載入 Faiss 索引
            index_files = ["spongebob_jina.index", "index.faiss", "spongebob.index", "image.index"]
            index_loaded = False
            
            for index_file in index_files:
                if os.path.exists(index_file):
                    self.index = faiss.read_index(index_file)
                    logger.info(f"成功載入索引: {index_file}")
                    index_loaded = True
                    break
            
            if not index_loaded:
                raise FileNotFoundError("找不到索引檔案")
            
            # 載入 metadata
            metadata_files = ["spongebob_jina_metadata.json", "metadata.json", "spongebob_metadata.json", "image_metadata.json"]
            metadata_loaded = False
            
            for metadata_file in metadata_files:
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)
                    logger.info(f"成功載入 metadata: {metadata_file}")
                    metadata_loaded = True
                    break
            
            if not metadata_loaded:
                raise FileNotFoundError("找不到 metadata 檔案")
            
            self.status_bar.showMessage(f"比奇堡準備就緒！共有 {self.index.ntotal} 張梗圖等你發現 🎉")
            
        except Exception as e:
            logger.error(f"載入資源失敗: {e}")
            QMessageBox.critical(self, "糟糕！", f"無法進入比奇堡:\n{str(e)}\n\n請確認所有檔案都在正確位置！")
            self.search_btn.setEnabled(False)
    
    def on_search_clicked(self):
        """處理搜尋按鈕點擊"""
        text = self.search_input.text().strip()
        if not text:
            return
        
        # 檢查資源
        if not all([self.index, self.metadata]):
            QMessageBox.warning(self, "等一下！", "系統還在準備中，請稍後再試！")
            return
        
        # 禁用搜尋
        self.search_btn.setEnabled(False)
        self.search_input.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        # 清除舊結果
        self.clear_results()
        
        # 啟動搜尋
        self.search_worker = SearchWorker(text, self.index, self.metadata)
        self.search_worker.progress.connect(self.on_search_progress)
        self.search_worker.result_ready.connect(self.on_search_complete)
        self.search_worker.error.connect(self.on_search_error)
        self.search_worker.start()
    
    def on_search_progress(self, message: str):
        """更新搜尋進度"""
        self.status_bar.showMessage(f"{message} 🔍")
    
    def on_search_complete(self, results: List[Dict]):
        """處理搜尋結果"""
        self.progress_bar.setVisible(False)
        self.search_btn.setEnabled(True)
        self.search_input.setEnabled(True)
        
        if not results:
            # 無結果提示
            no_result_widget = QWidget()
            no_result_layout = QVBoxLayout()
            no_result_layout.setAlignment(Qt.AlignCenter)
            no_result_widget.setLayout(no_result_layout)
            
            # 章魚哥表情
            icon_label = QLabel("🦑")
            icon_label.setStyleSheet("font-size: 100px;")
            icon_label.setAlignment(Qt.AlignCenter)
            no_result_layout.addWidget(icon_label)
            
            text_label = QLabel("找不到相關的梗圖...")
            text_label.setAlignment(Qt.AlignCenter)
            text_label.setStyleSheet("""
                QLabel {
                    color: #B8C5D6;
                    font-size: 28px;
                    margin: 20px 0;
                }
            """)
            no_result_layout.addWidget(text_label)
            
            hint_label = QLabel("章魚哥說：試試其他關鍵字吧！")
            hint_label.setAlignment(Qt.AlignCenter)
            hint_label.setStyleSheet("""
                QLabel {
                    color: #7A8CA0;
                    font-size: 22px;
                }
            """)
            no_result_layout.addWidget(hint_label)
            
            self.results_layout.addWidget(no_result_widget, 0, 0, 1, 3, Qt.AlignCenter)
            self.status_bar.showMessage("章魚哥：真是浪費時間... 😤")
            return

        # 顯示結果 (4x2 網格)
        cols = 4
        for i, result in enumerate(results[:8]):  # 只顯示前8個
            row = i // cols
            col = i % cols
            
            # 建立圖片卡片
            card = ImageCard(
                result["path"],
                result["name"],
                result["score"],
                result["rank"]
            )
            card.clicked.connect(self.show_image_detail)
            
            self.results_layout.addWidget(card, row, col, Qt.AlignCenter)
        
        # 更新狀態列
        happy_messages = [
            f"太棒了！找到 8 個超讚的梗圖！ 🎉",
            f"我準備好了！找到 8 個梗圖！ 🧽",
            f"耶！派大星會喜歡這 8 個梗圖的！ ⭐",
            f"蟹堡王等級的搜尋結果：8 個梗圖！ 🦀"
        ]
        import random
        self.status_bar.showMessage(random.choice(happy_messages))
    
    def show_image_detail(self, image_path: str, image_name: str, score: float, rank: int):
        """顯示圖片詳情"""
        dialog = SpongeBobImageDialog(image_path, image_name, score, rank, self)
        dialog.exec_()
    
    def on_search_error(self, error_msg: str):
        """處理搜尋錯誤"""
        self.progress_bar.setVisible(False)
        self.search_btn.setEnabled(True)
        self.search_input.setEnabled(True)
        
        logger.error(f"搜尋錯誤: {error_msg}")
        
        # 顯示錯誤訊息
        msg = QMessageBox(self)
        msg.setWindowTitle("噢不！")
        msg.setText(f"派大星把機器搞壞了！\n\n錯誤訊息：{error_msg}")
        msg.setIcon(QMessageBox.Warning)
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #162B45;
                color: #F5F7FA;
            }
            QMessageBox QPushButton {
                background-color: #FF8FA3;
                color: white;
                padding: 10px 25px;
                border-radius: 15px;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        msg.exec_()
        
        self.status_bar.showMessage("派大星：對不起... 😢")
    
    def resizeEvent(self, event):
        """處理視窗大小改變事件，實現響應式設計"""
        super().resizeEvent(event)
        
        # 根據視窗寬度調整網格列數和卡片大小
        width = self.width()
        
        # 動態調整字體大小
        if width < 1200:
            font_scale = 0.85
        elif width < 1600:
            font_scale = 1.0
        else:
            font_scale = 1.15
        
        # 更新搜尋框和按鈕的字體大小
        if hasattr(self, 'search_input'):
            self.search_input.setStyleSheet(f"""
                QLineEdit {{
                    background-color: #162B45;
                    border: 3px solid #1E3A5F;
                    border-radius: 30px;
                    padding: 16px 32px;
                    font-size: {int(20 * font_scale)}px;
                    color: #F5F7FA;
                    selection-background-color: #F4D03F;
                    selection-color: #0A1628;
                }}
                QLineEdit:focus {{
                    border-color: #F4D03F;
                    background-color: #1E3A5F;
                }}
                QLineEdit::placeholder {{
                    color: #7A8CA0;
                }}
            """)
        
        if hasattr(self, 'search_btn'):
            self.search_btn.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #FF8FA3, stop:1 #FFB6C1);
                    color: white;
                    border: none;
                    padding: 16px 40px;
                    border-radius: 30px;
                    font-size: {int(20 * font_scale)}px;
                    font-weight: bold;
                    min-width: 160px;
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #FF7A8F, stop:1 #FFA0B0);
                }}
                QPushButton:pressed {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #FF6A7F, stop:1 #FF90A0);
                }}
                QPushButton:disabled {{
                    background: #4A5568;
                    color: #718096;
                }}
            """)


def main():
    """主函數"""
    app = QApplication(sys.argv)
    
    # 設定應用程式資訊
    app.setApplicationName("海綿寶寶梗圖推薦機器人")
    app.setOrganizationName("比奇堡科技")
    
    # 設定全域字體
    app.setStyleSheet("""
        * {
            font-family: "Microsoft JhengHei", "微軟正黑體", "PingFang TC", "Arial", sans-serif;
        }
    """)
    
    # 建立並顯示主視窗
    window = SpongeBobMainWindow()
    window.show()
    
    # 讓視窗可以最大化
    window.showMaximized()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
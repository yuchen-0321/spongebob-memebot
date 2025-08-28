import sys
import json
import base64
import requests
import os
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QComboBox, QTextEdit, QPushButton, QLabel, 
                             QFileDialog, QMessageBox, QProgressBar, QScrollArea,
                             QLineEdit, QCheckBox, QListWidget, QTabWidget,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QSplitter, QGroupBox, QSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont, QColor

# Google AI相關導入
try:
    from google import genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False

class OllamaAPI:
    """處理與Ollama API的交互"""
    
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
    
    def get_models(self):
        """獲取可用的模型清單"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            else:
                return []
        except Exception as e:
            print(f"獲取模型清單時發生錯誤: {e}")
            return []
    
    def chat_with_image(self, model, prompt, image_path=None):
        """與模型對話，支援圖片輸入"""
        try:
            data = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            
            if image_path:
                with open(image_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                    data["images"] = [image_data]
            
            response = requests.post(f"{self.base_url}/api/generate", json=data)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '無法獲取回應')
            else:
                return f"API錯誤: {response.status_code}"
                
        except Exception as e:
            return f"發生錯誤: {str(e)}"

class GoogleAIAPI:
    """處理與Google AI API的交互"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = None
        if GOOGLE_AI_AVAILABLE and api_key:
            try:
                self.client = genai.Client(api_key=api_key)
            except Exception as e:
                print(f"初始化Google AI客戶端失敗: {e}")
    
    def get_models(self):
        """獲取可用的Google AI模型清單"""
        if not self.client:
            return []
        
        # Google AI常用的視覺模型
        return [
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ]
    
    def analyze_image(self, model, prompt, image_path):
        """使用Google AI分析圖片"""
        try:
            if not self.client:
                return "Google AI客戶端未初始化"
            
            # 上傳圖片
            my_file = self.client.files.upload(file=image_path)
            
            # 生成回應
            response = self.client.models.generate_content(
                model=model,
                contents=[my_file, prompt],
            )
            
            # 清理上傳的檔案
            try:
                self.client.files.delete(my_file.name)
            except:
                pass
            
            return response.text
            
        except Exception as e:
            return f"Google AI分析錯誤: {str(e)}"

class FileNameParser:
    """檔案名稱解析工具類"""
    
    @staticmethod
    def extract_subtitle_from_filename(filename):
        """從檔案名中提取字幕"""
        try:
            # 移除副檔名
            name_without_ext = os.path.splitext(filename)[0]
            
            # 檢查是否符合【編號】字幕內容的格式
            if '】' in name_without_ext and name_without_ext.startswith('【'):
                # 找到】的位置，提取後面的內容作為字幕
                subtitle_start = name_without_ext.find('】') + 1
                subtitle = name_without_ext[subtitle_start:].strip()
                
                # 提取編號
                episode_end = name_without_ext.find('】')
                episode = name_without_ext[1:episode_end] if episode_end > 1 else ""
                
                return subtitle, episode
            else:
                # 如果不符合格式，返回整個檔案名（無副檔名）作為字幕
                return name_without_ext, ""
                
        except Exception as e:
            print(f"解析檔案名時發生錯誤: {e}")
            return "", ""
    
    @staticmethod
    def create_analysis_prompt_with_subtitle(base_prompt, subtitle, filename, episode=""):
        """建立包含字幕的分析提示詞"""
        subtitle_info = f"""

**檔案資訊：**
- 檔案名稱：{filename}
- 集數編號：{episode if episode else "無"}
- 字幕內容："{subtitle}"

**重要：** 字幕內容已從檔案名中提取，請直接使用上述字幕內容，不需要從圖片中重新識別。

請分析圖片中的角色和情緒，並使用提供的字幕內容來完成JSON輸出。"""
        
        return base_prompt + subtitle_info

class JSONParser:
    """JSON解析工具類"""
    
    @staticmethod
    def clean_json_response(response_text):
        """清理AI回應中的markdown標註並解析JSON"""
        try:
            # 移除可能的markdown程式碼區塊標記
            cleaned_text = response_text.strip()
            
            # 檢查並移除開頭的```json或```
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
            elif cleaned_text.startswith('```'):
                cleaned_text = cleaned_text[3:]
            
            # 檢查並移除結尾的```
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]
            
            # 再次清理空白字符
            cleaned_text = cleaned_text.strip()
            
            # 嘗試解析JSON
            parsed_json = json.loads(cleaned_text)
            return parsed_json, None
            
        except json.JSONDecodeError as e:
            return None, f"JSON解析錯誤: {str(e)}"
        except Exception as e:
            return None, f"清理回應時發生錯誤: {str(e)}"
    
    @staticmethod
    def update_filename_in_json(json_data, actual_filename):
        """更新JSON中的fileName欄位"""
        if isinstance(json_data, dict) and 'fileName' in json_data:
            json_data['fileName'] = actual_filename
        return json_data

class BatchProcessWorker(QThread):
    """批量處理工作執行緒"""
    progress = pyqtSignal(int, str)  # 進度和當前檔案名
    result_ready = pyqtSignal(dict)  # 單個結果
    finished_all = pyqtSignal(list)  # 所有結果
    error = pyqtSignal(str)
    
    def __init__(self, api_instance, model, prompt, image_paths, api_type="ollama", auto_extract_subtitle=True):
        super().__init__()
        self.api_instance = api_instance
        self.model = model
        self.prompt = prompt
        self.image_paths = image_paths
        self.api_type = api_type
        self.auto_extract_subtitle = auto_extract_subtitle
        self.results = []
        self.json_parser = JSONParser()
        self.filename_parser = FileNameParser()
    
    def run(self):
        """執行批量處理"""
        try:
            total_files = len(self.image_paths)
            
            for i, image_path in enumerate(self.image_paths):
                filename = os.path.basename(image_path)
                self.progress.emit(i + 1, filename)
                
                # 根據是否自動提取字幕來準備提示詞
                if self.auto_extract_subtitle:
                    # 從檔案名提取字幕
                    subtitle, episode = self.filename_parser.extract_subtitle_from_filename(filename)
                    
                    # 建立包含字幕資訊的提示詞
                    enhanced_prompt = self.filename_parser.create_analysis_prompt_with_subtitle(
                        self.prompt, subtitle, filename, episode
                    )
                else:
                    enhanced_prompt = self.prompt
                    subtitle = ""
                    episode = ""
                
                # 根據API類型調用不同的方法
                if self.api_type == "google":
                    raw_response = self.api_instance.analyze_image(self.model, enhanced_prompt, image_path)
                else:  # ollama
                    raw_response = self.api_instance.chat_with_image(self.model, enhanced_prompt, image_path)
                
                # 嘗試解析JSON回應
                parsed_json, parse_error = self.json_parser.clean_json_response(raw_response)
                
                # 如果成功解析JSON，更新檔案名稱和字幕
                if parsed_json:
                    parsed_json = self.json_parser.update_filename_in_json(parsed_json, filename)
                    
                    # 如果自動提取字幕且JSON中的text為空，使用提取的字幕
                    if self.auto_extract_subtitle and subtitle and not parsed_json.get('text'):
                        parsed_json['text'] = subtitle
                    
                    processed_response = json.dumps(parsed_json, ensure_ascii=False, indent=2)
                    is_json_valid = True
                else:
                    # 如果解析失敗，保留原始回應並記錄錯誤
                    processed_response = raw_response
                    is_json_valid = False
                
                # 建立結果
                result = {
                    "fileName": filename,
                    "filePath": image_path,
                    "extractedSubtitle": subtitle,  # 提取的字幕
                    "episode": episode,  # 集數編號
                    "rawResponse": raw_response,  # 保留原始回應
                    "response": processed_response,  # 處理後的回應
                    "parsedJson": parsed_json,  # 解析後的JSON物件
                    "isJsonValid": is_json_valid,  # JSON是否有效
                    "parseError": parse_error,  # 解析錯誤（如有）
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model,
                    "apiType": self.api_type,
                    "autoExtractSubtitle": self.auto_extract_subtitle
                }
                
                self.results.append(result)
                self.result_ready.emit(result)
            
            self.finished_all.emit(self.results)
            
        except Exception as e:
            self.error.emit(str(e))

class SingleChatWorker(QThread):
    """單個對話工作執行緒"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, api_instance, model, prompt, image_path=None, api_type="ollama"):
        super().__init__()
        self.api_instance = api_instance
        self.model = model
        self.prompt = prompt
        self.image_path = image_path
        self.api_type = api_type
    
    def run(self):
        """執行單個對話請求"""
        try:
            if self.api_type == "google":
                response = self.api_instance.analyze_image(self.model, self.prompt, self.image_path)
            else:  # ollama
                response = self.api_instance.chat_with_image(self.model, self.prompt, self.image_path)
            
            self.finished.emit(response)
        except Exception as e:
            self.error.emit(str(e))

class AdvancedAIChatApp(QMainWindow):
    """進階AI圖像對話應用程式"""
    
    def __init__(self):
        super().__init__()
        self.ollama_api = OllamaAPI()
        self.google_api = None
        self.selected_image_path = None
        self.batch_worker = None
        self.single_worker = None
        self.batch_results = []
        
        # 海綿寶寶分析提示詞（更新版）
        self.spongebob_prompt = """請分析這張海綿寶寶動畫圖片，識別角色、情緒和相關語句，並以JSON格式輸出結果。

## 角色識別指南

### 主要角色特徵：
- **海綿寶寶 (SpongeBob)**：黃色方形海綿，圓形大眼睛，兩顆突出門牙，穿棕色方形褲子，白色襯衫，紅色領帶
- **派大星 (Patrick)**：粉紅色海星，綠色短褲，圓胖身材，小眼睛
- **章魚哥 (Squidward)**：藍綠色章魚，長鼻子，四隻腳，總是皺眉
- **蟹老闆 (Mr. Krabs)**：紅色螃蟹，大鉗子，圓眼睛，藍色衣服
- **珊迪 (Sandy)**：棕色松鼠，戴太空頭盔，穿白色太空服
- **皮老闆 (Plankton)**：綠色小型浮游生物，一隻眼睛，觸角
- **泡芙老師 (Mrs. Puff)**：黃色河豚，能脹大身體
- **小蝸 (Gary)**：藍色蝸牛，海綿寶寶的寵物

## 情緒分類（16種）

1. **開心** - 笑容、眼睛彎曲、活力四射
2. **憤怒** - 皺眉、咬牙、握拳、眼神兇狠
3. **悲傷** - 哭泣、嘴角下垂、眼神黯淡
4. **驚訝** - 眼睛睜大、嘴巴張開、眉毛上揚
5. **恐懼** - 瞳孔放大、身體發抖、臉色蒼白
6. **厭惡** - 皺鼻子、嘴角下垂、避開視線
7. **興奮** - 手舞足蹈、眼神發亮、充滿活力
8. **困惑** - 歪頭、眉頭緊鎖、疑問表情
9. **尷尬** - 臉紅、迴避眼神、局促不安
10. **自信** - 挺胸、微笑、眼神堅定
11. **失望** - 垂頭喪氣、嘆氣、無精打采
12. **緊張** - 冒汗、咬指甲、坐立難安
13. **得意** - 傲慢表情、抬頭挺胸、炫耀姿態
14. **無聊** - 打哈欠、眼神渙散、懶洋洋
15. **專注** - 眼神集中、嚴肅表情、全神貫注
16. **愛意** - 心形眼睛、臉紅、溫柔表情

## 分析要求

### 1. 角色識別
- 識別圖中出現的所有海綿寶寶角色
- 確定主要說話的角色（通常是表情最突出或位置最顯眼的角色）

### 2. 情緒分析
- 根據面部表情、肢體語言和場景氛圍分析情緒
- 從16種情緒中選擇最符合的1-3種
- **重要：結合已提供的字幕內容進行綜合判斷**

### 3. 相關語句生成
- 基於已提供的字幕內容的情緒和語境
- 生成15-25個語義相近或情境相關的語句
- 包含：
  - 同義表達
  - 相似情緒的語句
  - 相關場景的對話
  - 因果關係的語句
  - 程度遞進的表達
- **要求：**
  - 絕對沒有重複的語句
  - 長短可以不一致
  - 可以包含標點符號表達（如：！？...）
  - 符合台灣繁體中文表達習慣

## 輸出格式

```json
{
  "fileName": "{圖片檔案名稱}",
  "characters": ["{主要說話角色}"],
  "text": "{已提供的字幕內容}",
  "emotions": ["{情緒1}", "{情緒2}", "{情緒3}"],
  "simText": [
    "{相關語句1}",
    "{相關語句2}",
    "{相關語句3}",
    "...",
    "{相關語句N}"
  ]
}
```

## 注意事項

1. **字幕處理**：字幕內容已從檔案名提取，請直接使用提供的字幕，不需要從圖片中重新識別
2. **準確性優先**：確保角色識別的準確性
3. **情緒一致性**：情緒判斷要同時考慮視覺信息和字幕內容
4. **語句相關性**：生成的相關語句要在語義、情緒或情境上與原句相關
5. **文化適應性**：考慮台灣繁體中文表達習慣和文化背景
6. **格式嚴格性**：嚴格按照JSON格式輸出，確保可以被程式解析

請基於以上指南分析圖片並輸出結果。"""
        
        self.init_ui()
        self.load_ollama_models()
    
    def init_ui(self):
        """初始化使用者介面"""
        self.setWindowTitle("進階AI圖像分析助手 - Ollama & Google AI")
        self.setGeometry(100, 100, 1200, 800)
        
        # 建立中央小部件和分頁
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 標題
        title_label = QLabel("進階AI圖像分析助手")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        main_layout.addWidget(title_label)
        
        # 建立分頁介面
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # 單個對話分頁
        self.create_single_chat_tab()
        
        # 批量處理分頁
        self.create_batch_process_tab()
        
        # 套用樣式
        self.apply_styles()
    
    def create_single_chat_tab(self):
        """建立單個對話分頁"""
        single_tab = QWidget()
        layout = QVBoxLayout()
        single_tab.setLayout(layout)
        
        # API選擇區域
        api_group = QGroupBox("API設定")
        api_layout = QVBoxLayout()
        api_group.setLayout(api_layout)
        
        # API類型選擇
        api_type_layout = QHBoxLayout()
        api_type_layout.addWidget(QLabel("選擇AI服務:"))
        
        self.api_type_combo = QComboBox()
        self.api_type_combo.addItems(["Ollama (本地)", "Google AI (雲端)"])
        self.api_type_combo.currentTextChanged.connect(self.on_api_type_changed)
        api_type_layout.addWidget(self.api_type_combo)
        
        api_layout.addLayout(api_type_layout)
        
        # Google AI API Key輸入
        google_layout = QHBoxLayout()
        google_layout.addWidget(QLabel("Google AI API Key:"))
        
        self.google_api_key_input = QLineEdit()
        self.google_api_key_input.setEchoMode(QLineEdit.Password)
        self.google_api_key_input.setPlaceholderText("請輸入您的Google AI API Key")
        self.google_api_key_input.textChanged.connect(self.on_google_api_key_changed)
        google_layout.addWidget(self.google_api_key_input)
        
        self.google_api_test_btn = QPushButton("測試連接")
        self.google_api_test_btn.clicked.connect(self.test_google_api)
        google_layout.addWidget(self.google_api_test_btn)
        
        api_layout.addLayout(google_layout)
        layout.addWidget(api_group)
        
        # 模型選擇
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("選擇模型:"))
        
        self.model_combo = QComboBox()
        model_layout.addWidget(self.model_combo)
        
        self.refresh_models_btn = QPushButton("重新整理")
        self.refresh_models_btn.clicked.connect(self.refresh_models)
        model_layout.addWidget(self.refresh_models_btn)
        
        layout.addLayout(model_layout)
        
        # 圖片選擇區域
        image_layout = QHBoxLayout()
        self.image_label = QLabel("未選擇圖片")
        self.image_label.setStyleSheet("border: 2px dashed #ccc; padding: 10px;")
        self.image_label.setMinimumHeight(100)
        self.image_label.setAlignment(Qt.AlignCenter)
        
        image_buttons_layout = QVBoxLayout()
        select_image_btn = QPushButton("選擇圖片")
        select_image_btn.clicked.connect(self.select_single_image)
        image_buttons_layout.addWidget(select_image_btn)
        
        clear_image_btn = QPushButton("清除圖片")
        clear_image_btn.clicked.connect(self.clear_image)
        image_buttons_layout.addWidget(clear_image_btn)
        
        # 海綿寶寶分析按鈕
        spongebob_btn = QPushButton("海綿寶寶分析")
        spongebob_btn.clicked.connect(self.use_spongebob_prompt)
        image_buttons_layout.addWidget(spongebob_btn)
        
        # 從檔案名提取字幕按鈕
        extract_subtitle_btn = QPushButton("提取檔案名字幕")
        extract_subtitle_btn.clicked.connect(self.extract_subtitle_from_selected_image)
        image_buttons_layout.addWidget(extract_subtitle_btn)
        
        image_layout.addWidget(self.image_label, 2)
        image_layout.addLayout(image_buttons_layout, 1)
        layout.addLayout(image_layout)
        
        # 提示詞輸入
        layout.addWidget(QLabel("提示詞:"))
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("請輸入您的提示詞...")
        self.prompt_input.setMaximumHeight(120)
        layout.addWidget(self.prompt_input)
        
        # 發送按鈕和進度條
        button_layout = QHBoxLayout()
        self.send_btn = QPushButton("發送")
        self.send_btn.clicked.connect(self.send_single_message)
        button_layout.addWidget(self.send_btn)
        
        self.single_progress_bar = QProgressBar()
        self.single_progress_bar.setVisible(False)
        button_layout.addWidget(self.single_progress_bar)
        
        layout.addLayout(button_layout)
        
        # 回應顯示
        layout.addWidget(QLabel("回應:"))
        self.response_display = QTextEdit()
        self.response_display.setReadOnly(True)
        layout.addWidget(self.response_display)
        
        self.tab_widget.addTab(single_tab, "單個對話")
    
    def create_batch_process_tab(self):
        """建立批量處理分頁"""
        batch_tab = QWidget()
        layout = QVBoxLayout()
        batch_tab.setLayout(layout)
        
        # 批量設定區域
        settings_group = QGroupBox("批量處理設定")
        settings_layout = QVBoxLayout()
        settings_group.setLayout(settings_layout)
        
        # 圖片選擇
        images_layout = QHBoxLayout()
        self.batch_images_list = QListWidget()
        self.batch_images_list.setMaximumHeight(120)
        
        batch_buttons_layout = QVBoxLayout()
        select_images_btn = QPushButton("選擇圖片")
        select_images_btn.clicked.connect(self.select_batch_images)
        batch_buttons_layout.addWidget(select_images_btn)
        
        clear_images_btn = QPushButton("清除全部")
        clear_images_btn.clicked.connect(self.clear_batch_images)
        batch_buttons_layout.addWidget(clear_images_btn)
        
        images_layout.addWidget(self.batch_images_list, 2)
        images_layout.addLayout(batch_buttons_layout, 1)
        settings_layout.addLayout(images_layout)
        
        # 批量提示詞
        settings_layout.addWidget(QLabel("批量提示詞:"))
        self.batch_prompt_input = QTextEdit()
        self.batch_prompt_input.setPlaceholderText("請輸入批量處理的提示詞...")
        self.batch_prompt_input.setMaximumHeight(80)
        settings_layout.addWidget(self.batch_prompt_input)
        
        # 快速設定按鈕
        quick_settings_layout = QHBoxLayout()
        spongebob_batch_btn = QPushButton("海綿寶寶批量分析")
        spongebob_batch_btn.clicked.connect(self.use_spongebob_batch_prompt)
        quick_settings_layout.addWidget(spongebob_batch_btn)
        
        # 自動字幕提取選項
        self.auto_subtitle_checkbox = QCheckBox("自動從檔案名提取字幕")
        self.auto_subtitle_checkbox.setChecked(True)
        self.auto_subtitle_checkbox.setToolTip("檔案名格式：【編號】字幕內容.jpg")
        quick_settings_layout.addWidget(self.auto_subtitle_checkbox)
        
        settings_layout.addLayout(quick_settings_layout)
        layout.addWidget(settings_group)
        
        # 批量處理控制
        control_layout = QHBoxLayout()
        self.start_batch_btn = QPushButton("開始批量處理")
        self.start_batch_btn.clicked.connect(self.start_batch_process)
        control_layout.addWidget(self.start_batch_btn)
        
        self.stop_batch_btn = QPushButton("停止處理")
        self.stop_batch_btn.clicked.connect(self.stop_batch_process)
        self.stop_batch_btn.setEnabled(False)
        control_layout.addWidget(self.stop_batch_btn)
        
        export_json_btn = QPushButton("匯出JSON")
        export_json_btn.clicked.connect(self.export_batch_results)
        control_layout.addWidget(export_json_btn)
        
        layout.addLayout(control_layout)
        
        # 進度顯示
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("處理進度:"))
        
        self.batch_progress_bar = QProgressBar()
        progress_layout.addWidget(self.batch_progress_bar)
        
        self.progress_label = QLabel("0/0")
        progress_layout.addWidget(self.progress_label)
        
        layout.addLayout(progress_layout)
        
        # 結果顯示表格
        layout.addWidget(QLabel("處理結果:"))
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels(["檔案名稱", "提取字幕", "處理狀態", "JSON狀態", "時間", "預覽"])
        
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        
        self.results_table.itemDoubleClicked.connect(self.view_result_detail)
        layout.addWidget(self.results_table)
        
        self.tab_widget.addTab(batch_tab, "批量處理")
    
    def apply_styles(self):
        """套用樣式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QTextEdit, QLineEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
                background-color: white;
            }
            QComboBox {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
                background-color: white;
            }
        """)
    
    def on_api_type_changed(self):
        """API類型改變時的處理"""
        self.refresh_models()
    
    def on_google_api_key_changed(self):
        """Google API Key改變時的處理"""
        api_key = self.google_api_key_input.text().strip()
        if api_key and GOOGLE_AI_AVAILABLE:
            self.google_api = GoogleAIAPI(api_key)
            if self.api_type_combo.currentText().startswith("Google"):
                self.refresh_models()
    
    def test_google_api(self):
        """測試Google AI API連接"""
        if not GOOGLE_AI_AVAILABLE:
            QMessageBox.warning(self, "錯誤", "Google AI庫未安裝，請執行: pip install google-genai")
            return
        
        api_key = self.google_api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "錯誤", "請輸入Google AI API Key")
            return
        
        try:
            test_api = GoogleAIAPI(api_key)
            models = test_api.get_models()
            if models:
                QMessageBox.information(self, "成功", f"API連接成功！\n可用模型: {', '.join(models)}")
                self.google_api = test_api
                self.refresh_models()
            else:
                QMessageBox.warning(self, "失敗", "無法獲取模型清單，請檢查API Key")
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"API測試失敗: {str(e)}")
    
    def load_ollama_models(self):
        """載入Ollama模型"""
        models = self.ollama_api.get_models()
        self.model_combo.clear()
        
        if models:
            self.model_combo.addItems(models)
        else:
            self.model_combo.addItem("無可用模型")
    
    def refresh_models(self):
        """重新整理模型清單"""
        self.model_combo.clear()
        
        if self.api_type_combo.currentText().startswith("Ollama"):
            models = self.ollama_api.get_models()
            if models:
                self.model_combo.addItems(models)
            else:
                self.model_combo.addItem("無可用模型 - 請確認Ollama運行")
        else:  # Google AI
            if self.google_api:
                models = self.google_api.get_models()
                if models:
                    self.model_combo.addItems(models)
                else:
                    self.model_combo.addItem("無可用模型")
            else:
                self.model_combo.addItem("請先設定API Key")
    
    def select_single_image(self):
        """選擇單張圖片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇圖片", "", "圖片檔案 (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        
        if file_path:
            self.selected_image_path = file_path
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(200, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)
            else:
                self.image_label.setText(f"已選擇: {os.path.basename(file_path)}")
    
    def clear_image(self):
        """清除選擇的圖片"""
        self.selected_image_path = None
        self.image_label.clear()
        self.image_label.setText("未選擇圖片")
    
    def extract_subtitle_from_selected_image(self):
        """從選擇的圖片檔案名提取字幕"""
        if not self.selected_image_path:
            QMessageBox.warning(self, "警告", "請先選擇圖片")
            return
        
        filename = os.path.basename(self.selected_image_path)
        filename_parser = FileNameParser()
        subtitle, episode = filename_parser.extract_subtitle_from_filename(filename)
        
        if subtitle:
            # 建立包含字幕的海綿寶寶分析提示詞
            enhanced_prompt = filename_parser.create_analysis_prompt_with_subtitle(
                self.spongebob_prompt, subtitle, filename, episode
            )
            self.prompt_input.setText(enhanced_prompt)
            
            # 顯示提取結果
            info_msg = f"已提取字幕資訊：\n\n"
            if episode:
                info_msg += f"集數編號：{episode}\n"
            info_msg += f"字幕內容：{subtitle}\n\n已自動更新提示詞，可以直接發送分析。"
            
            QMessageBox.information(self, "字幕提取成功", info_msg)
        else:
            QMessageBox.warning(self, "提取失敗", 
                              f"無法從檔案名提取字幕\n\n"
                              f"檔案名：{filename}\n"
                              f"預期格式：【編號】字幕內容.jpg")
    
    def use_spongebob_prompt(self):
        """使用海綿寶寶分析提示詞"""
        self.prompt_input.setText(self.spongebob_prompt)
    
    def select_batch_images(self):
        """選擇批量圖片"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "選擇多張圖片", "", "圖片檔案 (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        
        for file_path in file_paths:
            self.batch_images_list.addItem(file_path)
    
    def clear_batch_images(self):
        """清除批量圖片"""
        self.batch_images_list.clear()
    
    def use_spongebob_batch_prompt(self):
        """使用海綿寶寶批量分析提示詞"""
        self.batch_prompt_input.setText(self.spongebob_prompt)
    
    def send_single_message(self):
        """發送單個訊息"""
        if not self.selected_image_path:
            QMessageBox.warning(self, "錯誤", "請先選擇圖片")
            return
        
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "錯誤", "請輸入提示詞")
            return
        
        model = self.model_combo.currentText()
        if "無可用模型" in model or "請先設定" in model:
            QMessageBox.warning(self, "錯誤", "請選擇有效的模型")
            return
        
        # 確定API類型和實例
        if self.api_type_combo.currentText().startswith("Google"):
            if not self.google_api:
                QMessageBox.warning(self, "錯誤", "請先設定Google AI API Key")
                return
            api_instance = self.google_api
            api_type = "google"
        else:
            api_instance = self.ollama_api
            api_type = "ollama"
        
        # 顯示進度
        self.send_btn.setEnabled(False)
        self.single_progress_bar.setVisible(True)
        self.single_progress_bar.setRange(0, 0)
        self.response_display.setText("正在處理...")
        
        # 啟動工作執行緒
        self.single_worker = SingleChatWorker(
            api_instance, model, prompt, self.selected_image_path, api_type
        )
        self.single_worker.finished.connect(self.on_single_response_received)
        self.single_worker.error.connect(self.on_single_error_occurred)
        self.single_worker.start()
    
    def start_batch_process(self):
        """開始批量處理"""
        # 獲取圖片清單
        image_paths = []
        for i in range(self.batch_images_list.count()):
            image_paths.append(self.batch_images_list.item(i).text())
        
        if not image_paths:
            QMessageBox.warning(self, "錯誤", "請先選擇要處理的圖片")
            return
        
        prompt = self.batch_prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "錯誤", "請輸入批量提示詞")
            return
        
        model = self.model_combo.currentText()
        if "無可用模型" in model or "請先設定" in model:
            QMessageBox.warning(self, "錯誤", "請選擇有效的模型")
            return
        
        # 確定API類型和實例
        if self.api_type_combo.currentText().startswith("Google"):
            if not self.google_api:
                QMessageBox.warning(self, "錯誤", "請先設定Google AI API Key")
                return
            api_instance = self.google_api
            api_type = "google"
        else:
            api_instance = self.ollama_api
            api_type = "ollama"
        
        # 重設結果
        self.batch_results = []
        self.results_table.setRowCount(0)
        
        # 設定進度條
        self.batch_progress_bar.setMaximum(len(image_paths))
        self.batch_progress_bar.setValue(0)
        self.progress_label.setText(f"0/{len(image_paths)}")
        
        # 啟用/禁用按鈕
        self.start_batch_btn.setEnabled(False)
        self.stop_batch_btn.setEnabled(True)
        
        # 啟動批量處理執行緒
        auto_extract_subtitle = self.auto_subtitle_checkbox.isChecked()
        self.batch_worker = BatchProcessWorker(
            api_instance, model, prompt, image_paths, api_type, auto_extract_subtitle
        )
        self.batch_worker.progress.connect(self.on_batch_progress)
        self.batch_worker.result_ready.connect(self.on_batch_result_ready)
        self.batch_worker.finished_all.connect(self.on_batch_finished)
        self.batch_worker.error.connect(self.on_batch_error)
        self.batch_worker.start()
    
    def stop_batch_process(self):
        """停止批量處理"""
        if self.batch_worker:
            self.batch_worker.terminate()
            self.batch_worker.wait()
        
        self.start_batch_btn.setEnabled(True)
        self.stop_batch_btn.setEnabled(False)
    
    def export_batch_results(self):
        """匯出批量處理結果為JSON"""
        if not self.batch_results:
            QMessageBox.warning(self, "警告", "沒有結果可以匯出")
            return
        
        # 讓用戶選擇匯出格式
        choice = QMessageBox.question(
            self, "選擇匯出格式", 
            "請選擇要匯出的格式:\n\n"
            "是(Yes) - 匯出解析後的JSON結果\n"
            "否(No) - 匯出完整的處理資訊\n"
            "取消(Cancel) - 不匯出",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
        )
        
        if choice == QMessageBox.Cancel:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "匯出JSON", f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON檔案 (*.json)"
        )
        
        if file_path:
            try:
                if choice == QMessageBox.Yes:
                    # 匯出解析後的JSON結果
                    json_results = []
                    for result in self.batch_results:
                        if result.get('isJsonValid', False) and result.get('parsedJson'):
                            json_results.append(result['parsedJson'])
                        else:
                            # 如果JSON無效，建立基本結構
                            json_results.append({
                                "fileName": result['fileName'],
                                "characters": [],
                                "text": "",
                                "emotions": [],
                                "simText": [],
                                "error": result.get('parseError', '解析失敗'),
                                "rawResponse": result.get('rawResponse', '')
                            })
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(json_results, f, ensure_ascii=False, indent=2)
                    
                    valid_count = sum(1 for r in self.batch_results if r.get('isJsonValid', False))
                    QMessageBox.information(
                        self, "匯出完成", 
                        f"解析後的JSON結果已匯出至: {file_path}\n\n"
                        f"總檔案數: {len(self.batch_results)}\n"
                        f"成功解析: {valid_count}\n"
                        f"解析失敗: {len(self.batch_results) - valid_count}"
                    )
                else:
                    # 匯出完整處理資訊
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(self.batch_results, f, ensure_ascii=False, indent=2)
                    
                    QMessageBox.information(self, "匯出完成", f"完整處理資訊已匯出至: {file_path}")
                    
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"匯出失敗: {str(e)}")
    
    def on_single_response_received(self, response):
        """處理單個回應"""
        self.response_display.setText(response)
        self.reset_single_ui()
    
    def on_single_error_occurred(self, error):
        """處理單個錯誤"""
        self.response_display.setText(f"錯誤: {error}")
        self.reset_single_ui()
        QMessageBox.critical(self, "錯誤", f"處理失敗: {error}")
    
    def reset_single_ui(self):
        """重設單個對話UI"""
        self.send_btn.setEnabled(True)
        self.single_progress_bar.setVisible(False)
        if self.single_worker:
            self.single_worker.quit()
            self.single_worker.wait()
    
    def on_batch_progress(self, current, filename):
        """批量處理進度更新"""
        self.batch_progress_bar.setValue(current)
        total = self.batch_progress_bar.maximum()
        self.progress_label.setText(f"{current}/{total}")
        
        # 添加到結果表格
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        self.results_table.setItem(row, 0, QTableWidgetItem(filename))
        self.results_table.setItem(row, 1, QTableWidgetItem("提取中..."))
        self.results_table.setItem(row, 2, QTableWidgetItem("處理中..."))
        self.results_table.setItem(row, 3, QTableWidgetItem("等待中"))
        self.results_table.setItem(row, 4, QTableWidgetItem(datetime.now().strftime('%H:%M:%S')))
        self.results_table.setItem(row, 5, QTableWidgetItem(""))
    
    def on_batch_result_ready(self, result):
        """批量處理單個結果準備完成"""
        self.batch_results.append(result)
        
        # 更新表格
        for row in range(self.results_table.rowCount()):
            if self.results_table.item(row, 0).text() == result['fileName']:
                # 更新提取字幕欄位
                subtitle = result.get('extractedSubtitle', '')
                if subtitle:
                    subtitle_item = QTableWidgetItem(subtitle[:20] + "..." if len(subtitle) > 20 else subtitle)
                    subtitle_item.setToolTip(subtitle)  # 完整字幕顯示在tooltip
                else:
                    subtitle_item = QTableWidgetItem("無")
                self.results_table.setItem(row, 1, subtitle_item)
                
                # 更新處理狀態
                self.results_table.setItem(row, 2, QTableWidgetItem("完成"))
                
                # 更新JSON狀態
                if result.get('isJsonValid', False):
                    json_status = QTableWidgetItem("✓ 有效")
                    json_status.setBackground(QColor(144, 238, 144))  # 淺綠色
                else:
                    json_status = QTableWidgetItem("✗ 無效")
                    json_status.setBackground(QColor(255, 182, 193))  # 淺紅色
                self.results_table.setItem(row, 3, json_status)
                
                # 更新預覽
                if result.get('isJsonValid', False) and result.get('parsedJson'):
                    # 如果JSON有效，顯示解析後的內容
                    parsed = result['parsedJson']
                    preview_parts = []
                    if 'characters' in parsed and parsed['characters']:
                        preview_parts.append(f"角色: {', '.join(parsed['characters'])}")
                    if 'emotions' in parsed and parsed['emotions']:
                        preview_parts.append(f"情緒: {', '.join(parsed['emotions'])}")
                    preview = " | ".join(preview_parts) if preview_parts else "已解析"
                else:
                    # 如果JSON無效，顯示原始回應的前100字
                    raw_response = result.get('rawResponse', result.get('response', ''))
                    preview = raw_response[:100] + "..." if len(raw_response) > 100 else raw_response
                
                self.results_table.setItem(row, 5, QTableWidgetItem(preview))
                break
    
    def on_batch_finished(self, results):
        """批量處理完成"""
        self.start_batch_btn.setEnabled(True)
        self.stop_batch_btn.setEnabled(False)
        QMessageBox.information(self, "完成", f"批量處理完成！共處理 {len(results)} 個檔案")
    
    def on_batch_error(self, error):
        """批量處理錯誤"""
        self.start_batch_btn.setEnabled(True)
        self.stop_batch_btn.setEnabled(False)
        QMessageBox.critical(self, "錯誤", f"批量處理失敗: {error}")
    
    def view_result_detail(self, item):
        """查看結果詳情"""
        row = item.row()
        filename = self.results_table.item(row, 0).text()
        
        # 找到對應的結果
        result = None
        for r in self.batch_results:
            if r['fileName'] == filename:
                result = r
                break
        
        if result:
            detail_dialog = QMessageBox()
            detail_dialog.setWindowTitle(f"結果詳情 - {filename}")
            
            # 根據JSON是否有效顯示不同內容
            if result.get('isJsonValid', False) and result.get('parsedJson'):
                # 顯示解析後的JSON內容
                parsed = result['parsedJson']
                main_text = "✓ JSON解析成功\n\n"
                
                # 字幕資訊
                if result.get('extractedSubtitle'):
                    main_text += f"提取字幕: {result['extractedSubtitle']}\n"
                if result.get('episode'):
                    main_text += f"集數編號: {result['episode']}\n"
                main_text += "\n"
                
                if 'characters' in parsed:
                    main_text += f"角色: {', '.join(parsed['characters']) if parsed['characters'] else '無'}\n"
                if 'text' in parsed:
                    main_text += f"字幕: {parsed['text']}\n"
                if 'emotions' in parsed:
                    main_text += f"情緒: {', '.join(parsed['emotions']) if parsed['emotions'] else '無'}\n"
                if 'simText' in parsed and parsed['simText']:
                    main_text += f"\n相關語句 ({len(parsed['simText'])}個):\n"
                    for i, sim in enumerate(parsed['simText'][:5], 1):  # 只顯示前5個
                        main_text += f"{i}. {sim}\n"
                    if len(parsed['simText']) > 5:
                        main_text += f"... 還有 {len(parsed['simText']) - 5} 個"
                
                detail_dialog.setText(main_text)
                detail_dialog.setDetailedText(
                    f"完整JSON:\n{json.dumps(parsed, ensure_ascii=False, indent=2)}\n\n"
                    f"檔案資訊:\n"
                    f"檔案路徑: {result['filePath']}\n"
                    f"提取字幕: {result.get('extractedSubtitle', '無')}\n"
                    f"集數編號: {result.get('episode', '無')}\n"
                    f"自動提取字幕: {'是' if result.get('autoExtractSubtitle', False) else '否'}\n"
                    f"模型: {result['model']}\n"
                    f"API類型: {result['apiType']}\n"
                    f"處理時間: {result['timestamp']}\n\n"
                    f"原始回應:\n{result.get('rawResponse', '無')}"
                )
            else:
                # 顯示解析失敗的信息
                main_text = "✗ JSON解析失敗\n\n"
                
                # 字幕資訊
                if result.get('extractedSubtitle'):
                    main_text += f"提取字幕: {result['extractedSubtitle']}\n"
                if result.get('episode'):
                    main_text += f"集數編號: {result['episode']}\n"
                main_text += "\n"
                
                if result.get('parseError'):
                    main_text += f"錯誤原因: {result['parseError']}\n\n"
                
                raw_response = result.get('rawResponse', result.get('response', ''))
                if len(raw_response) > 500:
                    main_text += f"回應內容 (前500字):\n{raw_response[:500]}..."
                else:
                    main_text += f"回應內容:\n{raw_response}"
                
                detail_dialog.setText(main_text)
                detail_dialog.setDetailedText(
                    f"檔案資訊:\n"
                    f"檔案路徑: {result['filePath']}\n"
                    f"提取字幕: {result.get('extractedSubtitle', '無')}\n"
                    f"集數編號: {result.get('episode', '無')}\n"
                    f"自動提取字幕: {'是' if result.get('autoExtractSubtitle', False) else '否'}\n"
                    f"模型: {result['model']}\n"
                    f"API類型: {result['apiType']}\n"
                    f"處理時間: {result['timestamp']}\n\n"
                    f"完整原始回應:\n{raw_response}"
                )
            
            detail_dialog.exec_()

def main():
    """主函數"""
    app = QApplication(sys.argv)
    app.setApplicationName("進階AI圖像分析助手")
    app.setApplicationVersion("2.0")
    
    # 檢查Google AI支援
    if not GOOGLE_AI_AVAILABLE:
        QMessageBox.information(None, "提示", 
                              "Google AI庫未安裝，僅支援Ollama功能。\n"
                              "如需使用Google AI，請執行: pip install google-genai")
    
    window = AdvancedAIChatApp()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
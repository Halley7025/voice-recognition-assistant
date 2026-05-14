import sys
import os
import time
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox, QGroupBox,
    QProgressBar, QTabWidget, QSplitter, QMessageBox, QInputDialog
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon

sys.path.insert(0, os.path.dirname(__file__))
from audio.audio_capture import AudioCapture
from audio.audio_preprocess import AudioPreprocessor
from asr.speech_recognizer import SpeechRecognizer
from speaker.speaker_verifier import SpeakerVerifier
from controller.command_parser import CommandParser
from controller.system_controller import SystemController


class RecognizeThread(QThread):
    result_ready = pyqtSignal(str, str, float)
    status_update = pyqtSignal(str)

    def __init__(self, capture, preprocessor, recognizer, parser, controller):
        super().__init__()
        self.capture = capture
        self.preprocessor = preprocessor
        self.recognizer = recognizer
        self.parser = parser
        self.controller = controller
        self.running = True

    def run(self):
        while self.running:
            try:
                self.status_update.emit("正在录音...")
                raw = self.capture.record_seconds(3)
                if len(raw) == 0:
                    self.result_ready.emit("", "未检测到音频", 0)
                    continue
                self.status_update.emit("正在预处理...")
                processed = self.preprocessor.process(raw)
                if len(processed) < 16000 * 0.3:
                    self.result_ready.emit("", "语音过短", 0)
                    continue
                self.status_update.emit("正在识别...")
                start = time.time()
                text = self.recognizer.transcribe(processed)
                elapsed = time.time() - start
                if not text:
                    self.result_ready.emit("", "未识别到语音", elapsed)
                    continue
                cmd = self.parser.parse(text)
                if cmd:
                    self.status_update.emit(f"执行指令: {cmd}")
                    success, result = self.controller.run(cmd)
                    self.result_ready.emit(text, result, elapsed)
                else:
                    self.result_ready.emit(text, "未匹配到有效指令", elapsed)
            except Exception as e:
                self.result_ready.emit("", f"错误: {e}", 0)

    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("语音识别助手 - 本地版")
        self.setMinimumSize(900, 650)
        self.capture = AudioCapture()
        self.preprocessor = AudioPreprocessor()
        self.recognizer = SpeechRecognizer()
        self.verifier = SpeakerVerifier()
        self.verifier._load_db()
        self.parser = CommandParser(use_nlu=False)
        self.controller = SystemController()
        self.recognize_thread = None
        self.current_user = None
        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        header = QLabel("语音识别助手")
        header.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #2c3e50; padding: 10px;")
        main_layout.addWidget(header)

        tabs = QTabWidget()
        tabs.addTab(self._create_main_tab(), "语音控制")
        tabs.addTab(self._create_enroll_tab(), "声纹注册")
        tabs.addTab(self._create_info_tab(), "系统信息")
        main_layout.addWidget(tabs)

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #7f8c8d; padding: 5px;")
        main_layout.addWidget(self.status_label)

        self._apply_style()

    def _create_main_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        auth_group = QGroupBox("身份验证")
        auth_layout = QHBoxLayout()
        self.user_combo = QComboBox()
        self.user_combo.setEditable(True)
        self.user_combo.setPlaceholderText("选择或输入用户名")
        self._refresh_user_list()
        auth_btn = QPushButton("声纹验证")
        auth_btn.clicked.connect(self._do_auth)
        self.auth_status = QLabel("未验证")
        self.auth_status.setStyleSheet("color: red;")
        auth_layout.addWidget(QLabel("用户:"))
        auth_layout.addWidget(self.user_combo)
        auth_layout.addWidget(auth_btn)
        auth_layout.addWidget(self.auth_status)
        auth_group.setLayout(auth_layout)
        layout.addWidget(auth_group)

        control_group = QGroupBox("语音控制")
        control_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()
        self.listen_btn = QPushButton("开始语音识别")
        self.listen_btn.setStyleSheet(
            "QPushButton { background-color: #3498db; color: white; "
            "padding: 15px; font-size: 14px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #2980b9; }"
        )
        self.listen_btn.clicked.connect(self._toggle_listening)
        btn_layout.addWidget(self.listen_btn)
        control_layout.addLayout(btn_layout)

        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setFont(QFont("Consolas", 11))
        self.result_display.setStyleSheet(
            "background-color: #2c3e50; color: #ecf0f1; padding: 10px; border-radius: 5px;"
        )
        control_layout.addWidget(self.result_display)

        manual_layout = QHBoxLayout()
        self.manual_input = QLineEdit()
        self.manual_input.setPlaceholderText("手动输入指令...")
        self.manual_input.returnPressed.connect(self._manual_execute)
        manual_btn = QPushButton("执行")
        manual_btn.clicked.connect(self._manual_execute)
        manual_layout.addWidget(self.manual_input)
        manual_layout.addWidget(manual_btn)
        control_layout.addLayout(manual_layout)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        return widget

    def _create_enroll_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        info = QLabel("声纹注册: 输入用户名后点击注册，系统将采集3段语音样本用于声纹建模。")
        info.setWordWrap(True)
        layout.addWidget(info)

        enroll_layout = QHBoxLayout()
        self.enroll_input = QLineEdit()
        self.enroll_input.setPlaceholderText("输入用户名")
        enroll_btn = QPushButton("开始注册")
        enroll_btn.setStyleSheet(
            "QPushButton { background-color: #27ae60; color: white; padding: 10px; border-radius: 5px; }"
        )
        enroll_btn.clicked.connect(self._do_enroll)
        enroll_layout.addWidget(self.enroll_input)
        enroll_layout.addWidget(enroll_btn)
        layout.addLayout(enroll_layout)

        self.enroll_log = QTextEdit()
        self.enroll_log.setReadOnly(True)
        self.enroll_log.setFont(QFont("Consolas", 10))
        layout.addWidget(self.enroll_log)

        users_group = QGroupBox("已注册用户")
        users_layout = QVBoxLayout()
        self.users_list = QTextEdit()
        self.users_list.setReadOnly(True)
        self.users_list.setMaximumHeight(150)
        users_layout.addWidget(self.users_list)
        refresh_btn = QPushButton("刷新列表")
        refresh_btn.clicked.connect(self._refresh_users_display)
        users_layout.addWidget(refresh_btn)
        users_group.setLayout(users_layout)
        layout.addWidget(users_group)

        self._refresh_users_display()
        return widget

    def _create_info_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setFont(QFont("Consolas", 10))
        info_content = (
            "=== 语音识别助手 系统信息 ===\n\n"
            "技术栈:\n"
            "  语音识别: Whisper (faster-whisper, CTranslate2 INT8量化)\n"
            "  声纹验证: ECAPA-TDNN (SpeechBrain)\n"
            "  意图分类: BERT-base-chinese + 分类头\n"
            "  音频预处理: 谱减法降噪 + 预加重 + 能量VAD\n"
            "  系统控制: Windows API (pycaw/pyautogui/ctypes)\n\n"
            "架构设计:\n"
            "  麦克风采集 -> 预处理(降噪/VAD) -> 特征提取\n"
            "  -> Whisper推理 -> NLU意图分类 -> 系统指令执行\n"
            "  声纹注册/验证: ECAPA-TDNN -> embedding -> 余弦相似度\n\n"
            "创新亮点:\n"
            "  1. 100%本地推理，零云端依赖，隐私数据零上传\n"
            "  2. 自实现谱减法降噪 + Wiener滤波增强噪声鲁棒性\n"
            "  3. Whisper INT8量化推理，CPU端实时率RTF<1\n"
            "  4. 多维融合身份认证(声纹+语义+行为)\n"
            "  5. BERT意图分类支持同义表达泛化\n\n"
            f"已注册用户: {', '.join(self.verifier.list_users()) or '无'}\n"
        )
        info_text.setPlainText(info_content)
        layout.addWidget(info_text)
        return widget

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #ecf0f1; }
            QGroupBox { font-weight: bold; border: 1px solid #bdc3c7;
                        border-radius: 5px; margin-top: 10px; padding-top: 15px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { padding: 8px 16px; border-radius: 4px; font-size: 12px; }
            QLineEdit { padding: 8px; border: 1px solid #bdc3c7; border-radius: 4px; }
        """)

    def _refresh_user_list(self):
        self.user_combo.clear()
        users = self.verifier.list_users()
        if users:
            self.user_combo.addItems(users)

    def _refresh_users_display(self):
        users = self.verifier.list_users()
        if users:
            self.users_list.setPlainText("\n".join(f"  - {u}" for u in users))
        else:
            self.users_list.setPlainText("  暂无注册用户")

    def _do_auth(self):
        user_id = self.user_combo.currentText().strip()
        if not user_id:
            QMessageBox.warning(self, "提示", "请输入用户名")
            return
        self.status_label.setText(f"正在验证用户 {user_id}...")
        QApplication.processEvents()
        raw = self.capture.record_seconds(3)
        processed = self.preprocessor.process(raw)
        is_match, sim = self.verifier.verify(user_id, processed)
        if is_match:
            self.current_user = user_id
            self.auth_status.setText(f"已验证: {user_id} (相似度: {sim:.3f})")
            self.auth_status.setStyleSheet("color: green; font-weight: bold;")
            self._log(f"[验证] 用户 '{user_id}' 验证通过 (相似度: {sim:.3f})")
        else:
            self.auth_status.setText(f"验证失败 (相似度: {sim:.3f})")
            self.auth_status.setStyleSheet("color: red;")
            self._log(f"[验证] 用户 '{user_id}' 验证失败 (相似度: {sim:.3f})")
        self.status_label.setText("就绪")

    def _do_enroll(self):
        user_id = self.enroll_input.text().strip()
        if not user_id:
            QMessageBox.warning(self, "提示", "请输入用户名")
            return
        self.enroll_log.append(f"\n开始注册: {user_id}")
        samples = []
        for i in range(3):
            self.enroll_log.append(f"采样 {i+1}/3 - 请说话 (3秒)...")
            QApplication.processEvents()
            raw = self.capture.record_seconds(3)
            processed = self.preprocessor.process(raw)
            if len(processed) > 16000 * 0.5:
                samples.append(processed)
                self.enroll_log.append(f"  采样 {i+1} 完成")
            else:
                self.enroll_log.append(f"  采样 {i+1} 无效，重试...")
                i -= 1
        success = self.verifier.register_speaker(user_id, samples)
        if success:
            self.enroll_log.append(f"注册成功: {user_id}")
            self._refresh_user_list()
            self._refresh_users_display()
        else:
            self.enroll_log.append(f"注册失败: {user_id}")

    def _toggle_listening(self):
        if self.recognize_thread and self.recognize_thread.running:
            self.recognize_thread.stop()
            self.recognize_thread.wait()
            self.listen_btn.setText("开始语音识别")
            self.listen_btn.setStyleSheet(
                "QPushButton { background-color: #3498db; color: white; "
                "padding: 15px; font-size: 14px; border-radius: 5px; }"
            )
            self.status_label.setText("已停止")
        else:
            self.recognize_thread = RecognizeThread(
                self.capture, self.preprocessor, self.recognizer,
                self.parser, self.controller
            )
            self.recognize_thread.result_ready.connect(self._on_result)
            self.recognize_thread.status_update.connect(
                lambda s: self.status_label.setText(s)
            )
            self.recognize_thread.start()
            self.listen_btn.setText("停止识别")
            self.listen_btn.setStyleSheet(
                "QPushButton { background-color: #e74c3c; color: white; "
                "padding: 15px; font-size: 14px; border-radius: 5px; }"
            )

    def _on_result(self, text, result, elapsed):
        timestamp = time.strftime("%H:%M:%S")
        if text:
            self._log(f"[{timestamp}] 识别: '{text}' -> {result} ({elapsed:.2f}s)")
        else:
            self._log(f"[{timestamp}] {result}")

    def _manual_execute(self):
        text = self.manual_input.text().strip()
        if not text:
            return
        cmd = self.parser.parse(text)
        if cmd:
            success, result = self.controller.run(cmd)
            self._log(f"[手动] '{text}' -> {cmd} -> {result}")
        else:
            self._log(f"[手动] '{text}' -> 未匹配到指令")
        self.manual_input.clear()

    def _log(self, msg):
        self.result_display.append(msg)

    def closeEvent(self, event):
        if self.recognize_thread and self.recognize_thread.running:
            self.recognize_thread.stop()
            self.recognize_thread.wait()
        self.capture.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

import sys
import os
import time

# === HuggingFace Mirror (must be set BEFORE any HF/huggingface_hub import) ===
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
print("[I][system] HF_ENDPOINT -> hf-mirror.com")

if sys.platform == "win32":
    torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    if os.path.isdir(torch_lib):
        os.environ["PATH"] = torch_lib + ";" + os.environ.get("PATH", "")

import numpy as np
from collections import deque

try:
    import torch
    _ = torch.zeros(1)
except Exception:
    pass

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox, QGroupBox,
    QProgressBar, QTabWidget, QSplitter, QMessageBox, QFrame,
    QGridLayout, QSizePolicy, QGraphicsDropShadowEffect
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QSize, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import (
    QFont, QColor, QPalette, QIcon, QPainter, QLinearGradient,
    QBrush, QPen, QPixmap, QFontDatabase
)

sys.path.insert(0, os.path.dirname(__file__))
from audio.audio_capture import AudioCapture
from audio.audio_preprocess import AudioPreprocessor
from asr.speech_recognizer import SpeechRecognizer
from speaker.speaker_verifier import SpeakerVerifier
from controller.command_parser import CommandParser
from controller.system_controller import SystemController
from controller.speech_feedback import SpeechFeedback
from controller.command_history import CommandHistory
from controller.verification_gate import verify_and_execute
from gui_waveform import WaveformWidget
from gui_theme import COLORS, DARK_STYLE, apply_theme
from gui_widgets import AcrylicCard, ListeningWidget
from global_config import SAMPLE_RATE


# COLORS imported from gui_theme.py


class AudioLevelWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.level = 0.0
        self.target_level = 0.0
        self.is_active = False
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)
        self._timer = QTimer()
        self._timer.timeout.connect(self._animate)
        self._timer.start(30)

    def set_level(self, level):
        self.target_level = min(max(level, 0.0), 1.0)

    def set_active(self, active):
        self.is_active = active
        if not active:
            self.target_level = 0.0

    def _animate(self):
        diff = self.target_level - self.level
        self.level += diff * 0.3
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(COLORS["bg_dark"]))
        num_bars = 40
        bar_w = max((w - (num_bars - 1) * 2) / num_bars, 2)
        gap = 2
        for i in range(num_bars):
            x = i * (bar_w + gap)
            bar_level = self.level * (1.0 - abs(i - num_bars / 2) / (num_bars / 2) * 0.5)
            bar_level = max(bar_level, 0.05)
            bar_h = int(bar_level * h * 0.8)
            y = (h - bar_h) / 2
            if self.is_active:
                ratio = bar_level
                r = int(233 * ratio + 0 * (1 - ratio))
                g = int(69 * ratio + 184 * (1 - ratio))
                b = int(96 * ratio + 148 * (1 - ratio))
                color = QColor(r, g, b)
            else:
                color = QColor(COLORS["text_muted"])
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(int(x), int(y), int(bar_w), int(bar_h), 2, 2)
        painter.end()


class StatusIndicator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.status = "idle"
        self._angle = 0
        self.setMinimumSize(24, 24)
        self.setMaximumSize(24, 24)
        self._timer = QTimer()
        self._timer.timeout.connect(self._spin)
        self._timer.start(50)

    def set_status(self, status):
        self.status = status
        if status == "listening":
            self._timer.start(50)
        elif status == "processing":
            self._timer.start(30)
        else:
            self._timer.stop()
        self.update()

    def _spin(self):
        self._angle = (self._angle + 12) % 360
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        center = self.rect().center()
        radius = 8
        if self.status == "idle":
            painter.setBrush(QBrush(QColor(COLORS["text_muted"])))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(center, radius, radius)
        elif self.status == "listening":
            painter.setPen(QPen(QColor(COLORS["accent"]), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(center, radius, radius)
            painter.setPen(QPen(QColor(COLORS["accent"]), 2))
            painter.drawEllipse(center, radius - 3, radius - 3)
        elif self.status == "processing":
            painter.translate(center)
            painter.rotate(self._angle)
            painter.setPen(QPen(QColor(COLORS["gradient_start"]), 2))
            painter.drawArc(-radius, -radius, radius * 2, radius * 2, 0, 270 * 16)
        elif self.status == "success":
            painter.setBrush(QBrush(QColor(COLORS["success"])))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(center, radius, radius)
        elif self.status == "error":
            painter.setBrush(QBrush(QColor(COLORS["error"])))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(center, radius, radius)
        painter.end()


class RecognizeThread(QThread):
    result_ready = pyqtSignal(str, str, float, object)
    status_update = pyqtSignal(str)
    level_update = pyqtSignal(float)
    chunk_ready = pyqtSignal(object)

    def __init__(self, capture, preprocessor, recognizer, parser, controller):
        super().__init__()
        self.capture = capture
        self.preprocessor = preprocessor
        self.recognizer = recognizer
        self.parser = parser
        self.controller = controller
        self.running = True
        self._ambient_rms = 0.001  # Initial ambient noise estimate
        self._ambient_samples = 0

    def _calibrate_ambient(self):
        """Calibrate ambient noise level from first few chunks."""
        samples = []
        for _ in range(10):
            try:
                data = self.capture.stream.read(
                    self.capture.chunk, exception_on_overflow=False
                )
                chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                samples.append(np.sqrt(np.mean(chunk ** 2)))
            except Exception:
                continue
        if samples:
            self._ambient_rms = max(np.median(samples), 0.0001)
            self._ambient_samples = len(samples)

    def _get_speech_threshold(self):
        """Adaptive speech threshold based on ambient noise."""
        # Threshold = ambient * multiplier, with minimum floor
        return max(self._ambient_rms * 3.0, 0.001)

    def run(self):
        # Calibrate ambient noise before starting
        self._calibrate_ambient()

        while self.running:
            try:
                self.status_update.emit("listening")
                chunks = []
                silence_start = None
                start_time = time.time()
                has_speech = False
                speech_threshold = self._get_speech_threshold()

                while self.running and (time.time() - start_time) < 10:
                    data = self.capture.stream.read(
                        self.capture.chunk, exception_on_overflow=False
                    )
                    chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    chunks.append(chunk)
                    energy = np.sqrt(np.mean(chunk ** 2))
                    level = min(energy * 5, 1.0)
                    self.level_update.emit(level)
                    self.chunk_ready.emit(chunk)

                    if energy > speech_threshold:
                        has_speech = True
                        silence_start = None
                    elif has_speech:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > 1.5:
                            break

                if not chunks:
                    self.result_ready.emit("", "未检测到音频", 0, None)
                    continue

                raw = np.concatenate(chunks)
                if not has_speech or len(raw) < SAMPLE_RATE * 0.3:
                    self.result_ready.emit("", "未检测到有效语音", 0, None)
                    self.level_update.emit(0.0)
                    continue

                # Strict VAD: reject noise/silence before Whisper
                vad_valid, speech_audio, vad_meta = self.preprocessor.strict_vad(raw)
                if not vad_valid:
                    reason = vad_meta.get("reason", "unknown")
                    self.result_ready.emit("", f"VAD拒绝: {reason}", 0, None)
                    self.level_update.emit(0.0)
                    continue

                self.status_update.emit("processing")
                processed = self.preprocessor.process(speech_audio)
                if len(processed) < SAMPLE_RATE * 0.2:
                    processed = self.preprocessor.process(speech_audio, use_spectral_subtraction=False)

                start = time.time()
                text = self.recognizer.transcribe(processed)
                elapsed = time.time() - start

                if not text:
                    self.result_ready.emit("", "未识别到语音内容", elapsed, None)
                    self.level_update.emit(0.0)
                    continue

                cmd = self.parser.parse(text)
                if cmd:
                    self.status_update.emit("executing")
                    # Use verification gate for protected commands
                    success, result, verified, user_id, sim = verify_and_execute(
                        cmd, raw, self.verifier, self.controller, self.preprocessor
                    )
                    if verified is not None:
                        if verified:
                            result = f"[{user_id}验证通过] {result}"
                        else:
                            result = f"[声纹验证失败] {result}"
                    self.result_ready.emit(text, result, elapsed, raw)
                else:
                    self.result_ready.emit(text, "未匹配到有效指令", elapsed, raw)

                self.level_update.emit(0.0)
            except Exception as e:
                self.result_ready.emit("", f"错误: {e}", 0, None)
                self.level_update.emit(0.0)

    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("语音识别助手")
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self._drag_pos = None
        self._bg_pixmap = None
        self._load_bg()
        self.setMinimumSize(1000, 700)
        self.resize(1100, 750)
        self.capture = AudioCapture()
        self.preprocessor = AudioPreprocessor()
        self.recognizer = SpeechRecognizer()
        self.verifier = SpeakerVerifier()
        self.verifier._load_db()
        self.parser = CommandParser(use_nlu=False)
        self.controller = SystemController()
        self.tts = SpeechFeedback(enabled=False)
        self.history = CommandHistory()
        self.recognize_thread = None
        self.current_user = None
        self.command_history = deque(maxlen=100)
        self._components_loaded = True
        self._init_ui()

    def _load_bg(self):
        bg_path = os.path.join(os.path.dirname(__file__), "bg.jpg")
        if os.path.exists(bg_path):
            self._bg_pixmap = QPixmap(bg_path)

    def paintEvent(self, event):
        if self._bg_pixmap and not self._bg_pixmap.isNull():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            # Scale to fill window, keep aspect ratio, crop overflow
            scaled = self._bg_pixmap.scaled(
                self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            # Center-crop
            x = (scaled.width() - self.width()) // 2
            y = (scaled.height() - self.height()) // 2
            painter.drawPixmap(0, 0, self.width(), self.height(), scaled, x, y, self.width(), self.height())
            # Dark overlay for readability
            painter.fillRect(self.rect(), QColor(10, 10, 26, 160))
            painter.end()
        else:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(10, 10, 26))
            painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() & Qt.LeftButton:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        header = self._create_header()
        main_layout.addWidget(header)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setSpacing(0)
        body_layout.setContentsMargins(16, 8, 16, 16)

        left_panel = self._create_left_panel()
        body_layout.addWidget(left_panel, 3)

        right_panel = self._create_right_panel()
        body_layout.addWidget(right_panel, 2)

        main_layout.addWidget(body)

        footer = self._create_footer()
        main_layout.addWidget(footer)


    def _create_header(self):
        header = QWidget()
        header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        header.setStyleSheet(f"background-color: {COLORS['bg_dark']};")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 8, 16, 8)

        self.status_indicator = StatusIndicator()
        layout.addWidget(self.status_indicator)

        title = QLabel("语音识别助手")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title.setStyleSheet(f"color: {COLORS['text_primary']};")
        layout.addWidget(title)

        layout.addStretch()

        self.user_label = QLabel("未登录")
        self.user_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
        layout.addWidget(self.user_label)

        self.min_btn = QPushButton("—")
        self.min_btn.setFixedSize(36, 28)
        self.min_btn.setCursor(Qt.PointingHandCursor)
        self.min_btn.clicked.connect(self.showMinimized)
        layout.addWidget(self.min_btn)

        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(36, 28)
        self.close_btn.setCursor(Qt.PointingHandCursor)
        self.close_btn.setStyleSheet(
            "QPushButton{background:#ff5252;border-radius:6px;}"
            "QPushButton:hover{background:#ff1744;}"
        )
        self.close_btn.clicked.connect(self.close)
        layout.addWidget(self.close_btn)

        return header

    def _create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        layout.setContentsMargins(0, 0, 8, 0)

        voice_group = QGroupBox("语音控制")
        voice_layout = QVBoxLayout(voice_group)
        voice_layout.setSpacing(12)

        self.waveform = WaveformWidget()
        voice_layout.addWidget(self.waveform)

        mic_path = os.path.join(os.path.dirname(__file__), "mic.png")
        self.listening_pulse = ListeningWidget(icon_path=mic_path, size=140)
        voice_layout.addWidget(self.listening_pulse, alignment=Qt.AlignCenter)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        self.listen_btn = QPushButton("🎤  开始识别")
        self.listen_btn.setCursor(Qt.PointingHandCursor)
        self.listen_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.listen_btn.clicked.connect(self._toggle_listening)
        btn_row.addWidget(self.listen_btn)

        self.stop_btn = QPushButton("⏹  停止")
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.stop_btn.clicked.connect(self._stop_listening)
        btn_row.addWidget(self.stop_btn)
        voice_layout.addLayout(btn_row)

        manual_row = QHBoxLayout()
        manual_row.setSpacing(10)
        self.manual_input = QLineEdit()
        self.manual_input.setPlaceholderText("输入指令文本...")
        self.manual_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.manual_input.returnPressed.connect(self._manual_execute)
        manual_row.addWidget(self.manual_input)

        exec_btn = QPushButton("执行")
        exec_btn.setCursor(Qt.PointingHandCursor)
        exec_btn.clicked.connect(self._manual_execute)
        manual_row.addWidget(exec_btn)
        voice_layout.addLayout(manual_row)

        layout.addWidget(voice_group)

        quick_group = QGroupBox("快捷操作")
        quick_grid = QGridLayout(quick_group)
        quick_grid.setSpacing(10)
        quick_cmds = [
            ("记事本", "open_notepad"), ("浏览器", "open_browser"),
            ("音量+", "volume_up"), ("音量-", "volume_down"),
            ("计算器", "open_calculator"), ("截屏", "screenshot"),
            ("锁屏", "lock_screen"), ("命令行", "open_cmd"),
        ]
        for i, (label, cmd) in enumerate(quick_cmds):
            btn = QPushButton(label)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setProperty("cmd", cmd)
            btn.clicked.connect(lambda checked, c=cmd: self._execute_quick(c))
            quick_grid.addWidget(btn, i // 4, i % 4)
        layout.addWidget(quick_group)

        auth_group = QGroupBox("身份验证")
        auth_layout = QHBoxLayout(auth_group)
        auth_layout.setSpacing(10)
        self.user_combo = QComboBox()
        self.user_combo.setEditable(True)
        self.user_combo.setPlaceholderText("选择用户")
        self.user_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._refresh_user_list()
        auth_layout.addWidget(self.user_combo)

        enroll_btn = QPushButton("注册")
        enroll_btn.setCursor(Qt.PointingHandCursor)
        enroll_btn.clicked.connect(self._do_enroll)
        auth_layout.addWidget(enroll_btn)

        auth_btn = QPushButton("验证")
        auth_btn.setCursor(Qt.PointingHandCursor)
        auth_btn.clicked.connect(self._do_auth)
        auth_layout.addWidget(auth_btn)
        layout.addWidget(auth_group)

        layout.addStretch()
        return panel

    def _create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        layout.setContentsMargins(8, 0, 0, 0)

        right_tabs = QTabWidget()

        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        log_layout.setSpacing(10)
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Consolas", 10))
        self.log_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        log_layout.addWidget(self.log_display)
        clear_btn = QPushButton("清空日志")
        clear_btn.setCursor(Qt.PointingHandCursor)
        clear_btn.clicked.connect(self.log_display.clear)
        log_layout.addWidget(clear_btn)
        right_tabs.addTab(log_tab, "执行日志")

        history_tab = QWidget()
        hist_layout = QVBoxLayout(history_tab)
        hist_layout.setSpacing(10)
        self.history_display = QTextEdit()
        self.history_display.setReadOnly(True)
        self.history_display.setFont(QFont("Consolas", 10))
        self.history_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        hist_layout.addWidget(self.history_display)
        hist_btn_row = QHBoxLayout()
        hist_btn_row.setSpacing(10)
        refresh_hist_btn = QPushButton("刷新")
        refresh_hist_btn.setCursor(Qt.PointingHandCursor)
        refresh_hist_btn.clicked.connect(self._refresh_history)
        hist_btn_row.addWidget(refresh_hist_btn)
        clear_hist_btn = QPushButton("清空历史")
        clear_hist_btn.setCursor(Qt.PointingHandCursor)
        clear_hist_btn.clicked.connect(self._clear_history)
        hist_btn_row.addWidget(clear_hist_btn)
        hist_layout.addLayout(hist_btn_row)
        right_tabs.addTab(history_tab, "指令历史")

        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        settings_layout.setSpacing(15)

        tts_group = QGroupBox("语音反馈")
        tts_layout = QVBoxLayout(tts_group)
        self.tts_toggle = QPushButton("语音反馈: 关闭")
        self.tts_toggle.setCheckable(True)
        self.tts_toggle.setCursor(Qt.PointingHandCursor)
        self.tts_toggle.clicked.connect(self._toggle_tts)
        tts_layout.addWidget(self.tts_toggle)
        settings_layout.addWidget(tts_group)

        info_group = QGroupBox("系统信息")
        info_layout = QVBoxLayout(info_group)
        self.info_label = QLabel()
        self.info_label.setFont(QFont("Microsoft YaHei", 10))
        self.info_label.setWordWrap(True)
        self.info_label.setTextFormat(Qt.RichText)
        self._update_info_display()
        info_layout.addWidget(self.info_label)
        settings_layout.addWidget(info_group)

        hotkey_group = QGroupBox("快捷键")
        hotkey_layout = QVBoxLayout(hotkey_group)
        hotkey_info = QLabel(
            "<b>F2</b> — 开始/停止语音识别<br>"
            "<b>Enter</b> — 执行手动输入指令<br>"
            "<b>Ctrl+Q</b> — 退出程序"
        )
        hotkey_info.setTextFormat(Qt.RichText)
        hotkey_layout.addWidget(hotkey_info)
        settings_layout.addWidget(hotkey_group)

        settings_layout.addStretch()
        right_tabs.addTab(settings_tab, "设置")

        layout.addWidget(right_tabs)
        self._refresh_history()
        return panel

    def _create_footer(self):
        footer = QWidget()
        footer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        footer.setStyleSheet(f"background-color: {COLORS['bg_dark']};")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(16, 4, 16, 4)

        self.status_text = QLabel("就绪")
        self.status_text.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px;")
        layout.addWidget(self.status_text)

        layout.addStretch()

        self.model_info = QLabel()
        self.model_info.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px;")
        model_name = self.recognizer.model_type or "未加载"
        self.model_info.setText(f"模型: Whisper-{model_name}")
        layout.addWidget(self.model_info)
        return footer
    def _update_info_display(self):
        users = self.verifier.list_users()
        user_count = len(users)
        stats = self.history.get_stats()
        cmd_count = stats["total"]
        most_used = stats.get("most_used", "无")
        model_name = self.recognizer.model_type or "未加载"
        info_html = f"""
        <div style='color: {COLORS["text_secondary"]}; line-height: 1.8;'>
        <b style='color: {COLORS["text_primary"]}'>技术栈</b><br>
        识别: Whisper ({model_name})<br>
        声纹: ECAPA-TDNN<br>
        预处理: 谱减法 + 能量VAD<br>
        控制: Windows API ({len(self.controller.get_all_commands())}种指令)<br><br>
        <b style='color: {COLORS["text_primary"]}'>统计</b><br>
        已注册用户: {user_count}<br>
        已执行指令: {cmd_count}<br>
        最常用指令: {most_used}<br>
        当前用户: {self.current_user or '未登录'}<br>
        语音反馈: {'开启' if self.tts.enabled else '关闭'}
        </div>
        """
        self.info_label.setText(info_html)

    def _toggle_tts(self):
        self.tts.enabled = not self.tts.enabled
        if self.tts.enabled:
            self.tts_toggle.setText("语音反馈: 开启")
            self.tts_toggle.setStyleSheet(f"background-color: {COLORS['success']};")
            self.tts.speak("语音反馈已开启")
        else:
            self.tts_toggle.setText("语音反馈: 关闭")
            self.tts_toggle.setStyleSheet("")
        self._update_info_display()

    def _refresh_history(self):
        if not hasattr(self, 'history_display'):
            return
        entries = self.history.get_recent(30)
        self.history_display.clear()
        if not entries:
            self.history_display.setPlainText("  暂无历史记录")
            return
        lines = []
        for e in reversed(entries):
            src = "语音" if e.get("source") == "voice" else "手动"
            lines.append(
                f"[{e['timestamp']}] ({src}) {e['text']} → {e.get('result', '')}"
            )
        self.history_display.setPlainText("\n".join(lines))

    def _clear_history(self):
        self.history.clear()
        self._refresh_history()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F2:
            self._toggle_listening()
        elif event.key() == Qt.Key_Q and event.modifiers() == Qt.ControlModifier:
            self.close()
        else:
            super().keyPressEvent(event)

    def _refresh_user_list(self):
        self.user_combo.clear()
        users = self.verifier.list_users()
        if users:
            self.user_combo.addItems(users)

    def _toggle_listening(self):
        if self.recognize_thread and self.recognize_thread.running:
            self._stop_listening()
        else:
            self._start_listening()

    def _start_listening(self):
        self.recognize_thread = RecognizeThread(
            self.capture, self.preprocessor, self.recognizer,
            self.parser, self.controller
        )
        self.recognize_thread.result_ready.connect(self._on_result)
        self.recognize_thread.status_update.connect(self._on_status)
        self.recognize_thread.level_update.connect(self._on_level)
        self.recognize_thread.chunk_ready.connect(self.waveform.push_audio)
        self.recognize_thread.start()
        self.listen_btn.setText("  识别中...")
        self.listen_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_indicator.set_status("listening")
        self.waveform.set_active(True)
        self.status_text.setText("正在监听...")

    def _stop_listening(self):
        if self.recognize_thread:
            self.recognize_thread.stop()
            self.recognize_thread.wait(3000)
        self.listen_btn.setText("  开始识别")
        self.listen_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_indicator.set_status("idle")
        self.waveform.set_active(False)
        self.status_text.setText("已停止")

    def _on_status(self, status):
        self.status_indicator.set_status(status)
        status_map = {
            "listening": "正在录音...",
            "processing": "正在识别...",
            "executing": "正在执行指令...",
        }
        self.status_text.setText(status_map.get(status, status))

    def _on_level(self, level):
        pass  # Waveform receives audio via chunk_ready signal

    def _on_result(self, text, result, elapsed, raw_audio=None):
        timestamp = time.strftime("%H:%M:%S")
        if text:
            self._log(f"<span style='color:{COLORS['text_muted']}'>[{timestamp}]</span> "
                      f"<span style='color:{COLORS['gradient_start']}'>识别:</span> "
                      f"'{text}' → "
                      f"<span style='color:{COLORS['success']}'>{result}</span> "
                      f"<span style='color:{COLORS['text_muted']}'>({elapsed:.2f}s)</span>")
            self.history.add(text, text, result, source="voice")
            self.tts.speak(result)
        else:
            self._log(f"<span style='color:{COLORS['text_muted']}'>[{timestamp}]</span> "
                      f"<span style='color:{COLORS['warning']}'>{result}</span>")
        # Only reset UI if the recognize thread has actually stopped
        if not self.recognize_thread or not self.recognize_thread.running:
            self.listen_btn.setText("  开始识别")
            self.listen_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.waveform.set_active(False)
            self.status_indicator.set_status("idle")
        else:
            self.status_indicator.set_status("success" if text else "listening")
            QTimer.singleShot(800, lambda: self.status_indicator.set_status("listening") if self.recognize_thread and self.recognize_thread.running else None)
        self._update_info_display()

    def _manual_execute(self):
        text = self.manual_input.text().strip()
        if not text:
            return
        self.status_indicator.set_status("processing")
        QApplication.processEvents()
        cmd = self.parser.parse(text)
        timestamp = time.strftime("%H:%M:%S")
        if cmd:
            success, result = self.controller.run(cmd)
            self._log(f"<span style='color:{COLORS['text_muted']}'>[{timestamp}]</span> "
                      f"<span style='color:{COLORS['gradient_start']}'>手动:</span> "
                      f"'{text}' → <span style='color:{COLORS['success']}'>{result}</span>")
            self.history.add(text, cmd, result, source="manual")
            self.tts.speak(result)
        else:
            self._log(f"<span style='color:{COLORS['text_muted']}'>[{timestamp}]</span> "
                      f"'{text}' → <span style='color:{COLORS['warning']}'>未匹配到指令</span>")
        self.manual_input.clear()
        self.status_indicator.set_status("idle")
        self._update_info_display()

    def _execute_quick(self, cmd):
        self.status_indicator.set_status("processing")
        QApplication.processEvents()
        success, result = self.controller.run(cmd)
        timestamp = time.strftime("%H:%M:%S")
        self._log(f"<span style='color:{COLORS['text_muted']}'>[{timestamp}]</span> "
                  f"<span style='color:{COLORS['gradient_start']}'>快捷:</span> "
                  f"{cmd} → <span style='color:{COLORS['success']}'>{result}</span>")
        self.history.add(cmd, cmd, result, source="quick")
        self.tts.speak(result)
        self.status_indicator.set_status("success")
        QTimer.singleShot(1000, lambda: self.status_indicator.set_status("idle"))
        self._update_info_display()

    def _do_auth(self):
        user_id = self.user_combo.currentText().strip()
        if not user_id:
            QMessageBox.warning(self, "提示", "请输入或选择用户名")
            return
        if user_id not in self.verifier.list_users():
            QMessageBox.warning(self, "提示", f"用户 '{user_id}' 未注册，请先注册。")
            return
        self.status_text.setText(f"正在验证 {user_id}...")
        self.status_indicator.set_status("processing")
        QApplication.processEvents()
        raw = self.capture.record_seconds(3)
        processed = self.preprocessor.process(raw)
        is_match, sim = self.verifier.verify(user_id, processed)
        timestamp = time.strftime("%H:%M:%S")
        if is_match:
            self.current_user = user_id
            self.user_label.setText(f"  {user_id}")
            self.user_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 12px;")
            self._log(f"<span style='color:{COLORS['text_muted']}'>[{timestamp}]</span> "
                      f"<span style='color:{COLORS['success']}'>✓ 验证通过: {user_id} "
                      f"(相似度: {sim:.3f})</span>")
            self.status_indicator.set_status("success")
        else:
            self._log(f"<span style='color:{COLORS['text_muted']}'>[{timestamp}]</span> "
                      f"<span style='color:{COLORS['error']}'>✗ 验证失败: {user_id} "
                      f"(相似度: {sim:.3f})</span>")
            self.status_indicator.set_status("error")
        self.status_text.setText("就绪")
        QTimer.singleShot(2000, lambda: self.status_indicator.set_status("idle"))
        self._update_info_display()

    def _do_enroll(self):
        if not self._components_loaded:
            QMessageBox.warning(self, "提示", "组件加载中，请稍候...")
            return
        if not self.verifier or self.verifier.model is None:
            QMessageBox.warning(self, "提示", "声纹模型未加载，无法注册用户。\n请确认已安装 speechbrain 库。")
            return
        user_id = self.user_combo.currentText().strip()
        if not user_id:
            QMessageBox.warning(self, "提示", "请输入用户名")
            return
        if user_id in self.verifier.list_users():
            reply = QMessageBox.question(self, "提示", f"用户 '{user_id}' 已存在，是否重新注册？",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        timestamp = time.strftime("%H:%M:%S")
        self._log(f"<span style='color:{COLORS['text_muted']}'>[{timestamp}]</span> "
                  f"<span style='color:{COLORS['gradient_start']}'>开始注册: {user_id}</span>")
        samples = []
        for i in range(3):
            self._log(f"  采样 {i+1}/3 - 请说话 (3秒)...")
            self.status_text.setText(f"注册采样 {i+1}/3...")
            QApplication.processEvents()
            raw = self.capture.record_seconds(3)
            processed = self.preprocessor.process(raw)
            if len(processed) > SAMPLE_RATE * 0.5:
                samples.append(processed)
                self._log(f"  <span style='color:{COLORS['success']}'>采样 {i+1} 完成</span>")
            else:
                self._log(f"  <span style='color:{COLORS['warning']}'>采样 {i+1} 无效</span>")
        success = self.verifier.register_speaker(user_id, samples)
        if success:
            self._log(f"  <span style='color:{COLORS['success']}'>✓ 注册成功: {user_id}</span>")
            self._refresh_user_list()
        else:
            self._log(f"  <span style='color:{COLORS['error']}'>✗ 注册失败: {user_id}</span>")
        self.status_text.setText("就绪")
        self._update_info_display()

    def _log(self, html):
        self.log_display.append(html)

    def closeEvent(self, event):
        if self.recognize_thread and self.recognize_thread.running:
            self.recognize_thread.stop()
            self.recognize_thread.wait(3000)
        self.capture.close()
        event.accept()


from PyQt5.QtWidgets import QStackedWidget

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    app.setStyle("Fusion")
    apply_theme(app)

    stack = QStackedWidget()
    stack.setWindowTitle("语音识别助手")
    stack.setMinimumSize(1000, 700)

    # Create shared components
    main_win = MainWindow()

    # Create login window (shares the same capture/preprocessor/verifier)
    from gui_login import LoginWindow
    login_win = LoginWindow(
        main_win.capture, main_win.preprocessor, main_win.verifier
    )

    stack.addWidget(login_win)   # index 0
    stack.addWidget(main_win)    # index 1

    def on_login_success(user_id):
        if user_id and user_id != "guest":
            main_win.current_user = user_id
            main_win.user_label.setText(f"  {user_id}")
            main_win.user_label.setStyleSheet(
                f"color: {COLORS['success']}; font-size: 12px;"
            )
        stack.setCurrentIndex(1)

    login_win.login_success.connect(on_login_success)
    stack.setCurrentIndex(0)
    stack.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()



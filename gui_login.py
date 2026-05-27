"""Login window with voiceprint verification."""
import os
import time
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPainter, QPixmap

from gui_theme import COLORS
from gui_widgets import ListeningWidget


class VerifyThread(QThread):
    status = pyqtSignal(str, str)
    level = pyqtSignal(float)

    def __init__(self, capture, preprocessor, verifier):
        super().__init__()
        self.capture = capture
        self.preprocessor = preprocessor
        self.verifier = verifier
        self.running = True

    def run(self):
        if not self.verifier or self.verifier.model is None:
            self.status.emit("error", "声纹模型未加载")
            return
        if not self.verifier.list_users():
            self.status.emit("error", "无已注册用户，请先注册")
            return

        self.status.emit("recording", "请说话 (3秒)...")
        chunks = []
        for _ in range(int(self.capture.sample_rate / self.capture.chunk * 3)):
            if not self.running:
                return
            try:
                data = self.capture.stream.read(
                    self.capture.chunk, exception_on_overflow=False
                )
                chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                chunks.append(chunk)
                energy = np.sqrt(np.mean(chunk ** 2))
                self.level.emit(min(energy * 5, 1.0))
            except Exception:
                continue

        raw = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        if len(raw) < 16000 * 0.5:
            self.status.emit("fail", "录音太短或无声音")
            return

        is_valid, speech_audio, vad_meta = self.preprocessor.strict_vad(raw)
        if not is_valid:
            reason = vad_meta.get("reason", "unknown")
            self.status.emit("fail", f"未检测到有效人声 ({reason})")
            return

        self.status.emit("verifying", "正在验证身份...")
        processed = self.preprocessor.process_for_speaker(speech_audio)
        is_verified, user_id, sim = self.verifier.verify_any_user(processed)

        if is_verified:
            self.status.emit("pass", f"{user_id}|{sim:.3f}")
        else:
            best = user_id if user_id else "未知"
            self.status.emit("fail", f"最接近: {best} ({sim:.3f})")

    def stop(self):
        self.running = False


class LoginWindow(QWidget):
    """Voiceprint verification login screen."""
    login_success = pyqtSignal(str)

    def __init__(self, capture, preprocessor, verifier):
        super().__init__()
        self.capture = capture
        self.preprocessor = preprocessor
        self.verifier = verifier
        self.verify_thread = None
        self._bg_pixmap = None
        bg_path = os.path.join(os.path.dirname(__file__), "bg.jpg")
        if os.path.exists(bg_path):
            self._bg_pixmap = QPixmap(bg_path)
        self._build_ui()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        if self._bg_pixmap and not self._bg_pixmap.isNull():
            scaled = self._bg_pixmap.scaled(
                self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            x = (scaled.width() - self.width()) // 2
            y = (scaled.height() - self.height()) // 2
            painter.drawPixmap(0, 0, self.width(), self.height(), scaled, x, y, self.width(), self.height())
            painter.fillRect(self.rect(), QColor(10, 10, 26, 180))
        else:
            painter.fillRect(self.rect(), QColor(10, 10, 26))
        painter.end()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignCenter)
        layout.addStretch(2)

        title = QLabel("语音识别助手")
        title.setFont(QFont("Microsoft YaHei", 28, QFont.Bold))
        title.setStyleSheet(f"color: {COLORS['text_primary']}; background: transparent;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("声纹身份验证")
        subtitle.setFont(QFont("Microsoft YaHei", 14))
        subtitle.setStyleSheet(f"color: {COLORS['text_secondary']}; background: transparent;")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        layout.addStretch(1)

        # Listening pulse (mic icon)
        mic_path = os.path.join(os.path.dirname(__file__), "mic.png")
        self.pulse = ListeningWidget(icon_path=mic_path, size=160)
        layout.addWidget(self.pulse, alignment=Qt.AlignCenter)

        # Status label
        self.status_label = QLabel("点击下方按钮开始验证")
        self.status_label.setFont(QFont("Microsoft YaHei", 13))
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']}; background: transparent;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # User hint
        users = self.verifier.list_users() if self.verifier else []
        hint_text = f"已注册用户: {', '.join(users)}" if users else "暂无已注册用户"
        hint = QLabel(hint_text)
        hint.setFont(QFont("Microsoft YaHei", 10))
        hint.setStyleSheet(f"color: {COLORS['text_muted']}; background: transparent;")
        hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint)

        layout.addSpacing(16)

        # Verify button
        self.verify_btn = QPushButton("  开始声纹验证")
        self.verify_btn.setFixedHeight(48)
        self.verify_btn.setFixedWidth(260)
        self.verify_btn.setFont(QFont("Microsoft YaHei", 12))
        self.verify_btn.setCursor(Qt.PointingHandCursor)
        self.verify_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['gradient_start']}, stop:1 {COLORS['gradient_end']});
                color: white; border: none; border-radius: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background: {COLORS['accent']}; }}
        """)
        self.verify_btn.clicked.connect(self._start_verify)
        layout.addWidget(self.verify_btn, alignment=Qt.AlignCenter)

        # Skip button
        self.skip_btn = QPushButton("跳过验证")
        self.skip_btn.setFont(QFont("Microsoft YaHei", 10))
        self.skip_btn.setCursor(Qt.PointingHandCursor)
        self.skip_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {COLORS['text_muted']};
                border: 1px solid {COLORS['text_muted']}; border-radius: 6px;
                padding: 6px 20px;
            }}
            QPushButton:hover {{ color: {COLORS['text_primary']}; border-color: {COLORS['text_primary']}; }}
        """)
        self.skip_btn.clicked.connect(lambda: self.login_success.emit("guest"))
        layout.addWidget(self.skip_btn, alignment=Qt.AlignCenter)

        layout.addStretch(3)

    def _start_verify(self):
        self.verify_btn.setEnabled(False)
        self.verify_btn.setText("  验证中...")
        self.pulse.start_animation()

        self.verify_thread = VerifyThread(
            self.capture, self.preprocessor, self.verifier
        )
        self.verify_thread.status.connect(self._on_status)
        self.verify_thread.start()

    def _on_status(self, key, detail):
        self.pulse.stop_animation()
        self.verify_btn.setEnabled(True)
        self.verify_btn.setText("  重新验证")

        if key == "recording":
            self.status_label.setText(detail)
            self.status_label.setStyleSheet(f"color: {COLORS['warning']}; background: transparent;")
            self.pulse.start_animation()
        elif key == "verifying":
            self.status_label.setText(detail)
            self.status_label.setStyleSheet(f"color: {COLORS['accent']}; background: transparent;")
        elif key == "pass":
            uid = detail.split("|")[0]
            sim = detail.split("|")[1] if "|" in detail else ""
            self.status_label.setText(f"验证通过! 欢迎, {uid} ({sim})")
            self.status_label.setStyleSheet(f"color: {COLORS['success']}; font-weight: bold; background: transparent;")
            QTimer.singleShot(1200, lambda: self.login_success.emit(uid))
        elif key == "fail":
            self.status_label.setText(f"验证失败 - {detail}")
            self.status_label.setStyleSheet(f"color: {COLORS['error']}; background: transparent;")
        elif key == "error":
            self.status_label.setText(detail)
            self.status_label.setStyleSheet(f"color: {COLORS['error']}; background: transparent;")

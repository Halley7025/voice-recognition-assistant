"""Login window with voiceprint verification and enrollment."""
import os
import time
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
    QLineEdit, QComboBox, QStackedWidget, QSizePolicy
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPainter, QPixmap

from gui_theme import COLORS
from gui_widgets import ListeningWidget
from global_config import SAMPLE_RATE
from audio.dynamic_sampler import DynamicAudioSampler

# 声纹验证录音时长（秒）
_VERIFY_DURATION_SEC = 7


# ============================================================
# VerifyThread: voiceprint verification in background
# ============================================================
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
            self.status.emit("error", "\u58f0\u7eb9\u6a21\u578b\u672a\u52a0\u8f7d")
            return
        if not self.verifier.list_users():
            self.status.emit("error", "\u65e0\u5df2\u6ce8\u518c\u7528\u6237\uff0c\u8bf7\u5148\u6ce8\u518c")
            return

        dur = _VERIFY_DURATION_SEC
        self.status.emit("recording", f"\u8bf7\u8bf4\u8bdd ({dur}\u79d2)...")

        chunks = []
        total_chunks = int(self.capture.sample_rate / self.capture.chunk * dur)
        for _ in range(total_chunks):
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
            self.status.emit("fail", "\u5f55\u97f3\u592a\u77ed\u6216\u65e0\u58f0\u97f3")
            return

        is_valid, speech_audio, vad_meta = self.preprocessor.strict_vad(raw)
        if not is_valid:
            reason = vad_meta.get("reason", "unknown")
            self.status.emit("fail", f"\u672a\u68c0\u6d4b\u5230\u6709\u6548\u4eba\u58f0 ({reason})")
            return

        self.status.emit("verifying", "\u6b63\u5728\u9a8c\u8bc1\u8eab\u4efd...")
        processed = self.preprocessor.process_for_speaker(speech_audio)
        is_verified, user_id, sim = self.verifier.verify_any_user(processed)

        if is_verified:
            self.status.emit("pass", f"{user_id}|{sim:.3f}")
        else:
            best = user_id if user_id else "\u672a\u77e5"
            self.status.emit("fail", f"\u6700\u63a5\u8fd1: {best} ({sim:.3f})")

    def stop(self):
        self.running = False


# ============================================================
# EnrollThread: voiceprint enrollment (3 samples) in background
# ============================================================
class EnrollThread(QThread):
    """Background thread for voiceprint enrollment with 3 audio samples."""
    status = pyqtSignal(str, str)
    level = pyqtSignal(float)

    def __init__(self, capture, preprocessor, verifier, user_id, dynamic_sampler):
        super().__init__()
        self.capture = capture
        self.preprocessor = preprocessor
        self.verifier = verifier
        self.user_id = user_id
        self.dynamic_sampler = dynamic_sampler
        self.running = True

    def run(self):
        if not self.verifier or self.verifier.model is None:
            self.status.emit("error", "\u58f0\u7eb9\u6a21\u578b\u672a\u52a0\u8f7d")
            return

        prompts = self.dynamic_sampler.get_prompts(count=3)
        samples = []

        for i, prompt_info in enumerate(prompts):
            if not self.running:
                return
            prompt_text = prompt_info["text"]
            duration = prompt_info["duration"]

            # \u901a\u77e5 UI \u663e\u793a\u63d0\u793a\u8bcd\u548c\u5012\u8ba1\u65f6
            self.status.emit("prompt", f"{i+1}|{prompt_text}|{duration:.1f}")

            # \u5f55\u97f3
            self.status.emit("recording", f"\u91c7\u6837 {i+1}/3 \u5f55\u97f3\u4e2d ({duration:.1f}s)...")
            chunks = []
            total_chunks = int(self.capture.sample_rate / self.capture.chunk * duration)
            for j in range(total_chunks):
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
            processed = self.preprocessor.process(raw)

            if len(processed) > SAMPLE_RATE * 0.5:
                samples.append(processed)
                self.status.emit("sample_ok", f"{i+1}")
            else:
                self.status.emit("sample_fail", f"{i+1}")

        # \u6ce8\u518c
        self.status.emit("enrolling", "\u6b63\u5728\u6ce8\u518c...")
        success = self.verifier.register_speaker(self.user_id, samples)
        if success:
            self.status.emit("done", f"{len(samples)}|{len(prompts)}")
        else:
            self.status.emit("error", "\u6ce8\u518c\u5931\u8d25")

    def stop(self):
        self.running = False


# ============================================================
# LoginWindow: login screen with verify + enroll
# ============================================================
class LoginWindow(QWidget):
    """Voiceprint verification and enrollment login screen."""
    login_success = pyqtSignal(str)

    def __init__(self, capture, preprocessor, verifier):
        super().__init__()
        self.capture = capture
        self.preprocessor = preprocessor
        self.verifier = verifier
        self.verify_thread = None
        self.enroll_thread = None
        self.dynamic_sampler = DynamicAudioSampler(samples_per_enroll=3)
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

        # \u9876\u90e8\u7559\u767d\uff0c\u628a\u6807\u9898\u533a\u57df\u5f80\u4e0b\u63a8
        layout.addStretch(1)

        # ---- \u4e3b\u6807\u9898 ----
        title = QLabel("\u8bed\u97f3\u8bc6\u522b\u52a9\u624b")
        title.setStyleSheet(
            f"color: {COLORS['text_primary']}; font-family: Microsoft YaHei; font-size: 80px; font-weight: bold; background: transparent;"
        )
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        layout.addSpacing(12)

        # ---- \u526f\u6807\u9898 ----
        subtitle = QLabel("声纹身份验证")
        subtitle.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-family: Microsoft YaHei; font-size: 72px; background: transparent;"
        )
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(40)

        # ---- \u8109\u51b2\u52a8\u753b ----
        mic_path = os.path.join(os.path.dirname(__file__), "mic.png")
        self.pulse = ListeningWidget(icon_path=mic_path, size=180)
        layout.addWidget(self.pulse, alignment=Qt.AlignCenter)

        layout.addSpacing(30)

        # ---- \u72b6\u6001\u6807\u7b7e ----
        self.status_label = QLabel("\u70b9\u51fb\u4e0b\u65b9\u6309\u94ae\u5f00\u59cb\u9a8c\u8bc1")
        self.status_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-family: Microsoft YaHei; font-size: 36px; font-weight: bold; background: transparent;"
        )
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # ---- \u7528\u6237\u540d\u8f93\u5165\u6846\uff08\u6ce8\u518c\u7528\uff09 ----
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("\u8f93\u5165\u7528\u6237\u540d\u7528\u4e8e\u6ce8\u518c...")
        self.user_input.setFixedWidth(800)
        self.user_input.setFixedHeight(110)
        self.user_input.setFont(QFont("Microsoft YaHei", 28))
        self.user_input.setStyleSheet(
            "QLineEdit {"
            f"  background: rgba(255,255,255,0.08);"
            f"  color: {COLORS['text_primary']};"
            f"  border: 1px solid {COLORS['text_muted']};"
            "  border-radius: 25px;"
            "  padding: 15px 40px;"
            "}"
            f"QLineEdit:focus {{ border-color: {COLORS['accent']}; }}"
        )
        self.user_input.setVisible(False)
        layout.addWidget(self.user_input, alignment=Qt.AlignCenter)

        layout.addSpacing(24)

        # ---- \u7528\u6237\u5217\u8868\u63d0\u793a ----
        self._refresh_user_hint()
        layout.addWidget(self.hint_label)

        layout.addSpacing(30)

        # ---- \u6309\u94ae\u533a\u57df\uff1a\u9a8c\u8bc1 + \u6ce8\u518c ----
        btn_row = QHBoxLayout()
        btn_row.setSpacing(60)
        btn_row.setAlignment(Qt.AlignCenter)

        btn_style_grad = (
            "QPushButton {"
            f"  background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            f"    stop:0 {COLORS['gradient_start']}, stop:1 {COLORS['gradient_end']});"
            "  color: white; border: none; border-radius: 30px;"
            "  font-family: Microsoft YaHei; font-size: 40px; font-weight: bold;"
            "}"
            f"QPushButton:hover {{ background: {COLORS['accent']}; }}"
            "QPushButton:disabled { background: #333; color: #666; }"
        )
        btn_style_outline = (
            "QPushButton {"
            "  background: rgba(255,255,255,0.06);"
            f"  color: {COLORS['text_primary']};"
            f"  border: 1px solid {COLORS['gradient_start']};"
            "  border-radius: 30px;"
            "  font-family: Microsoft YaHei; font-size: 40px; font-weight: bold;"
            f"QPushButton:hover {{ background: rgba(0,150,255,0.15); border-color: {COLORS['accent']}; }}"
            "QPushButton:disabled { background: #222; color: #555; border-color: #333; }"
        )

        self.verify_btn = QPushButton("  \u5f00\u59cb\u58f0\u7eb9\u9a8c\u8bc1")
        self.verify_btn.setFixedHeight(126)
        self.verify_btn.setMinimumWidth(490)
        self.verify_btn.setCursor(Qt.PointingHandCursor)
        self.verify_btn.setStyleSheet(btn_style_grad)
        self.verify_btn.clicked.connect(self._start_verify)
        btn_row.addWidget(self.verify_btn)

        self.enroll_btn = QPushButton("  \u6ce8\u518c\u65b0\u7528\u6237")
        self.enroll_btn.setFixedHeight(126)
        self.enroll_btn.setMinimumWidth(490)
        self.enroll_btn.setCursor(Qt.PointingHandCursor)
        self.enroll_btn.setStyleSheet(
            "QPushButton {"
            "  background: rgba(255,255,255,0.06);"
            f"  color: {COLORS['text_primary']};"
            f"  border: 1px solid {COLORS['gradient_start']};"
            "  border-radius: 30px;"
            "  font-family: Microsoft YaHei; font-size: 40px; font-weight: bold;"
            "}"
            f"QPushButton:hover {{ background: rgba(0,150,255,0.15); border-color: {COLORS['accent']}; }}"
            "QPushButton:disabled { background: #222; color: #555; border-color: #333; }"
        )
        self.enroll_btn.clicked.connect(self._start_enroll)
        btn_row.addWidget(self.enroll_btn)

        layout.addLayout(btn_row)

        layout.addSpacing(20)

        # ---- \u8df3\u8fc7\u6309\u94ae ----
        self.skip_btn = QPushButton("\u8df3\u8fc7\u9a8c\u8bc1")
        self.skip_btn.setCursor(Qt.PointingHandCursor)
        self.skip_btn.setMinimumHeight(90)
        self.skip_btn.setStyleSheet(
            "QPushButton {"
            f"  background: transparent; color: {COLORS['text_muted']};"
            f"  border: 1px solid {COLORS['text_muted']}; border-radius: 25px; font-family: Microsoft YaHei; font-size: 35px;"
            "}"
            f"QPushButton:hover {{ color: {COLORS['text_primary']}; border-color: {COLORS['text_primary']}; }}"
        )
        self.skip_btn.clicked.connect(lambda: self.login_success.emit("guest"))
        layout.addWidget(self.skip_btn, alignment=Qt.AlignCenter)

        # \u5e95\u90e8\u7559\u767d\uff0c\u8ba9\u5185\u5bb9\u5742\u7a33\u5c45\u4e2d
        layout.addStretch(2)

    # ------------------------------------------------------------------
    # \u7528\u6237\u5217\u8868\u63d0\u793a\u5237\u65b0
    # ------------------------------------------------------------------
    def _refresh_user_hint(self):
        users = self.verifier.list_users() if self.verifier else []
        hint_text = (
            f"\u5df2\u6ce8\u518c\u7528\u6237: {', '.join(users)}"
            if users else "\u6682\u65e0\u5df2\u6ce8\u518c\u7528\u6237"
        )
        if hasattr(self, "hint_label"):
            self.hint_label.setText(hint_text)
        else:
            self.hint_label = QLabel(hint_text)
            self.hint_label.setStyleSheet(
                f"color: {COLORS['text_muted']}; font-family: Microsoft YaHei; font-size: 36px; background: transparent;"
            )
            self.hint_label.setAlignment(Qt.AlignCenter)

    def refresh_users(self):
        """\u5916\u90e8\u8c03\u7528\uff1a\u767b\u5f55\u754c\u9762\u663e\u793a\u65f6\u5237\u65b0\u7528\u6237\u5217\u8868"""
        self._refresh_user_hint()

    # ------------------------------------------------------------------
    # \u58f0\u7eb9\u9a8c\u8bc1
    # ------------------------------------------------------------------
    def _start_verify(self):
        self._set_buttons_enabled(False)
        self.verify_btn.setText("  \u9a8c\u8bc1\u4e2d...")
        self.user_input.setVisible(False)
        self.pulse.start_animation()

        self.verify_thread = VerifyThread(
            self.capture, self.preprocessor, self.verifier
        )
        self.verify_thread.status.connect(self._on_verify_status)
        self.verify_thread.start()

    def _on_verify_status(self, key, detail):
        self.pulse.stop_animation()
        self._set_buttons_enabled(True)
        self.verify_btn.setText("  \u91cd\u65b0\u9a8c\u8bc1")

        if key == "recording":
            self.status_label.setText(detail)
            self.status_label.setStyleSheet(
                f"color: {COLORS['warning']}; font-family: Microsoft YaHei; font-size: 36px; font-weight: bold; background: transparent;"
            )
            self.pulse.start_animation()
        elif key == "verifying":
            self.status_label.setText(detail)
            self.status_label.setStyleSheet(
                f"color: {COLORS['accent']}; font-family: Microsoft YaHei; font-size: 36px; background: transparent;"
            )
        elif key == "pass":
            uid = detail.split("|")[0]
            sim = detail.split("|")[1] if "|" in detail else ""
            self.status_label.setText(f"\u9a8c\u8bc1\u901a\u8fc7! \u6b22\u8fce, {uid} ({sim})")
            self.status_label.setStyleSheet(
                f"color: {COLORS['success']}; font-family: Microsoft YaHei; font-size: 36px; font-weight: bold; background: transparent;"
            )
            QTimer.singleShot(1200, lambda: self.login_success.emit(uid))
        elif key == "fail":
            self.status_label.setText(f"\u9a8c\u8bc1\u5931\u8d25 - {detail}")
            self.status_label.setStyleSheet(
                f"color: {COLORS['error']}; font-family: Microsoft YaHei; font-size: 36px; font-weight: bold; background: transparent;"
            )
        elif key == "error":
            self.status_label.setText(detail)
            self.status_label.setStyleSheet(
                f"color: {COLORS['error']}; font-family: Microsoft YaHei; font-size: 36px; font-weight: bold; background: transparent;"
            )

    # ------------------------------------------------------------------
    # \u58f0\u7eb9\u6ce8\u518c
    # ------------------------------------------------------------------
    def _start_enroll(self):
        """\u51c6\u5907\u6ce8\u518c\uff1a\u663e\u793a\u7528\u6237\u540d\u8f93\u5165\u6846"""
        if not self.verifier or self.verifier.model is None:
            self.status_label.setText("\u58f0\u7eb9\u6a21\u578b\u672a\u52a0\u8f7d\uff0c\u65e0\u6cd5\u6ce8\u518c")
            self.status_label.setStyleSheet(
                f"color: {COLORS['error']}; font-family: Microsoft YaHei; font-size: 36px; font-weight: bold; background: transparent;"
            )
            return

        # \u5982\u679c\u8f93\u5165\u6846\u5df2\u663e\u793a\uff0c\u8bf4\u660e\u662f\u7b2c\u4e8c\u6b21\u70b9\u51fb -> \u6267\u884c\u6ce8\u518c
        if self.user_input.isVisible():
            user_id = self.user_input.text().strip()
            if not user_id:
                self.status_label.setText("\u8bf7\u8f93\u5165\u7528\u6237\u540d")
                self.status_label.setStyleSheet(
                    f"color: {COLORS['warning']}; font-family: Microsoft YaHei; font-size: 36px; font-weight: bold; background: transparent;"
                )
                return
            # \u68c0\u67e5\u91cd\u590d
            if user_id in self.verifier.list_users():
                self.status_label.setText(
                    f"\u7528\u6237 '{user_id}' \u5df2\u5b58\u5728\uff0c\u5c06\u91cd\u65b0\u6ce8\u518c"
                )
                self.status_label.setStyleSheet(
                    f"color: {COLORS['warning']}; font-family: Microsoft YaHei; font-size: 36px; font-weight: bold; background: transparent;"
                )
            self._do_enroll(user_id)
            return

        # \u7b2c\u4e00\u6b21\u70b9\u51fb -> \u663e\u793a\u8f93\u5165\u6846
        self.user_input.setVisible(True)
        self.user_input.setFocus()
        self.status_label.setText(
            "\u8bf7\u8f93\u5165\u7528\u6237\u540d\uff0c\u518d\u6b21\u70b9\u51fb\u6ce8\u518c\u6309\u94ae\u5f00\u59cb"
        )
        self.status_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-family: Microsoft YaHei; font-size: 36px; font-weight: bold; background: transparent;"
        )

    def _do_enroll(self, user_id):
        """\u6267\u884c\u6ce8\u518c\u6d41\u7a0b"""
        self._set_buttons_enabled(False)
        self.enroll_btn.setText("  \u6ce8\u518c\u4e2d...")
        self.user_input.setEnabled(False)
        self.pulse.start_animation()

        self.enroll_thread = EnrollThread(
            self.capture, self.preprocessor, self.verifier,
            user_id, self.dynamic_sampler
        )
        self.enroll_thread.status.connect(
            lambda key, detail: self._on_enroll_status(key, detail, user_id)
        )
        self.enroll_thread.start()

    def _on_enroll_status(self, key, detail, user_id):
        """\u5904\u7406\u6ce8\u518c\u7ebf\u7a0b\u7684\u72b6\u6001\u56de\u8c03"""
        if key == "prompt":
            parts = detail.split("|")
            idx, prompt_text, duration = parts[0], parts[1], parts[2]
            self.status_label.setText(
                f"\u91c7\u6837 {idx}/3 \u8bf7\u6717\u8bfb: {prompt_text}"
            )
            self.status_label.setStyleSheet(
                f"color: #00E5FF; font-family: Microsoft YaHei; font-size: 36px; background: transparent;"
            )
        elif key == "recording":
            self.status_label.setText(detail)
            self.status_label.setStyleSheet(
                f"color: {COLORS['warning']}; font-family: Microsoft YaHei; font-size: 36px; font-weight: bold; background: transparent;"
            )
        elif key == "sample_ok":
            self.status_label.setText(f"\u2713 \u91c7\u6837 {detail} \u5b8c\u6210")
            self.status_label.setStyleSheet(
                f"color: {COLORS['success']}; font-family: Microsoft YaHei; font-size: 36px; background: transparent;"
            )
        elif key == "sample_fail":
            self.status_label.setText(
                f"\u2717 \u91c7\u6837 {detail} \u65e0\u6548\uff0c\u5f55\u97f3\u8fc7\u77ed"
            )
            self.status_label.setStyleSheet(
                f"color: {COLORS['warning']}; font-family: Microsoft YaHei; font-size: 36px; font-weight: bold; background: transparent;"
            )
        elif key == "enrolling":
            self.status_label.setText(detail)
            self.status_label.setStyleSheet(
                f"color: {COLORS['accent']}; font-family: Microsoft YaHei; font-size: 36px; background: transparent;"
            )
        elif key == "done":
            ok_count, total = detail.split("|")
            self.pulse.stop_animation()
            self._set_buttons_enabled(True)
            self.enroll_btn.setText("  \u6ce8\u518c\u65b0\u7528\u6237")
            self.user_input.setEnabled(True)
            self.user_input.setVisible(False)
            self.user_input.clear()
            self.status_label.setText(
                f"\u2713 \u6ce8\u518c\u6210\u529f: {user_id} (\u6709\u6548\u6837\u672c {ok_count}/{total})"
            )
            self.status_label.setStyleSheet(
                f"color: {COLORS['success']}; font-family: Microsoft YaHei; font-size: 36px; font-weight: bold; background: transparent;"
            )
            self._refresh_user_hint()
        elif key == "error":
            self.pulse.stop_animation()
            self._set_buttons_enabled(True)
            self.enroll_btn.setText("  \u6ce8\u518c\u65b0\u7528\u6237")
            self.user_input.setEnabled(True)
            self.status_label.setText(detail)
            self.status_label.setStyleSheet(
                f"color: {COLORS['error']}; font-family: Microsoft YaHei; font-size: 36px; font-weight: bold; background: transparent;"
            )

    # ------------------------------------------------------------------
    # \u5de5\u5177\u65b9\u6cd5
    # ------------------------------------------------------------------
    def _set_buttons_enabled(self, enabled):
        self.verify_btn.setEnabled(enabled)
        self.enroll_btn.setEnabled(enabled)
        self.skip_btn.setEnabled(enabled)

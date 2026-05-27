"""Real-time audio waveform widget using QPainter (no external dependencies)."""
import numpy as np
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtGui import QPainter, QPen, QColor, QLinearGradient, QBrush


class WaveformWidget(QWidget):
    """Real-time audio waveform display with glowing cyan line.

    Usage:
        widget = WaveformWidget()
        # Feed audio chunks:
        widget.push_audio(chunk_numpy_array)
    """

    def __init__(self, parent=None, history_len=2048, color="#00e5ff"):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setMaximumHeight(120)
        self._buffer = np.zeros(history_len, dtype=np.float32)
        self._history_len = history_len
        self._color = QColor(color)
        self._glow_color = QColor(color)
        self._glow_color.setAlpha(60)
        self._is_active = False
        self._smooth_level = 0.0

        # Animation timer
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)  # ~30fps

    def push_audio(self, chunk):
        """Push a numpy audio chunk into the display buffer."""
        if chunk is None or len(chunk) == 0:
            return
        chunk = np.asarray(chunk, dtype=np.float32)
        # Shift buffer and append new data
        n = len(chunk)
        if n >= self._history_len:
            self._buffer[:] = chunk[-self._history_len:]
        else:
            self._buffer[:-n] = self._buffer[n:]
            self._buffer[-n:] = chunk
        # Update smooth level (peak RMS of recent chunk)
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        self._smooth_level = max(rms * 3.0, self._smooth_level * 0.85)

    def set_active(self, active):
        self._is_active = active
        if not active:
            self._smooth_level = 0.0

    def _tick(self):
        if not self._is_active:
            self._smooth_level *= 0.92
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        mid_y = h / 2.0

        # Background
        painter.fillRect(0, 0, w, h, QColor("#0a0a1a"))

        # Draw center line (subtle)
        painter.setPen(QPen(QColor(255, 255, 255, 15), 1))
        painter.drawLine(0, int(mid_y), w, int(mid_y))

        if self._smooth_level < 0.001:
            painter.end()
            return

        # Prepare waveform points
        buf = self._buffer
        n = len(buf)
        # Downsample to ~200 points for smooth drawing
        target_points = min(200, w)
        if n > target_points:
            indices = np.linspace(0, n - 1, target_points).astype(int)
            samples = buf[indices]
        else:
            samples = buf
            target_points = n

        # Scale
        amplitude = mid_y * 0.8 * min(self._smooth_level * 5.0, 1.0)

        # Build point list
        points = []
        for i, s in enumerate(samples):
            x = i * w / max(target_points - 1, 1)
            y = mid_y - s * amplitude
            points.append(QPointF(x, y))

        # Draw glow layer (thick, transparent)
        glow_pen = QPen(self._glow_color, 6, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(glow_pen)
        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])

        # Draw main line (thin, bright)
        main_pen = QPen(self._color, 2, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(main_pen)
        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])

        painter.end()

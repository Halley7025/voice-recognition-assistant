"""Premium UI widgets: AcrylicCard + BreathingPulse."""
import os
import math
from PyQt5.QtWidgets import QWidget, QGraphicsDropShadowEffect, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty, QRectF, QPointF
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QPixmap, QPainterPath


class AcrylicCard(QWidget):
    """Translucent acrylic card with rounded corners and soft shadow."""

    def __init__(self, parent=None, radius=15, bg_color=(20, 25, 35, 180)):
        super().__init__(parent)
        self._radius = radius
        self._bg = QColor(*bg_color)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")

        # Soft floating shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(25)
        shadow.setColor(QColor(0, 0, 0, 90))
        shadow.setOffset(0, 6)
        self.setGraphicsEffect(shadow)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        rect = QRectF(self.rect()).adjusted(4, 4, -4, -4)
        path.addRoundedRect(rect, self._radius, self._radius)
        painter.fillPath(path, self._bg)
        painter.end()


class ListeningWidget(QWidget):
    """Breathing pulse sphere with ripple animation for voice input.

    Displays a center circle (with optional mic icon) surrounded by
    expanding/fading ripple rings in cyberpunk cyan style.
    """

    def __init__(self, parent=None, icon_path=None, size=160):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._icon_path = icon_path
        self._icon_pixmap = None
        if icon_path and os.path.exists(icon_path):
            px = QPixmap(icon_path)
            icon_size = int(size * 0.35)
            self._icon_pixmap = px.scaled(
                icon_size, icon_size,
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

        self._phase = 0.0          # Animation phase [0, 2*PI)
        self._active = False       # Is animation running
        self._opacity = 0.0        # Fade in/out opacity
        self._target_opacity = 0.0
        self._center_radius = int(size * 0.22)
        self._color1 = QColor(0, 240, 255)   # Cyan #00F0FF
        self._color2 = QColor(181, 55, 242)  # Neon purple #B537F2

        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)  # ~30fps

    def start_animation(self):
        self._active = True
        self._target_opacity = 1.0

    def stop_animation(self):
        self._active = False
        self._target_opacity = 0.0

    def _tick(self):
        # Smooth opacity transition
        diff = self._target_opacity - self._opacity
        self._opacity += diff * 0.08
        if abs(diff) < 0.005:
            self._opacity = self._target_opacity

        if self._active:
            self._phase = (self._phase + 0.08) % (2 * math.pi)
        elif self._opacity < 0.01:
            return  # Don't repaint when fully hidden

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setOpacity(max(self._opacity, 0.0))

        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0

        # Background (transparent)
        painter.fillRect(0, 0, w, h, Qt.transparent)

        if self._active and self._opacity > 0.05:
            # Draw 2 ripple rings
            for i in range(2):
                phase = self._phase + i * math.pi  # offset by PI
                # Radius expands from center_radius to 2x
                t = (math.sin(phase) + 1) / 2  # [0, 1]
                ring_r = self._center_radius + t * self._center_radius * 1.2
                # Alpha fades from 150 to 0
                alpha = int(150 * (1 - t))
                # Alternate colors
                color = QColor(self._color1) if i == 0 else QColor(self._color2)
                color.setAlpha(alpha)
                pen = QPen(color, 2.5)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(
                    QPointF(cx, cy), ring_r, ring_r
                )

        # Center solid circle
        center_color = QColor(0, 240, 255, int(200 * self._opacity))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(center_color))
        painter.drawEllipse(QPointF(cx, cy), self._center_radius, self._center_radius)

        # Draw mic icon if available
        if self._icon_pixmap and self._opacity > 0.1:
            icon_w = self._icon_pixmap.width()
            icon_h = self._icon_pixmap.height()
            x = cx - icon_w / 2
            y = cy - icon_h / 2
            painter.setOpacity(self._opacity * 0.9)
            painter.drawPixmap(int(x), int(y), self._icon_pixmap)

        painter.end()




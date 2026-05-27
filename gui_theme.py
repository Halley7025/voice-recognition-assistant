"""Modern dark theme - enlarged fonts, proper sizing, Fluent Design."""

COLORS = {
    "bg_dark": "#0a0a1a",
    "bg_card": "#141923",
    "bg_input": "#1a2030",
    "accent": "#00e5ff",
    "accent_hover": "#40efff",
    "accent_dim": "#006680",
    "success": "#00e676",
    "warning": "#ffab40",
    "error": "#ff5252",
    "text_primary": "#e2e8f0",
    "text_secondary": "#8899aa",
    "text_muted": "#4a5568",
    "border": "#1e2d4a",
    "gradient_start": "#667eea",
    "gradient_end": "#764ba2",
}

DARK_STYLE = """
/* ========== Global ========== */
* {
    font-family: "Microsoft YaHei", "Segoe UI", "PingFang SC", sans-serif;
}
QMainWindow, QWidget {
    background-color: transparent;
    color: #e2e8f0;
    font-size: 14px;
}

/* ========== Buttons ========== */
QPushButton {
    background-color: #1e293b;
    color: #e2e8f0;
    border: none;
    border-radius: 8px;
    padding: 8px 20px;
    font-size: 15px;
    min-height: 38px;
}
QPushButton:hover {
    background-color: #2563eb;
}
QPushButton:pressed {
    background-color: #1d4ed8;
    padding-top: 9px;
    padding-bottom: 7px;
}
QPushButton:disabled {
    background-color: #0f172a;
    color: #475569;
}

/* ========== Inputs ========== */
QLineEdit {
    background-color: rgba(255, 255, 255, 10);
    color: #e2e8f0;
    border: none;
    border-bottom: 2px solid #1e2d4a;
    border-radius: 6px;
    padding: 0 15px;
    font-size: 14px;
    min-height: 40px;
    selection-background-color: #00e5ff;
}
QLineEdit:focus {
    border-bottom: 2px solid #00e5ff;
    background-color: rgba(255, 255, 255, 15);
}

/* ========== ComboBox ========== */
QComboBox {
    background-color: rgba(255, 255, 255, 10);
    color: #e2e8f0;
    border: none;
    border-bottom: 2px solid #1e2d4a;
    border-radius: 6px;
    padding: 0 15px;
    font-size: 14px;
    min-height: 40px;
}
QComboBox:focus {
    border-bottom: 2px solid #00e5ff;
}
QComboBox::drop-down {
    border: none;
    width: 30px;
}
QComboBox QAbstractItemView {
    background-color: #141923;
    color: #e2e8f0;
    selection-background-color: #2563eb;
    border: 1px solid #1e2d4a;
    border-radius: 6px;
    font-size: 14px;
    padding: 4px;
}

/* ========== TextEdit ========== */
QTextEdit {
    background-color: rgba(255, 255, 255, 5);
    color: #e2e8f0;
    border: none;
    border-radius: 8px;
    padding: 12px;
    font-size: 14px;
}

/* ========== GroupBox (Card sections) ========== */
QGroupBox {
    background-color: rgba(20, 25, 35, 180);
    border: none;
    border-radius: 12px;
    margin-top: 16px;
    padding: 24px 20px 20px 20px;
    font-size: 16px;
    font-weight: bold;
    color: #8899aa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 20px;
    padding: 0 10px;
    color: #00e5ff;
    font-size: 16px;
    font-weight: bold;
}

/* ========== TabWidget ========== */
QTabWidget::pane {
    background-color: rgba(20, 25, 35, 160);
    border: none;
    border-radius: 10px;
    padding: 8px;
}
QTabBar::tab {
    background-color: transparent;
    color: #8899aa;
    padding: 12px 24px;
    border-bottom: 2px solid transparent;
    font-size: 14px;
}
QTabBar::tab:selected {
    color: #00e5ff;
    border-bottom: 2px solid #00e5ff;
}
QTabBar::tab:hover {
    color: #e2e8f0;
}

/* ========== ProgressBar ========== */
QProgressBar {
    background-color: rgba(255, 255, 255, 8);
    border: none;
    border-radius: 5px;
    text-align: center;
    color: #e2e8f0;
    font-size: 12px;
    min-height: 12px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #667eea, stop:1 #764ba2);
    border-radius: 5px;
}

/* ========== ScrollBar ========== */
QScrollBar:vertical {
    background: transparent;
    width: 8px;
}
QScrollBar::handle:vertical {
    background: #1e2d4a;
    border-radius: 4px;
    min-height: 40px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

/* ========== Labels ========== */
QLabel {
    font-size: 14px;
    color: #e2e8f0;
}
"""


def apply_theme(app):
    """Apply modern dark theme to the application."""
    app.setStyleSheet(DARK_STYLE)

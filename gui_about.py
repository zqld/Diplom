import os
import sys
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QPushButton,
                             QFrame, QHBoxLayout, QTextEdit)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QTextOption
from src.screen_utils import window_geometry
from build_utils import resource_path


DARK_COLORS = {
    'bg_main': '#1A1A1F',
    'bg_card': '#252530',
    'bg_input': '#2D2D3A',
    'text_primary': '#FFFFFF',
    'text_secondary': '#A0A0B0',
    'text_muted': '#6A6A7A',
    'accent': '#6B8AFE',
    'accent_hover': '#8AA3FF',
    'border': '#3A3A45',
    'good': '#4ADE80',
    'warning': '#FBBF24',
    'danger': '#F87171',
}


CONSENT_TEXT_FILENAME = "consent_text.txt"

CONSENT_TEXT_FALLBACK = (
    "Внимание! Данное программное обеспечение производит локальный анализ "
    "биометрических параметров (точек лица, рук, плеч) исключительно в целях "
    "мониторинга усталости. Данные сохраняются на данном компьютере в "
    "обезличенном виде и не передаются в сеть Интернет. Использование "
    "программы сотрудниками организации допускается только после подписания "
    "Согласия на обработку ПДн у ответственного сотрудника отдела кадров."
)


def _get_consent_text_path():
    """Получить путь к файлу согласия, создать файл с текстом по умолчанию если его нет."""
    if getattr(sys, 'frozen', False):
        bundled = resource_path('assets/consent_text.txt')
        if os.path.exists(bundled):
            return bundled
        return None

    assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
    if not os.path.exists(assets_dir):
        try:
            os.makedirs(assets_dir)
        except Exception:
            pass

    file_path = os.path.join(assets_dir, CONSENT_TEXT_FILENAME)

    if not os.path.exists(file_path):
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(CONSENT_TEXT_FALLBACK)
        except Exception:
            pass

    return file_path


def _load_consent_text():
    """Загрузить текст согласия из файла."""
    path = _get_consent_text_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                return text
    except Exception:
        pass
    return CONSENT_TEXT_FALLBACK


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("О программе")
        x, y, w, h = window_geometry(0.4)
        self.setGeometry(x, y, w, h)
        self.setMinimumSize(400, 350)
        self.setModal(True)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {DARK_COLORS['bg_main']};
            }}
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(28, 24, 28, 24)
        main_layout.setSpacing(16)

        # Заголовок
        title = QLabel("ⓘ  О программе NeuroFocus")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {DARK_COLORS['text_primary']};")
        main_layout.addWidget(title)

        # Версия и краткое описание
        version_label = QLabel("Версия 1.0  •  Система мониторинга усталости")
        version_label.setFont(QFont("Segoe UI", 11))
        version_label.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
        main_layout.addWidget(version_label)

        # Разделитель
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet(f"background-color: {DARK_COLORS['border']};")
        separator.setFixedHeight(1)
        main_layout.addWidget(separator)

        # Подзаголовок
        consent_title = QLabel("Согласие на обработку персональных данных")
        consent_title.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
        consent_title.setStyleSheet(f"color: {DARK_COLORS['accent']};")
        main_layout.addWidget(consent_title)

        # Текст согласия в прокручиваемом окне
        text_frame = QFrame()
        text_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {DARK_COLORS['bg_card']};
                border-radius: 10px;
                border: 1px solid {DARK_COLORS['border']};
            }}
        """)
        text_layout = QVBoxLayout(text_frame)
        text_layout.setContentsMargins(14, 12, 14, 12)
        text_layout.setSpacing(0)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Segoe UI", 12))
        self.text_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: transparent;
                color: {DARK_COLORS['text_secondary']};
                border: none;
                padding: 0px;
            }}
            QScrollBar:vertical {{
                background: {DARK_COLORS['bg_input']};
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {DARK_COLORS['border']};
                border-radius: 4px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {DARK_COLORS['accent']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                border: none;
                background: none;
                height: 0px;
            }}
        """)
        self.text_edit.setText(_load_consent_text())
        self.text_edit.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        text_layout.addWidget(self.text_edit)

        main_layout.addWidget(text_frame)

        # Кнопка закрытия
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        btn_close = QPushButton("Закрыть")
        btn_close.setMinimumHeight(36)
        btn_close.setMaximumHeight(48)
        btn_close.setMinimumWidth(100)
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close.setFont(QFont("Segoe UI", 12, QFont.Weight.DemiBold))
        btn_close.setStyleSheet(f"""
            QPushButton {{
                background-color: {DARK_COLORS['accent']};
                color: #FFFFFF;
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: {DARK_COLORS['accent_hover']};
            }}
            QPushButton:pressed {{
                background-color: {DARK_COLORS['accent']};
            }}
        """)
        btn_close.clicked.connect(self.close)
        button_layout.addWidget(btn_close)

        main_layout.addLayout(button_layout)

        # Закрытие по Escape
        self.shortcut_escape = Qt.Key.Key_Escape
        # Используем keyPressEvent для обработки Escape

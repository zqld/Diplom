from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QFrame, QHBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


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
}


class GestureHelpWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Управление жестами")
        self.setFixedSize(520, 620)
        self.setModal(True)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {DARK_COLORS['bg_card']};
                border-radius: 16px;
                border: none;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(20)
        
        header = QLabel("🖱️  Управление жестами")
        header.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        header.setStyleSheet(f"color: {DARK_COLORS['text_primary']};")
        layout.addWidget(header)
        
        gestures_frame = QFrame()
        gestures_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {DARK_COLORS['bg_input']};
                border-radius: 12px;
                border: none;
            }}
        """)
        gestures_layout = QVBoxLayout(gestures_frame)
        gestures_layout.setContentsMargins(16, 16, 16, 16)
        gestures_layout.setSpacing(14)
        
        gestures = [
            ("☝️", "Только указательный палец", "Движение курсором"),
            ("✊", "Кулак (все пальцы сжаты)", "Левый клик"),
            ("🤙", "Указательный + мизинец", "Правый клик"),
            ("🤟", "3 пальца: указ. + сред. + безым.", "Прокрутка вверх ↑"),
            ("🖖", "4 пальца: указ. + сред. + безым. + мизинец", "Прокрутка вниз ↓"),
        ]
        
        for icon, gesture, action in gestures:
            row = QHBoxLayout()
            row.setSpacing(12)
            
            icon_label = QLabel(icon)
            icon_label.setFont(QFont("Segoe UI", 24))
            icon_label.setFixedWidth(50)
            row.addWidget(icon_label)
            
            text_col = QVBoxLayout()
            text_col.setSpacing(2)
            
            gesture_label = QLabel(gesture)
            gesture_label.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
            gesture_label.setStyleSheet(f"color: {DARK_COLORS['text_primary']};")
            text_col.addWidget(gesture_label)
            
            action_label = QLabel(action)
            action_label.setFont(QFont("Segoe UI", 12))
            action_label.setStyleSheet(f"color: {DARK_COLORS['text_secondary']};")
            text_col.addWidget(action_label)
            
            row.addLayout(text_col)
            row.addStretch()
            gestures_layout.addLayout(row)
        
        layout.addWidget(gestures_frame)
        
        shortcuts_frame = QFrame()
        shortcuts_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {DARK_COLORS['bg_input']};
                border-radius: 12px;
                border: none;
            }}
        """)
        shortcuts_layout = QVBoxLayout(shortcuts_frame)
        shortcuts_layout.setContentsMargins(16, 16, 16, 16)
        shortcuts_layout.setSpacing(10)
        
        shortcuts_title = QLabel("⌨️  Горячие клавиши")
        shortcuts_title.setFont(QFont("Segoe UI", 14, QFont.Weight.DemiBold))
        shortcuts_title.setStyleSheet(f"color: {DARK_COLORS['accent']};")
        shortcuts_layout.addWidget(shortcuts_title)
        
        shortcuts = [
            ("G", "Включить/выключить управление мышью"),
            ("ESC", "Экстренное выключение"),
        ]
        
        for key, desc in shortcuts:
            row = QHBoxLayout()
            row.setSpacing(12)
            
            key_label = QLabel(f"[{key}]")
            key_label.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
            key_label.setStyleSheet(f"color: {DARK_COLORS['accent']}; background-color: {DARK_COLORS['bg_card']}; padding: 4px 10px; border-radius: 6px;")
            row.addWidget(key_label)
            
            desc_label = QLabel(desc)
            desc_label.setFont(QFont("Segoe UI", 12))
            desc_label.setStyleSheet(f"color: {DARK_COLORS['text_secondary']};")
            row.addWidget(desc_label)
            row.addStretch()
            shortcuts_layout.addLayout(row)
        
        layout.addWidget(shortcuts_frame)
        
        tips_frame = QFrame()
        tips_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {DARK_COLORS['bg_input']};
                border-radius: 10px;
                border: none;
            }}
        """)
        tips_layout = QVBoxLayout(tips_frame)
        tips_layout.setContentsMargins(14, 12, 14, 12)
        tips_layout.setSpacing(6)

        tips_title = QLabel("💡  Советы")
        tips_title.setFont(QFont("Segoe UI", 12, QFont.Weight.DemiBold))
        tips_title.setStyleSheet(f"color: {DARK_COLORS['accent']}; background: transparent;")
        tips_layout.addWidget(tips_title)

        tips = [
            "Держите руку на расстоянии 30–60 см от камеры",
            "Для прокрутки: три пальца вверх → листает вверх, вниз → вниз",
            "Чем выше чувствительность — тем меньше движений нужно",
            "Нажмите G для быстрого вкл/выкл управления мышью",
        ]
        for tip in tips:
            lbl = QLabel(f"• {tip}")
            lbl.setFont(QFont("Segoe UI", 11))
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color: {DARK_COLORS['text_muted']}; background: transparent;")
            tips_layout.addWidget(lbl)

        layout.addWidget(tips_frame)
        
        btn_close = QPushButton("Закрыть")
        btn_close.setFixedHeight(46)
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close.setStyleSheet(f"""
            QPushButton {{
                background-color: {DARK_COLORS['accent']};
                color: #FFFFFF;
                border: none;
                border-radius: 10px;
                padding: 10px 24px;
                font-size: 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {DARK_COLORS['accent_hover']};
            }}
        """)
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close)

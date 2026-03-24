from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QFrame, QHBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from neurofocus.ui.theme import theme_manager


class GestureHelpWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Управление жестами")
        self.setFixedSize(520, 480)
        self.setModal(True)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {theme_manager.colors['bg_card']};
                border-radius: 16px;
                border: none;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(20)
        
        header = QLabel("🖱️  Управление жестами")
        header.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        header.setStyleSheet(f"color: {theme_manager.colors['text_primary']};")
        layout.addWidget(header)
        
        gestures_frame = QFrame()
        gestures_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {theme_manager.colors['bg_input']};
                border-radius: 12px;
                border: none;
            }}
        """)
        gestures_layout = QVBoxLayout(gestures_frame)
        gestures_layout.setContentsMargins(16, 16, 16, 16)
        gestures_layout.setSpacing(14)
        
        gestures = [
            ("☝️", "Указательный палец", "Движение курсором"),
            ("🤏", "Пинч (большой + указательный)", "Левый клик"),
            ("✌️", "Указательный + средний", "Правый клик"),
            ("✊", "Кулак (все пальцы вниз)", "Перетаскивание (drag)"),
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
            gesture_label.setStyleSheet(f"color: {theme_manager.colors['text_primary']};")
            text_col.addWidget(gesture_label)
            
            action_label = QLabel(action)
            action_label.setFont(QFont("Segoe UI", 12))
            action_label.setStyleSheet(f"color: {theme_manager.colors['text_secondary']};")
            text_col.addWidget(action_label)
            
            row.addLayout(text_col)
            row.addStretch()
            gestures_layout.addLayout(row)
        
        layout.addWidget(gestures_frame)
        
        shortcuts_frame = QFrame()
        shortcuts_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {theme_manager.colors['bg_input']};
                border-radius: 12px;
                border: none;
            }}
        """)
        shortcuts_layout = QVBoxLayout(shortcuts_frame)
        shortcuts_layout.setContentsMargins(16, 16, 16, 16)
        shortcuts_layout.setSpacing(10)
        
        shortcuts_title = QLabel("⌨️  Горячие клавиши")
        shortcuts_title.setFont(QFont("Segoe UI", 14, QFont.Weight.DemiBold))
        shortcuts_title.setStyleSheet(f"color: {theme_manager.colors['accent']};")
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
            key_label.setStyleSheet(f"color: {theme_manager.colors['accent']}; background-color: {theme_manager.colors['bg_card']}; padding: 4px 10px; border-radius: 6px;")
            row.addWidget(key_label)
            
            desc_label = QLabel(desc)
            desc_label.setFont(QFont("Segoe UI", 12))
            desc_label.setStyleSheet(f"color: {theme_manager.colors['text_secondary']};")
            row.addWidget(desc_label)
            row.addStretch()
            shortcuts_layout.addLayout(row)
        
        layout.addWidget(shortcuts_frame)
        
        tip_label = QLabel("💡 Для стабильной работы держите руку на расстоянии 30-60 см от камеры")
        tip_label.setFont(QFont("Segoe UI", 11))
        tip_label.setStyleSheet(f"color: {theme_manager.colors['text_muted']}; padding: 8px; background-color: {theme_manager.colors['bg_input']}; border-radius: 8px;")
        tip_label.setWordWrap(True)
        layout.addWidget(tip_label)
        
        btn_close = QPushButton("Закрыть")
        btn_close.setFixedHeight(46)
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close.setStyleSheet(f"""
            QPushButton {{
                background-color: {theme_manager.colors['accent']};
                color: #FFFFFF;
                border: none;
                border-radius: 10px;
                padding: 10px 24px;
                font-size: 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {theme_manager.colors['accent_hover']};
            }}
        """)
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close)
    
    def refresh_theme(self):
        c = theme_manager.colors
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {c['bg_card']};
                border-radius: 16px;
                border: none;
            }}
        """)
        for widget in self.findChildren(QLabel):
            if widget.font().pointSize() == 20:
                widget.setStyleSheet(f"color: {c['text_primary']};")
            elif widget.font().pointSize() == 14 and "[" in widget.text():
                widget.setStyleSheet(f"color: {c['accent']}; background-color: {c['bg_card']}; padding: 4px 10px; border-radius: 6px;")
        for widget in self.findChildren(QFrame):
            widget.setStyleSheet(f"""
                QFrame {{
                    background-color: {c['bg_input']};
                    border-radius: 12px;
                    border: none;
                }}
            """)

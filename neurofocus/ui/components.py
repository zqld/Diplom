from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QHBoxLayout, QProgressBar, QPushButton
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from neurofocus.ui.theme import theme_manager


class MetricCard(QFrame):
    def __init__(self, title: str, value: str = "--", unit: str = "", status: str = "normal", parent=None):
        super().__init__(parent)
        self.title_text = title
        self.value_text = value
        self.unit_text = unit
        self.status = status
        
        self.setFixedHeight(100)
        self.setMinimumWidth(150)
        self._update_style()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)
        
        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Medium))
        layout.addWidget(self.title_label)
        
        value_row = QHBoxLayout()
        value_row.setSpacing(8)
        
        self.value_label = QLabel(f"{value}{unit}")
        self.value_label.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        self.value_label.setMinimumWidth(140)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        value_row.addWidget(self.value_label)
        
        self.status_indicator = QLabel()
        self.status_indicator.setFixedSize(8, 8)
        value_row.addWidget(self.status_indicator)
        value_row.addStretch()
        
        layout.addLayout(value_row)
        
        self.update_status(status)
    
    def _update_style(self):
        c = theme_manager.colors
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {c['bg_card']};
                border: 1px solid {c['border']};
                border-radius: 12px;
            }}
        """)
        self.title_label.setStyleSheet(f"color: {c['text_muted']}; background: transparent;")
        self.value_label.setStyleSheet(f"color: {c['text_primary']}; background: transparent;")
    
    def update_style(self):
        self._update_style()
        self.update_status(self.status)
    
    def update_value(self, value: str, unit: str = None):
        self.value_label.setText(f"{value}{unit if unit else self.unit_text}")
    
    def update_status(self, status: str):
        self.status = status
        c = theme_manager.colors
        color = c.get('text_muted', '#6A6A7A')
        if status == "good":
            color = c.get('good', '#4ADE80')
        elif status == "warning":
            color = c.get('warning', '#FBBF24')
        elif status == "danger":
            color = c.get('danger', '#F87171')
        self.status_indicator.setStyleSheet(f"background-color: {color}; border-radius: 4px;")


class ProgressCard(MetricCard):
    def __init__(self, title: str, parent=None):
        super().__init__(title, "100", "%", "good", parent)
        
        c = theme_manager.colors
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                background-color: {c['bg_input']};
                border-radius: 4px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {c['accent']};
                border-radius: 4px;
            }}
        """)
        
        layout = self.layout()
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.progress_bar)
    
    def update_progress(self, value: int, status: str = None):
        self.progress_bar.setValue(max(0, min(100, value)))
        if status:
            self.update_status(status)


class ModernButton(QPushButton):
    def __init__(self, text: str, primary: bool = False, parent=None):
        super().__init__(text, parent)
        self.primary = primary
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(44)
        self._update_style()
    
    def _update_style(self):
        self.setStyleSheet(theme_manager.get_button_style(primary=self.primary))
    
    def update_style(self):
        self._update_style()


class ModernPrimaryButton(ModernButton):
    def __init__(self, text: str, parent=None):
        super().__init__(text, primary=True, parent=parent)
        self.setFixedHeight(48)


class DangerButton(ModernButton):
    def __init__(self, text: str, parent=None):
        super().__init__(text, primary=False, parent=parent)
        self.setStyleSheet(theme_manager.get_button_style(danger=True))


class StatusIndicator(QLabel):
    def __init__(self, status: str = "normal", parent=None):
        super().__init__(parent)
        self.status = status
        self.setFixedSize(10, 10)
        self.update_status(status)
    
    def update_status(self, status: str):
        self.status = status
        c = theme_manager.colors
        color = c.get('text_muted', '#6A6A7A')
        if status == "good":
            color = c.get('good', '#4ADE80')
        elif status == "warning":
            color = c.get('warning', '#FBBF24')
        elif status == "danger":
            color = c.get('danger', '#F87171')
        self.setStyleSheet(f"background-color: {color}; border-radius: 5px;")

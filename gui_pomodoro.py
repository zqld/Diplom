from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QPushButton, QFrame, QHBoxLayout)
from PyQt6.QtCore import QTimer, Qt
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
    'good': '#4ADE80',
    'warning': '#FBBF24',
    'danger': '#F87171',
    'border': '#3A3A45',
}


class PomodoroTimer(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pomodoro Таймер")
        self.setFixedSize(360, 400)
        self.setModal(False)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {DARK_COLORS['bg_card']};
                border-radius: 16px;
            }}
        """)
        
        self.work_minutes = 25
        self.break_minutes = 5
        self.is_work = True
        self.is_running = False
        self.remaining_time = self.work_minutes * 60
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(10)
        
        header = QLabel("🍅 Pomodoro")
        header.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        header.setStyleSheet(f"color: {DARK_COLORS['text_primary']};")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)
        
        self.timer_label = QLabel("25:00")
        self.timer_label.setFont(QFont("Segoe UI", 52, QFont.Weight.Bold))
        self.timer_label.setStyleSheet(f"color: {DARK_COLORS['accent']};")
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.timer_label)
        
        self.status_label = QLabel("Время работы")
        self.status_label.setFont(QFont("Segoe UI", 12))
        self.status_label.setStyleSheet(f"color: {DARK_COLORS['text_secondary']};")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        settings_frame = QFrame()
        settings_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {DARK_COLORS['bg_input']};
                border-radius: 10px;
            }}
        """)
        settings_layout = QHBoxLayout(settings_frame)
        settings_layout.setContentsMargins(16, 12, 16, 12)
        settings_layout.setSpacing(24)
        
        work_col = QVBoxLayout()
        work_col.setSpacing(4)
        
        work_title = QLabel("Работа")
        work_title.setFont(QFont("Segoe UI", 10))
        work_title.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
        work_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        work_col.addWidget(work_title)
        
        work_time_row = QHBoxLayout()
        work_time_row.setSpacing(6)
        
        self.btn_work_minus = QPushButton("−")
        self.btn_work_minus.setFixedSize(32, 30)
        self.btn_work_minus.setStyleSheet(f"""
            QPushButton {{
                background-color: {DARK_COLORS['bg_card']};
                color: {DARK_COLORS['text_primary']};
                border: 1px solid {DARK_COLORS['border']};
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                border-color: {DARK_COLORS['accent']};
            }}
        """)
        self.btn_work_minus.clicked.connect(lambda: self.change_time('work', -5))
        work_time_row.addWidget(self.btn_work_minus)
        
        self.work_val = QLabel("25")
        self.work_val.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.work_val.setStyleSheet(f"color: {DARK_COLORS['text_primary']};")
        self.work_val.setFixedWidth(28)
        self.work_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        work_time_row.addWidget(self.work_val)
        
        self.btn_work_plus = QPushButton("+")
        self.btn_work_plus.setFixedSize(32, 30)
        self.btn_work_plus.setStyleSheet(f"""
            QPushButton {{
                background-color: {DARK_COLORS['bg_card']};
                color: {DARK_COLORS['text_primary']};
                border: 1px solid {DARK_COLORS['border']};
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                border-color: {DARK_COLORS['accent']};
            }}
        """)
        self.btn_work_plus.clicked.connect(lambda: self.change_time('work', 5))
        work_time_row.addWidget(self.btn_work_plus)
        
        work_col.addLayout(work_time_row)
        
        work_unit = QLabel("мин")
        work_unit.setFont(QFont("Segoe UI", 9))
        work_unit.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
        work_unit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        work_col.addWidget(work_unit)
        
        settings_layout.addLayout(work_col)
        
        break_col = QVBoxLayout()
        break_col.setSpacing(4)
        
        break_title = QLabel("Отдых")
        break_title.setFont(QFont("Segoe UI", 10))
        break_title.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
        break_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        break_col.addWidget(break_title)
        
        break_time_row = QHBoxLayout()
        break_time_row.setSpacing(6)
        
        self.btn_break_minus = QPushButton("−")
        self.btn_break_minus.setFixedSize(32, 30)
        self.btn_break_minus.setStyleSheet(f"""
            QPushButton {{
                background-color: {DARK_COLORS['bg_card']};
                color: {DARK_COLORS['text_primary']};
                border: 1px solid {DARK_COLORS['border']};
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                border-color: {DARK_COLORS['accent']};
            }}
        """)
        self.btn_break_minus.clicked.connect(lambda: self.change_time('break', -1))
        break_time_row.addWidget(self.btn_break_minus)
        
        self.break_val = QLabel("5")
        self.break_val.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.break_val.setStyleSheet(f"color: {DARK_COLORS['text_primary']};")
        self.break_val.setFixedWidth(28)
        self.break_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        break_time_row.addWidget(self.break_val)
        
        self.btn_break_plus = QPushButton("+")
        self.btn_break_plus.setFixedSize(32, 30)
        self.btn_break_plus.setStyleSheet(f"""
            QPushButton {{
                background-color: {DARK_COLORS['bg_card']};
                color: {DARK_COLORS['text_primary']};
                border: 1px solid {DARK_COLORS['border']};
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                border-color: {DARK_COLORS['accent']};
            }}
        """)
        self.btn_break_plus.clicked.connect(lambda: self.change_time('break', 1))
        break_time_row.addWidget(self.btn_break_plus)
        
        break_col.addLayout(break_time_row)
        
        break_unit = QLabel("мин")
        break_unit.setFont(QFont("Segoe UI", 9))
        break_unit.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
        break_unit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        break_col.addWidget(break_unit)
        
        settings_layout.addLayout(break_col)
        
        main_layout.addWidget(settings_frame)
        
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(12)
        
        self.btn_reset = QPushButton("↺")
        self.btn_reset.setFixedSize(44, 44)
        self.btn_reset.setStyleSheet(f"""
            QPushButton {{
                background-color: {DARK_COLORS['bg_input']};
                color: {DARK_COLORS['text_secondary']};
                border: 1px solid {DARK_COLORS['border']};
                border-radius: 10px;
                font-size: 18px;
            }}
            QPushButton:hover {{
                border-color: {DARK_COLORS['accent']};
            }}
        """)
        self.btn_reset.clicked.connect(self.reset_timer)
        buttons_layout.addWidget(self.btn_reset)
        
        self.btn_toggle = QPushButton("▶ Старт")
        self.btn_toggle.setFixedHeight(44)
        self.btn_toggle.setStyleSheet(f"""
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
        self.btn_toggle.clicked.connect(self.toggle_timer)
        buttons_layout.addWidget(self.btn_toggle)
        
        main_layout.addLayout(buttons_layout)
        
        main_layout.addStretch()
        
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tick)
    
    def change_time(self, type_, delta):
        if self.is_running:
            return
        
        if type_ == 'work':
            new_val = self.work_minutes + delta
            if 5 <= new_val <= 60:
                self.work_minutes = new_val
                self.work_val.setText(str(new_val))
                if self.is_work:
                    self.remaining_time = new_val * 60
                    self.timer_label.setText(self._format_time(self.remaining_time))
        else:
            new_val = self.break_minutes + delta
            if 1 <= new_val <= 30:
                self.break_minutes = new_val
                self.break_val.setText(str(new_val))
                if not self.is_work:
                    self.remaining_time = new_val * 60
                    self.timer_label.setText(self._format_time(self.remaining_time))
    
    def _format_time(self, seconds):
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins:02d}:{secs:02d}"
    
    def toggle_timer(self):
        if self.is_running:
            self.timer.stop()
            self.is_running = False
            self.btn_toggle.setText("▶ Старт")
        else:
            self.timer.start()
            self.is_running = True
            self.btn_toggle.setText("⏸ Пауза")
    
    def reset_timer(self):
        self.timer.stop()
        self.is_running = False
        self.is_work = True
        self.remaining_time = self.work_minutes * 60
        self.timer_label.setText(self._format_time(self.remaining_time))
        self.timer_label.setStyleSheet(f"color: {DARK_COLORS['accent']};")
        self.status_label.setText("Время работы")
        self.status_label.setStyleSheet(f"color: {DARK_COLORS['text_secondary']};")
        self.btn_toggle.setText("▶ Старт")
    
    def tick(self):
        self.remaining_time -= 1
        
        if self.remaining_time <= 0:
            self.timer.stop()
            self.is_running = False
            
            if self.is_work:
                self.status_label.setText("⏰ Перерыв!")
                self.status_label.setStyleSheet(f"color: {DARK_COLORS['good']};")
                self.remaining_time = self.break_minutes * 60
                self.is_work = False
                self.timer_label.setStyleSheet(f"color: {DARK_COLORS['good']};")
            else:
                self.status_label.setText("✅ За работу!")
                self.status_label.setStyleSheet(f"color: {DARK_COLORS['accent']};")
                self.remaining_time = self.work_minutes * 60
                self.is_work = True
                self.timer_label.setStyleSheet(f"color: {DARK_COLORS['accent']};")
            
            self.btn_toggle.setText("▶ Старт")
        else:
            self.timer_label.setText(self._format_time(self.remaining_time))
    
    def closeEvent(self, event):
        self.timer.stop()
        event.accept()

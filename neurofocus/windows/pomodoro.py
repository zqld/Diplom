from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QPushButton, QFrame, QHBoxLayout)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont
from neurofocus.ui.theme import theme_manager


class PomodoroTimer(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pomodoro Таймер")
        self.setFixedSize(360, 480)
        self.setModal(False)
        self._apply_theme()
    
    def _apply_theme(self):
        c = theme_manager.colors
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {c['bg_card']};
                border-radius: 16px;
            }}
            QLabel {{
                color: {c['text_primary']};
                background: transparent;
            }}
        """)
        
        self.work_minutes = 25
        self.break_minutes = 5
        self.long_break_minutes = 15
        self.cycles_before_long_break = 4
        self.is_work = True
        self.is_running = False
        self.remaining_time = self.work_minutes * 60
        self.completed_cycles = 0
        
        self.sound_manager = None
        self._init_sound()
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(10)
        
        header = QLabel("🍅 Pomodoro")
        header.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        header.setStyleSheet(f"color: {theme_manager.colors['text_primary']};")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)
        
        self.cycle_label = QLabel(f"Цикл: {self.completed_cycles}/{self.cycles_before_long_break}")
        self.cycle_label.setFont(QFont("Segoe UI", 10))
        self.cycle_label.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
        self.cycle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.cycle_label)
        
        self.timer_label = QLabel("25:00")
        self.timer_label.setFont(QFont("Segoe UI", 52, QFont.Weight.Bold))
        self.timer_label.setStyleSheet(f"color: {theme_manager.colors['accent']};")
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.timer_label)
        
        self.status_label = QLabel("Время работы")
        self.status_label.setFont(QFont("Segoe UI", 12))
        self.status_label.setStyleSheet(f"color: {theme_manager.colors['text_secondary']};")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        settings_frame = QFrame()
        settings_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {theme_manager.colors['bg_input']};
                border-radius: 10px;
            }}
        """)
        settings_layout = QVBoxLayout(settings_frame)
        settings_layout.setContentsMargins(16, 12, 16, 12)
        settings_layout.setSpacing(12)
        
        work_break_row = QHBoxLayout()
        work_break_row.setSpacing(24)
        
        work_col = QVBoxLayout()
        work_col.setSpacing(4)
        
        work_title = QLabel("Работа")
        work_title.setFont(QFont("Segoe UI", 10))
        work_title.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
        work_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        work_col.addWidget(work_title)
        
        work_time_row = QHBoxLayout()
        work_time_row.setSpacing(6)
        
        btn_work_minus = QPushButton("−")
        btn_work_minus.setFixedSize(28, 28)
        btn_work_minus.setStyleSheet(f"""
            QPushButton {{
                background-color: {theme_manager.colors['bg_card']};
                color: {theme_manager.colors['text_primary']};
                border: 1px solid {theme_manager.colors['border']};
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                border-color: {theme_manager.colors['accent']};
            }}
        """)
        btn_work_minus.clicked.connect(lambda: self.change_time('work', -5))
        work_time_row.addWidget(btn_work_minus)
        
        self.work_val = QLabel("25")
        self.work_val.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.work_val.setStyleSheet(f"color: {theme_manager.colors['text_primary']};")
        self.work_val.setFixedWidth(28)
        self.work_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        work_time_row.addWidget(self.work_val)
        
        btn_work_plus = QPushButton("+")
        btn_work_plus.setFixedSize(28, 28)
        btn_work_plus.setStyleSheet(f"""
            QPushButton {{
                background-color: {theme_manager.colors['bg_card']};
                color: {theme_manager.colors['text_primary']};
                border: 1px solid {theme_manager.colors['border']};
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                border-color: {theme_manager.colors['accent']};
            }}
        """)
        btn_work_plus.clicked.connect(lambda: self.change_time('work', 5))
        work_time_row.addWidget(btn_work_plus)
        
        work_col.addLayout(work_time_row)
        
        work_unit = QLabel("мин")
        work_unit.setFont(QFont("Segoe UI", 9))
        work_unit.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
        work_unit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        work_col.addWidget(work_unit)
        
        work_break_row.addLayout(work_col)
        
        break_col = QVBoxLayout()
        break_col.setSpacing(4)
        
        break_title = QLabel("Перерыв")
        break_title.setFont(QFont("Segoe UI", 10))
        break_title.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
        break_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        break_col.addWidget(break_title)
        
        break_time_row = QHBoxLayout()
        break_time_row.setSpacing(6)
        
        btn_break_minus = QPushButton("−")
        btn_break_minus.setFixedSize(28, 28)
        btn_break_minus.setStyleSheet(f"""
            QPushButton {{
                background-color: {theme_manager.colors['bg_card']};
                color: {theme_manager.colors['text_primary']};
                border: 1px solid {theme_manager.colors['border']};
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                border-color: {theme_manager.colors['accent']};
            }}
        """)
        btn_break_minus.clicked.connect(lambda: self.change_time('break', -1))
        break_time_row.addWidget(btn_break_minus)
        
        self.break_val = QLabel("5")
        self.break_val.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.break_val.setStyleSheet(f"color: {theme_manager.colors['text_primary']};")
        self.break_val.setFixedWidth(28)
        self.break_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        break_time_row.addWidget(self.break_val)
        
        btn_break_plus = QPushButton("+")
        btn_break_plus.setFixedSize(28, 28)
        btn_break_plus.setStyleSheet(f"""
            QPushButton {{
                background-color: {theme_manager.colors['bg_card']};
                color: {theme_manager.colors['text_primary']};
                border: 1px solid {theme_manager.colors['border']};
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                border-color: {theme_manager.colors['accent']};
            }}
        """)
        btn_break_plus.clicked.connect(lambda: self.change_time('break', 1))
        break_time_row.addWidget(btn_break_plus)
        
        break_col.addLayout(break_time_row)
        
        break_unit = QLabel("мин")
        break_unit.setFont(QFont("Segoe UI", 9))
        break_unit.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
        break_unit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        break_col.addWidget(break_unit)
        
        work_break_row.addLayout(break_col)
        
        settings_layout.addLayout(work_break_row)
        
        long_break_row = QHBoxLayout()
        long_break_row.setSpacing(8)
        
        long_break_title = QLabel("Длинный перерыв:")
        long_break_title.setFont(QFont("Segoe UI", 10))
        long_break_title.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
        long_break_row.addWidget(long_break_title)
        
        btn_long_minus = QPushButton("−")
        btn_long_minus.setFixedSize(28, 28)
        btn_long_minus.setStyleSheet(f"""
            QPushButton {{
                background-color: {theme_manager.colors['bg_card']};
                color: {theme_manager.colors['text_primary']};
                border: 1px solid {theme_manager.colors['border']};
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                border-color: {theme_manager.colors['accent']};
            }}
        """)
        btn_long_minus.clicked.connect(lambda: self.change_time('long_break', -5))
        long_break_row.addWidget(btn_long_minus)
        
        self.long_break_val = QLabel("15")
        self.long_break_val.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.long_break_val.setStyleSheet(f"color: {theme_manager.colors['text_primary']};")
        self.long_break_val.setFixedWidth(28)
        self.long_break_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        long_break_row.addWidget(self.long_break_val)
        
        btn_long_plus = QPushButton("+")
        btn_long_plus.setFixedSize(28, 28)
        btn_long_plus.setStyleSheet(f"""
            QPushButton {{
                background-color: {theme_manager.colors['bg_card']};
                color: {theme_manager.colors['text_primary']};
                border: 1px solid {theme_manager.colors['border']};
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                border-color: {theme_manager.colors['accent']};
            }}
        """)
        btn_long_plus.clicked.connect(lambda: self.change_time('long_break', 5))
        long_break_row.addWidget(btn_long_plus)
        
        long_break_unit = QLabel("мин")
        long_break_unit.setFont(QFont("Segoe UI", 9))
        long_break_unit.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
        long_break_row.addWidget(long_break_unit)
        
        long_break_row.addStretch()
        
        settings_layout.addLayout(long_break_row)
        
        main_layout.addWidget(settings_frame)
        
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(12)
        
        self.btn_reset = QPushButton("↺")
        self.btn_reset.setFixedSize(44, 44)
        self.btn_reset.setStyleSheet(f"""
            QPushButton {{
                background-color: {theme_manager.colors['bg_input']};
                color: {theme_manager.colors['text_secondary']};
                border: 1px solid {theme_manager.colors['border']};
                border-radius: 10px;
                font-size: 18px;
            }}
            QPushButton:hover {{
                border-color: {theme_manager.colors['accent']};
            }}
        """)
        self.btn_reset.clicked.connect(self.reset_timer)
        buttons_layout.addWidget(self.btn_reset)
        
        self.btn_toggle = QPushButton("▶ Старт")
        self.btn_toggle.setFixedHeight(44)
        self.btn_toggle.setStyleSheet(f"""
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
        self.btn_toggle.clicked.connect(self.toggle_timer)
        buttons_layout.addWidget(self.btn_toggle)
        
        main_layout.addLayout(buttons_layout)
        
        self.message_label = QLabel("")
        self.message_label.setFont(QFont("Segoe UI", 10))
        self.message_label.setStyleSheet(f"color: {theme_manager.colors['warning']};")
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message_label.hide()
        main_layout.addWidget(self.message_label)
        
        main_layout.addStretch()
        
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tick)
    
    def _init_sound(self):
        try:
            from neurofocus.utils.sound import sound_manager
            self.sound_manager = sound_manager
        except ImportError:
            self.sound_manager = None
    
    def _play_sound(self, sound_type):
        if self.sound_manager:
            try:
                if sound_type == 'work':
                    self.sound_manager.pomodoro_work()
                elif sound_type == 'break':
                    self.sound_manager.pomodoro_break()
                elif sound_type == 'long_break':
                    self.sound_manager.pomodoro_long_break()
            except AttributeError:
                pass
    
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
        elif type_ == 'break':
            new_val = self.break_minutes + delta
            if 1 <= new_val <= 30:
                self.break_minutes = new_val
                self.break_val.setText(str(new_val))
        else:
            new_val = self.long_break_minutes + delta
            if 5 <= new_val <= 60:
                self.long_break_minutes = new_val
                self.long_break_val.setText(str(new_val))
    
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
            self.message_label.hide()
    
    def reset_timer(self):
        self.timer.stop()
        self.is_running = False
        self.is_work = True
        self.completed_cycles = 0
        self.remaining_time = self.work_minutes * 60
        self.timer_label.setText(self._format_time(self.remaining_time))
        self.timer_label.setStyleSheet(f"color: {theme_manager.colors['accent']};")
        self.status_label.setText("Время работы")
        self.status_label.setStyleSheet(f"color: {theme_manager.colors['text_secondary']};")
        self.btn_toggle.setText("▶ Старт")
        self.cycle_label.setText(f"Цикл: {self.completed_cycles}/{self.cycles_before_long_break}")
        self.message_label.hide()
    
    def suggest_break(self, reason=""):
        if self.is_work and not self.is_running:
            self.message_label.setText(f"⚠ Рекомендуется перерыв: {reason}" if reason else "⚠ Рекомендуется перерыв")
            self.message_label.show()
        elif self.is_work and self.is_running:
            self.message_label.setText(f"⚠ {reason}" if reason else "⚠ Усталость detected!")
            self.message_label.show()
    
    def tick(self):
        self.remaining_time -= 1
        
        if self.remaining_time <= 0:
            self.timer.stop()
            self.is_running = False
            
            if self.is_work:
                self.completed_cycles += 1
                self.cycle_label.setText(f"Цикл: {self.completed_cycles}/{self.cycles_before_long_break}")
                
                if self.completed_cycles >= self.cycles_before_long_break:
                    self.status_label.setText("☕ Длинный перерыв!")
                    self.status_label.setStyleSheet(f"color: {theme_manager.colors['good']};")
                    self.remaining_time = self.long_break_minutes * 60
                    self.is_work = False
                    self.timer_label.setStyleSheet(f"color: {theme_manager.colors['good']};")
                    self.completed_cycles = 0
                    self._play_sound('long_break')
                else:
                    self.status_label.setText("⏰ Перерыв!")
                    self.status_label.setStyleSheet(f"color: {theme_manager.colors['good']};")
                    self.remaining_time = self.break_minutes * 60
                    self.is_work = False
                    self.timer_label.setStyleSheet(f"color: {theme_manager.colors['good']};")
                    self._play_sound('break')
            else:
                self.status_label.setText("✅ За работу!")
                self.status_label.setStyleSheet(f"color: {theme_manager.colors['accent']};")
                self.remaining_time = self.work_minutes * 60
                self.is_work = True
                self.timer_label.setStyleSheet(f"color: {theme_manager.colors['accent']};")
                self._play_sound('work')
            
            self.btn_toggle.setText("▶ Старт")
        else:
            self.timer_label.setText(self._format_time(self.remaining_time))
    
    def closeEvent(self, event):
        self.timer.stop()
        event.accept()

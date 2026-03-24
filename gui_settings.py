from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QPushButton,
                             QFrame, QHBoxLayout, QSlider, QCheckBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from src.notifications import DEFAULT_SETTINGS


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
    'border_light': '#4A4A55',
    'good': '#4ADE80',
    'warning': '#FBBF24',
    'danger': '#F87171',
}


class NumberStepper(QFrame):
    def __init__(self, min_val, max_val, value, suffix="", parent=None):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.current_value = value
        self.suffix = suffix
        
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {DARK_COLORS['bg_input']};
                border-radius: 8px;
                border: 1px solid {DARK_COLORS['border']};
            }}
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)
        
        self.btn_minus = QPushButton("−")
        self.btn_minus.setFixedSize(36, 36)
        self.btn_minus.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_minus.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {DARK_COLORS['text_secondary']};
                border: none;
                border-radius: 6px;
                font-size: 18px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {DARK_COLORS['border']};
                color: {DARK_COLORS['text_primary']};
            }}
        """)
        self.btn_minus.clicked.connect(self.decrease)
        layout.addWidget(self.btn_minus)
        
        self.value_label = QLabel(f"{value}{suffix}")
        self.value_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.value_label.setStyleSheet(f"color: {DARK_COLORS['text_primary']}; background: transparent;")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_label.setMinimumWidth(60)
        layout.addWidget(self.value_label)
        
        self.btn_plus = QPushButton("+")
        self.btn_plus.setFixedSize(36, 36)
        self.btn_plus.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_plus.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {DARK_COLORS['text_secondary']};
                border: none;
                border-radius: 6px;
                font-size: 18px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {DARK_COLORS['border']};
                color: {DARK_COLORS['text_primary']};
            }}
        """)
        self.btn_plus.clicked.connect(self.increase)
        layout.addWidget(self.btn_plus)
    
    def increase(self):
        if self.current_value < self.max_val:
            self.current_value += 1
            self.value_label.setText(f"{self.current_value}{self.suffix}")
    
    def decrease(self):
        if self.current_value > self.min_val:
            self.current_value -= 1
            self.value_label.setText(f"{self.current_value}{self.suffix}")
    
    def value(self):
        return self.current_value


class SettingsWindow(QDialog):
    def __init__(self, current_settings=None, calibration_manager=None):
        super().__init__()
        self.setWindowTitle("Настройки")
        self.resize(480, 700)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {DARK_COLORS['bg_main']};
                font-family: 'Segoe UI', sans-serif;
            }}
            QLabel {{
                font-family: 'Segoe UI', sans-serif;
                color: {DARK_COLORS['text_primary']};
            }}
            QSlider::groove:horizontal {{
                border: none;
                height: 6px;
                background: {DARK_COLORS['border']};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {DARK_COLORS['accent']};
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {DARK_COLORS['accent_hover']};
            }}
            QCheckBox {{
                color: {DARK_COLORS['text_primary']};
                spacing: 12px;
                font-size: 14px;
            }}
            QCheckBox::indicator {{
                width: 20px;
                height: 20px;
                border-radius: 6px;
                border: 2px solid {DARK_COLORS['border']};
                background: {DARK_COLORS['bg_input']};
            }}
            QCheckBox::indicator:checked {{
                background: {DARK_COLORS['accent']};
                border-color: {DARK_COLORS['accent']};
            }}
        """)

        self.settings = current_settings if current_settings else DEFAULT_SETTINGS.copy()
        self.calibration = calibration_manager
        self.result_settings = None

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(28, 24, 28, 24)
        main_layout.setSpacing(20)

        header = QLabel("Настройки")
        header.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        header.setStyleSheet(f"color: {DARK_COLORS['text_primary']};")
        main_layout.addWidget(header)

        content_layout = QVBoxLayout()
        content_layout.setSpacing(18)

        section_mouse = QLabel("🖱️  Управление мышью")
        section_mouse.setFont(QFont("Segoe UI", 14, QFont.Weight.DemiBold))
        section_mouse.setStyleSheet(f"color: {DARK_COLORS['accent']};")
        content_layout.addWidget(section_mouse)

        sens_row = QHBoxLayout()
        sens_row.setSpacing(16)
        sens_label = QLabel("Чувствительность")
        sens_label.setFont(QFont("Segoe UI", 13))
        sens_label.setStyleSheet(f"color: {DARK_COLORS['text_secondary']};")
        sens_row.addWidget(sens_label)
        
        self.sens_value = QLabel("1.0x")
        self.sens_value.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        self.sens_value.setStyleSheet(f"color: {DARK_COLORS['accent']};")
        self.sens_value.setFixedWidth(45)
        sens_row.addWidget(self.sens_value)
        sens_row.addStretch()
        content_layout.addLayout(sens_row)

        self.sens_slider = QSlider(Qt.Orientation.Horizontal)
        self.sens_slider.setRange(30, 300)
        self.sens_slider.setValue(int((self.calibration.sensitivity if self.calibration else 1.0) * 100))
        self.sens_slider.valueChanged.connect(
            lambda v: self.sens_value.setText(f"{v/100:.1f}x")
        )
        content_layout.addWidget(self.sens_slider)

        sens_hint = QLabel("Чем меньше значение, тем меньше движений рукой")
        sens_hint.setFont(QFont("Segoe UI", 11))
        sens_hint.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
        content_layout.addWidget(sens_hint)

        separator1 = QFrame()
        separator1.setFixedHeight(1)
        separator1.setStyleSheet(f"background-color: {DARK_COLORS['border']};")
        content_layout.addWidget(separator1)

        section_calib = QLabel("⚙️  Калибровка")
        section_calib.setFont(QFont("Segoe UI", 14, QFont.Weight.DemiBold))
        section_calib.setStyleSheet(f"color: {DARK_COLORS['accent']};")
        content_layout.addWidget(section_calib)

        self.auto_check = QCheckBox("Автоматическая калибровка при запуске")
        self.auto_check.setChecked(self.calibration.auto_calibrate if self.calibration else False)
        content_layout.addWidget(self.auto_check)

        face_row = QHBoxLayout()
        face_row.setSpacing(16)
        face_label = QLabel("Калибровка лица")
        face_label.setFont(QFont("Segoe UI", 13))
        face_label.setStyleSheet(f"color: {DARK_COLORS['text_secondary']};")
        face_row.addWidget(face_label)
        face_row.addStretch()
        
        self.face_status = QLabel("Не выполнена" if not (self.calibration and self.calibration.face_calibration["calibrated"]) else "Выполнена ✓")
        self.face_status.setFont(QFont("Segoe UI", 12))
        if self.calibration and self.calibration.face_calibration["calibrated"]:
            self.face_status.setStyleSheet(f"color: {DARK_COLORS['good']}; font-weight: 600;")
        else:
            self.face_status.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
        face_row.addWidget(self.face_status)
        
        self.btn_face = QPushButton("Начать")
        self.btn_face.setFixedSize(90, 38)
        self.btn_face.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_face.setStyleSheet(f"""
            QPushButton {{
                background-color: {DARK_COLORS['bg_input']};
                color: {DARK_COLORS['accent']};
                border: 1px solid {DARK_COLORS['accent']};
                border-radius: 8px;
                padding: 6px 16px;
                font-size: 13px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {DARK_COLORS['accent']};
                color: #FFFFFF;
            }}
        """)
        self.btn_face.clicked.connect(self.start_face_calibration)
        face_row.addWidget(self.btn_face)
        content_layout.addLayout(face_row)

        hand_row = QHBoxLayout()
        hand_row.setSpacing(16)
        hand_label = QLabel("Калибровка руки")
        hand_label.setFont(QFont("Segoe UI", 13))
        hand_label.setStyleSheet(f"color: {DARK_COLORS['text_secondary']};")
        hand_row.addWidget(hand_label)
        hand_row.addStretch()
        
        self.hand_status = QLabel("Не выполнена" if not (self.calibration and self.calibration.hand_calibration["calibrated"]) else "Выполнена ✓")
        self.hand_status.setFont(QFont("Segoe UI", 12))
        if self.calibration and self.calibration.hand_calibration["calibrated"]:
            self.hand_status.setStyleSheet(f"color: {DARK_COLORS['good']}; font-weight: 600;")
        else:
            self.hand_status.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
        hand_row.addWidget(self.hand_status)
        
        self.btn_hand = QPushButton("Начать")
        self.btn_hand.setFixedSize(90, 38)
        self.btn_hand.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_hand.setStyleSheet(f"""
            QPushButton {{
                background-color: {DARK_COLORS['bg_input']};
                color: {DARK_COLORS['accent']};
                border: 1px solid {DARK_COLORS['accent']};
                border-radius: 8px;
                padding: 6px 16px;
                font-size: 13px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {DARK_COLORS['accent']};
                color: #FFFFFF;
            }}
        """)
        self.btn_hand.clicked.connect(self.start_hand_calibration)
        hand_row.addWidget(self.btn_hand)
        content_layout.addLayout(hand_row)

        separator2 = QFrame()
        separator2.setFixedHeight(1)
        separator2.setStyleSheet(f"background-color: {DARK_COLORS['border']};")
        content_layout.addWidget(separator2)

        section_notify = QLabel("🔔  Уведомления")
        section_notify.setFont(QFont("Segoe UI", 14, QFont.Weight.DemiBold))
        section_notify.setStyleSheet(f"color: {DARK_COLORS['accent']};")
        content_layout.addWidget(section_notify)

        work_row = QHBoxLayout()
        work_row.setSpacing(16)
        work_label = QLabel("Перерыв каждые")
        work_label.setFont(QFont("Segoe UI", 13))
        work_label.setStyleSheet(f"color: {DARK_COLORS['text_secondary']};")
        work_row.addWidget(work_label)
        work_row.addStretch()
        
        self.stepper_work = NumberStepper(1, 240, self.settings["work_limit_minutes"], " мин")
        work_row.addWidget(self.stepper_work)
        content_layout.addLayout(work_row)

        posture_row = QHBoxLayout()
        posture_row.setSpacing(16)
        posture_label = QLabel("Чувствительность осанки")
        posture_label.setFont(QFont("Segoe UI", 13))
        posture_label.setStyleSheet(f"color: {DARK_COLORS['text_secondary']};")
        posture_row.addWidget(posture_label)
        posture_row.addStretch()
        
        self.stepper_posture = NumberStepper(5, 100, self.settings["posture_bad_percent"], "%")
        posture_row.addWidget(self.stepper_posture)
        content_layout.addLayout(posture_row)

        yawn_row = QHBoxLayout()
        yawn_row.setSpacing(16)
        yawn_label = QLabel("Лимит зевков (за 10 мин)")
        yawn_label.setFont(QFont("Segoe UI", 13))
        yawn_label.setStyleSheet(f"color: {DARK_COLORS['text_secondary']};")
        yawn_row.addWidget(yawn_label)
        yawn_row.addStretch()
        
        self.stepper_yawn = NumberStepper(1, 10, self.settings["yawn_limit"], "")
        yawn_row.addWidget(self.stepper_yawn)
        content_layout.addLayout(yawn_row)

        content_layout.addStretch()
        main_layout.addLayout(content_layout)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(16)
        btn_layout.addStretch()
        
        self.btn_cancel = QPushButton("Отмена")
        self.btn_cancel.setFixedSize(120, 46)
        self.btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cancel.setStyleSheet(f"""
            QPushButton {{
                background-color: {DARK_COLORS['bg_card']};
                color: {DARK_COLORS['text_secondary']};
                border: 1px solid {DARK_COLORS['border']};
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                border-color: {DARK_COLORS['border_light']};
                color: {DARK_COLORS['text_primary']};
            }}
        """)
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        self.btn_save = QPushButton("Сохранить")
        self.btn_save.setFixedSize(130, 46)
        self.btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save.setStyleSheet(f"""
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
        self.btn_save.clicked.connect(self.save_settings)
        btn_layout.addWidget(self.btn_save)

        main_layout.addLayout(btn_layout)

    def start_face_calibration(self):
        self.face_status.setText("Сбор данных...")
        self.face_status.setStyleSheet(f"color: {DARK_COLORS['warning']}; font-weight: 600;")
        if self.calibration:
            self.calibration.start_face_calibration()

    def start_hand_calibration(self):
        self.hand_status.setText("Сбор данных...")
        self.hand_status.setStyleSheet(f"color: {DARK_COLORS['warning']}; font-weight: 600;")
        if self.calibration:
            self.calibration.start_hand_calibration()

    def save_settings(self):
        sensitivity = self.sens_slider.value() / 100.0
        
        if self.calibration:
            self.calibration.set_sensitivity(sensitivity)
            self.calibration.set_auto_calibrate(self.auto_check.isChecked())
            
            if self.calibration.face_calibration["calibrated"]:
                self.face_status.setText("Выполнена ✓")
                self.face_status.setStyleSheet(f"color: {DARK_COLORS['good']}; font-weight: 600;")
            
            if self.calibration.hand_calibration["calibrated"]:
                self.hand_status.setText("Выполнена ✓")
                self.hand_status.setStyleSheet(f"color: {DARK_COLORS['good']}; font-weight: 600;")
        
        self.result_settings = {
            "work_limit_minutes": self.stepper_work.value(),
            "posture_window_minutes": 3,
            "posture_bad_percent": self.stepper_posture.value(),
            "yawn_limit": self.stepper_yawn.value(),
            "yawn_window_minutes": 10,
            "gesture_sensitivity": sensitivity,
        }
        self.accept()

    def get_settings(self):
        return self.result_settings

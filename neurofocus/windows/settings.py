from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QPushButton,
                             QFrame, QHBoxLayout, QSlider, QCheckBox, QComboBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from neurofocus.core.settings_manager import settings_manager
from neurofocus.ui.theme import theme_manager
from neurofocus.utils.sound import sound_manager


class NumberStepper(QFrame):
    def __init__(self, min_val, max_val, value, suffix="", parent=None):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.current_value = value
        self.suffix = suffix
        self._update_style()
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)
        
        self.btn_minus = QPushButton("−")
        self.btn_minus.setFixedSize(36, 36)
        self.btn_minus.setCursor(Qt.CursorShape.PointingHandCursor)
        self._style_button(self.btn_minus)
        self.btn_minus.clicked.connect(self.decrease)
        layout.addWidget(self.btn_minus)
        
        self.value_label = QLabel(f"{value}{suffix}")
        self.value_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.value_label.setStyleSheet(f"color: {theme_manager.colors['text_primary']}; background: transparent;")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_label.setMinimumWidth(60)
        layout.addWidget(self.value_label)
        
        self.btn_plus = QPushButton("+")
        self.btn_plus.setFixedSize(36, 36)
        self.btn_plus.setCursor(Qt.CursorShape.PointingHandCursor)
        self._style_button(self.btn_plus)
        self.btn_plus.clicked.connect(self.increase)
        layout.addWidget(self.btn_plus)
    
    def _update_style(self):
        c = theme_manager.colors
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {c['bg_input']};
                border-radius: 8px;
                border: 1px solid {c['border']};
            }}
        """)
    
    def _style_button(self, btn):
        c = theme_manager.colors
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {c['text_secondary']};
                border: none;
                border-radius: 6px;
                font-size: 18px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {c['border']};
                color: {c['text_primary']};
            }}
        """)
    
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
    
    def update_style(self):
        self._update_style()
        self._style_button(self.btn_minus)
        self._style_button(self.btn_plus)
        self.value_label.setStyleSheet(f"color: {theme_manager.colors['text_primary']}; background: transparent;")


class SettingsWindow(QDialog):
    def __init__(self, current_settings=None, calibration_manager=None):
        super().__init__()
        self.setWindowTitle("Настройки")
        self.resize(480, 780)
        self._apply_theme_style()
        
        self.settings = settings_manager.settings.copy()
        self.calibration = calibration_manager
        self.result_settings = None

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(28, 24, 28, 24)
        main_layout.setSpacing(20)

        header = QLabel("Настройки")
        header.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        header.setStyleSheet(f"color: {theme_manager.colors['text_primary']};")
        main_layout.addWidget(header)

        content_layout = QVBoxLayout()
        content_layout.setSpacing(18)
        
        section_appearance = QLabel("🎨  Внешний вид")
        section_appearance.setFont(QFont("Segoe UI", 14, QFont.Weight.DemiBold))
        section_appearance.setStyleSheet(f"color: {theme_manager.colors['accent']};")
        content_layout.addWidget(section_appearance)
        
        sound_row = QHBoxLayout()
        sound_row.setSpacing(16)
        sound_label = QLabel("Громкость звуков")
        sound_label.setFont(QFont("Segoe UI", 13))
        sound_label.setStyleSheet(f"color: {theme_manager.colors['text_secondary']};")
        sound_row.addWidget(sound_label)
        sound_row.addStretch()
        
        self.sound_volume = QSlider(Qt.Orientation.Horizontal)
        self.sound_volume.setRange(0, 100)
        self.sound_volume.setValue(int(settings_manager.get('sound_volume', 0.5) * 100))
        self.sound_volume.setFixedWidth(150)
        self.sound_volume.setStyleSheet(theme_manager.get_slider_style())
        sound_row.addWidget(self.sound_volume)
        
        self.volume_label = QLabel(f"{int(self.sound_volume.value())}%")
        self.volume_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.volume_label.setStyleSheet(f"color: {theme_manager.colors['accent']};")
        self.volume_label.setFixedWidth(40)
        sound_row.addWidget(self.volume_label)
        
        self.sound_volume.valueChanged.connect(lambda v: self.volume_label.setText(f"{v}%"))
        content_layout.addLayout(sound_row)
        
        self.sound_enabled_check = QCheckBox("Включить звуковые уведомления")
        self.sound_enabled_check.setChecked(settings_manager.get('sound_enabled', True))
        content_layout.addWidget(self.sound_enabled_check)

        separator0 = QFrame()
        separator0.setFixedHeight(1)
        separator0.setStyleSheet(f"background-color: {theme_manager.colors['border']};")
        content_layout.addWidget(separator0)

        section_mouse = QLabel("🖱️  Управление мышью")
        section_mouse.setFont(QFont("Segoe UI", 14, QFont.Weight.DemiBold))
        section_mouse.setStyleSheet(f"color: {theme_manager.colors['accent']};")
        content_layout.addWidget(section_mouse)

        sens_row = QHBoxLayout()
        sens_row.setSpacing(16)
        sens_label = QLabel("Чувствительность")
        sens_label.setFont(QFont("Segoe UI", 13))
        sens_label.setStyleSheet(f"color: {theme_manager.colors['text_secondary']};")
        sens_row.addWidget(sens_label)
        
        self.sens_value = QLabel("1.0x")
        self.sens_value.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        self.sens_value.setStyleSheet(f"color: {theme_manager.colors['accent']};")
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
        self.sens_slider.setStyleSheet(theme_manager.get_slider_style())
        content_layout.addWidget(self.sens_slider)

        sens_hint = QLabel("Чем меньше значение, тем меньше движений рукой")
        sens_hint.setFont(QFont("Segoe UI", 11))
        sens_hint.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
        content_layout.addWidget(sens_hint)

        separator1 = QFrame()
        separator1.setFixedHeight(1)
        separator1.setStyleSheet(f"background-color: {theme_manager.colors['border']};")
        content_layout.addWidget(separator1)

        section_calib = QLabel("⚙️  Калибровка")
        section_calib.setFont(QFont("Segoe UI", 14, QFont.Weight.DemiBold))
        section_calib.setStyleSheet(f"color: {theme_manager.colors['accent']};")
        content_layout.addWidget(section_calib)

        self.auto_check = QCheckBox("Автоматическая калибровка при запуске")
        self.auto_check.setChecked(self.calibration.auto_calibrate if self.calibration else False)
        content_layout.addWidget(self.auto_check)

        face_row = QHBoxLayout()
        face_row.setSpacing(16)
        face_label = QLabel("Калибровка лица")
        face_label.setFont(QFont("Segoe UI", 13))
        face_label.setStyleSheet(f"color: {theme_manager.colors['text_secondary']};")
        face_row.addWidget(face_label)
        face_row.addStretch()
        
        self.face_status = QLabel("Не выполнена" if not (self.calibration and self.calibration.face_calibration["calibrated"]) else "Выполнена ✓")
        self.face_status.setFont(QFont("Segoe UI", 12))
        if self.calibration and self.calibration.face_calibration["calibrated"]:
            self.face_status.setStyleSheet(f"color: {theme_manager.colors['good']}; font-weight: 600;")
        else:
            self.face_status.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
        face_row.addWidget(self.face_status)
        
        self.btn_face = QPushButton("Начать")
        self.btn_face.setFixedSize(90, 38)
        self.btn_face.setCursor(Qt.CursorShape.PointingHandCursor)
        self._style_primary_button(self.btn_face)
        self.btn_face.clicked.connect(self.start_face_calibration)
        face_row.addWidget(self.btn_face)
        content_layout.addLayout(face_row)

        hand_row = QHBoxLayout()
        hand_row.setSpacing(16)
        hand_label = QLabel("Калибровка руки")
        hand_label.setFont(QFont("Segoe UI", 13))
        hand_label.setStyleSheet(f"color: {theme_manager.colors['text_secondary']};")
        hand_row.addWidget(hand_label)
        hand_row.addStretch()
        
        self.hand_status = QLabel("Не выполнена" if not (self.calibration and self.calibration.hand_calibration["calibrated"]) else "Выполнена ✓")
        self.hand_status.setFont(QFont("Segoe UI", 12))
        if self.calibration and self.calibration.hand_calibration["calibrated"]:
            self.hand_status.setStyleSheet(f"color: {theme_manager.colors['good']}; font-weight: 600;")
        else:
            self.hand_status.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
        hand_row.addWidget(self.hand_status)
        
        self.btn_hand = QPushButton("Начать")
        self.btn_hand.setFixedSize(90, 38)
        self.btn_hand.setCursor(Qt.CursorShape.PointingHandCursor)
        self._style_primary_button(self.btn_hand)
        self.btn_hand.clicked.connect(self.start_hand_calibration)
        hand_row.addWidget(self.btn_hand)
        content_layout.addLayout(hand_row)

        separator_fatigue = QFrame()
        separator_fatigue.setFixedHeight(1)
        separator_fatigue.setStyleSheet(f"background-color: {theme_manager.colors['border']};")
        content_layout.addWidget(separator_fatigue)

        section_fatigue = QLabel("🧠  Калибровка усталости")
        section_fatigue.setFont(QFont("Segoe UI", 14, QFont.Weight.DemiBold))
        section_fatigue.setStyleSheet(f"color: {theme_manager.colors['accent']};")
        content_layout.addWidget(section_fatigue)

        fatigue_hint = QLabel("Калибровка определяет вашу норму морганий, открытия глаз и рта. Это персонализирует систему под ваши индивидуальные особенности.")
        fatigue_hint.setFont(QFont("Segoe UI", 11))
        fatigue_hint.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
        fatigue_hint.setWordWrap(True)
        content_layout.addWidget(fatigue_hint)

        fatigue_row = QHBoxLayout()
        fatigue_row.setSpacing(16)
        fatigue_label = QLabel("Персонализация")
        fatigue_label.setFont(QFont("Segoe UI", 13))
        fatigue_label.setStyleSheet(f"color: {theme_manager.colors['text_secondary']};")
        fatigue_row.addWidget(fatigue_label)
        fatigue_row.addStretch()
        
        self.fatigue_status = QLabel("Не выполнена")
        self.fatigue_status.setFont(QFont("Segoe UI", 12))
        self.fatigue_status.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
        fatigue_row.addWidget(self.fatigue_status)
        
        self.btn_fatigue = QPushButton("Калибровать")
        self.btn_fatigue.setFixedSize(110, 38)
        self.btn_fatigue.setCursor(Qt.CursorShape.PointingHandCursor)
        self._style_primary_button(self.btn_fatigue)
        self.btn_fatigue.clicked.connect(self.start_fatigue_calibration)
        fatigue_row.addWidget(self.btn_fatigue)
        content_layout.addLayout(fatigue_row)

        self.fatigue_progress = QLabel("")
        self.fatigue_progress.setFont(QFont("Segoe UI", 11))
        self.fatigue_progress.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
        content_layout.addWidget(self.fatigue_progress)

        baseline_row = QHBoxLayout()
        baseline_row.setSpacing(16)
        baseline_label = QLabel("Ваша норма:")
        baseline_label.setFont(QFont("Segoe UI", 12))
        baseline_label.setStyleSheet(f"color: {theme_manager.colors['text_secondary']};")
        baseline_row.addWidget(baseline_label)
        baseline_row.addStretch()
        
        self.baseline_ear = QLabel("EAR: --")
        self.baseline_ear.setFont(QFont("Segoe UI", 12))
        self.baseline_ear.setStyleSheet(f"color: {theme_manager.colors['text_secondary']};")
        baseline_row.addWidget(self.baseline_ear)
        
        self.baseline_mar = QLabel("MAR: --")
        self.baseline_mar.setFont(QFont("Segoe UI", 12))
        self.baseline_mar.setStyleSheet(f"color: {theme_manager.colors['text_secondary']};")
        baseline_row.addWidget(self.baseline_mar)
        content_layout.addLayout(baseline_row)

        self.reset_calibration_btn = QPushButton("Сбросить калибровку")
        self.reset_calibration_btn.setFixedHeight(34)
        self.reset_calibration_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._style_secondary_button(self.reset_calibration_btn)
        self.reset_calibration_btn.clicked.connect(self.reset_fatigue_calibration)
        content_layout.addWidget(self.reset_calibration_btn)

        separator2 = QFrame()
        separator2.setFixedHeight(1)
        separator2.setStyleSheet(f"background-color: {theme_manager.colors['border']};")
        content_layout.addWidget(separator2)

        section_notify = QLabel("🔔  Уведомления")
        section_notify.setFont(QFont("Segoe UI", 14, QFont.Weight.DemiBold))
        section_notify.setStyleSheet(f"color: {theme_manager.colors['accent']};")
        content_layout.addWidget(section_notify)

        self.tray_check = QCheckBox("Сворачивать в трей вместо закрытия")
        self.tray_check.setChecked(settings_manager.get("minimize_to_tray", True))
        content_layout.addWidget(self.tray_check)

        separator_notify = QFrame()
        separator_notify.setFixedHeight(1)
        separator_notify.setStyleSheet(f"background-color: {theme_manager.colors['border']};")
        content_layout.addWidget(separator_notify)

        work_row = QHBoxLayout()
        work_row.setSpacing(16)
        work_label = QLabel("Перерыв каждые")
        work_label.setFont(QFont("Segoe UI", 13))
        work_label.setStyleSheet(f"color: {theme_manager.colors['text_secondary']};")
        work_row.addWidget(work_label)
        work_row.addStretch()
        
        self.stepper_work = NumberStepper(1, 240, self.settings.get("work_limit_minutes", 45), " мин")
        work_row.addWidget(self.stepper_work)
        content_layout.addLayout(work_row)

        posture_row = QHBoxLayout()
        posture_row.setSpacing(16)
        posture_label = QLabel("Чувствительность осанки")
        posture_label.setFont(QFont("Segoe UI", 13))
        posture_label.setStyleSheet(f"color: {theme_manager.colors['text_secondary']};")
        posture_row.addWidget(posture_label)
        posture_row.addStretch()
        
        self.stepper_posture = NumberStepper(5, 100, self.settings.get("posture_bad_percent", 30), "%")
        posture_row.addWidget(self.stepper_posture)
        content_layout.addLayout(posture_row)

        yawn_row = QHBoxLayout()
        yawn_row.setSpacing(16)
        yawn_label = QLabel("Лимит зевков (за 10 мин)")
        yawn_label.setFont(QFont("Segoe UI", 13))
        yawn_label.setStyleSheet(f"color: {theme_manager.colors['text_secondary']};")
        yawn_row.addWidget(yawn_label)
        yawn_row.addStretch()
        
        self.stepper_yawn = NumberStepper(1, 10, self.settings.get("yawn_limit", 4), "")
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
        self._style_secondary_button(self.btn_cancel)
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        self.btn_save = QPushButton("Сохранить")
        self.btn_save.setFixedSize(130, 46)
        self.btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self._style_primary_button(self.btn_save)
        self.btn_save.clicked.connect(self.save_settings)
        btn_layout.addWidget(self.btn_save)

        main_layout.addLayout(btn_layout)
    
    def _apply_theme_style(self):
        self.setStyleSheet(theme_manager.get_complete_stylesheet())
    
    def _style_primary_button(self, btn):
        c = theme_manager.colors
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {c['accent']};
                color: #FFFFFF;
                border: none;
                border-radius: 8px;
                padding: 6px 16px;
                font-size: 13px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {c['accent_hover']};
            }}
        """)
    
    def _style_secondary_button(self, btn):
        c = theme_manager.colors
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {c['bg_card']};
                color: {c['text_secondary']};
                border: 1px solid {c['border']};
                border-radius: 8px;
                padding: 6px 16px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                border-color: {c['border_light']};
                color: {c['text_primary']};
            }}
        """)

    def start_face_calibration(self):
        self.face_status.setText("Сбор данных...")
        self.face_status.setStyleSheet(f"color: {theme_manager.colors['warning']}; font-weight: 600;")
        if self.calibration:
            self.calibration.start_face_calibration()

    def start_hand_calibration(self):
        self.hand_status.setText("Сбор данных...")
        self.hand_status.setStyleSheet(f"color: {theme_manager.colors['warning']}; font-weight: 600;")
        if self.calibration:
            self.calibration.start_hand_calibration()

    def start_fatigue_calibration(self):
        """Start fatigue calibration process."""
        if not hasattr(self, 'user_profile_manager'):
            try:
                from neurofocus.ml import UserProfileManager
                self.user_profile_manager = UserProfileManager()
                self.user_profile_manager.load_profile('default')
            except Exception as e:
                print(f"Failed to init UserProfileManager: {e}")
                return
        
        self.fatigue_status.setText("Сбор данных... 0%")
        self.fatigue_status.setStyleSheet(f"color: {theme_manager.colors['warning']}; font-weight: 600;")
        self.btn_fatigue.setText("Идёт...")
        self.btn_fatigue.setEnabled(False)
        
        self.user_profile_manager.start_calibration()
        self._fatigue_calibrating = True
        self._fatigue_calibration_start = 60  # 60 seconds
        
    def update_fatigue_calibration(self, ear, mar):
        """Update calibration with new sample."""
        if hasattr(self, '_fatigue_calibrating') and self._fatigue_calibrating:
            if hasattr(self, 'user_profile_manager'):
                self.user_profile_manager.add_calibration_sample(ear, mar)
                
                samples = self.user_profile_manager.get_current_profile()['calibration']['samples_collected']
                progress = min(100, int(samples / 6))  # 1800 samples = 100%
                remaining = max(0, 60 - samples // 30)
                
                self.fatigue_progress.setText(f"Сбор данных... {progress}% ({remaining} сек)")
                
                if samples >= 1800:  # 60 seconds at 30fps
                    self.finish_fatigue_calibration()
    
    def finish_fatigue_calibration(self):
        """Finish fatigue calibration."""
        self._fatigue_calibrating = False
        
        if hasattr(self, 'user_profile_manager'):
            success = self.user_profile_manager.finish_calibration()
            
            if success:
                profile = self.user_profile_manager.get_current_profile()
                baseline = profile['baseline']
                
                self.fatigue_status.setText("Выполнена ✓")
                self.fatigue_status.setStyleSheet(f"color: {theme_manager.colors['good']}; font-weight: 600;")
                self.fatigue_progress.setText("Калибровка завершена!")
                
                self.baseline_ear.setText(f"EAR: {baseline['ear']:.3f}")
                self.baseline_mar.setText(f"MAR: {baseline['mar']:.3f}")
            else:
                self.fatigue_status.setText("Не выполнена")
                self.fatigue_status.setStyleSheet(f"color: {theme_manager.colors['danger']};")
                self.fatigue_progress.setText("Недостаточно данных")
        
        self.btn_fatigue.setText("Калибровать")
        self.btn_fatigue.setEnabled(True)
    
    def reset_fatigue_calibration(self):
        """Reset fatigue calibration to defaults."""
        if hasattr(self, 'user_profile_manager'):
            self.user_profile_manager.reset_calibration()
            
            self.fatigue_status.setText("Сброшена")
            self.fatigue_status.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
            self.fatigue_progress.setText("")
            self.baseline_ear.setText("EAR: --")
            self.baseline_mar.setText("MAR: --")

    def save_settings(self):
        sensitivity = self.sens_slider.value() / 100.0
        
        if self.calibration:
            self.calibration.set_sensitivity(sensitivity)
            self.calibration.set_auto_calibrate(self.auto_check.isChecked())
            
            if self.calibration.face_calibration["calibrated"]:
                self.face_status.setText("Выполнена ✓")
                self.face_status.setStyleSheet(f"color: {theme_manager.colors['good']}; font-weight: 600;")
            
            if self.calibration.hand_calibration["calibrated"]:
                self.hand_status.setText("Выполнена ✓")
                self.hand_status.setStyleSheet(f"color: {theme_manager.colors['good']}; font-weight: 600;")
        
        volume = self.sound_volume.value() / 100.0
        settings_manager.set('sound_volume', volume)
        sound_manager.set_volume(volume)
        sound_manager.set_enabled(self.sound_enabled_check.isChecked())
        settings_manager.set('sound_enabled', self.sound_enabled_check.isChecked())
        
        self.result_settings = {
            "work_limit_minutes": self.stepper_work.value(),
            "posture_window_minutes": 3,
            "posture_bad_percent": self.stepper_posture.value(),
            "yawn_limit": self.stepper_yawn.value(),
            "yawn_window_minutes": 10,
            "gesture_sensitivity": sensitivity,
            "minimize_to_tray": self.tray_check.isChecked(),
            "theme": "dark",
            "sound_volume": volume,
            "sound_enabled": self.sound_enabled_check.isChecked(),
        }
        self.accept()

    def get_settings(self):
        return self.result_settings

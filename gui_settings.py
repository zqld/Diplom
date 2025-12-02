from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QPushButton, 
                             QSpinBox, QFormLayout, QFrame)
from src.notifications import DEFAULT_SETTINGS

STYLESHEET_SETTINGS = """
QDialog { background-color: #1e1e1e; color: white; }
QLabel { font-size: 13px; font-family: 'Segoe UI'; }
QSpinBox { 
    background-color: #333; color: white; border: 1px solid #555; padding: 5px; 
}
QPushButton { 
    background-color: #007acc; color: white; border-radius: 5px; padding: 8px; font-weight: bold; 
}
QPushButton:hover { background-color: #005c99; }
"""

class SettingsWindow(QDialog):
    def __init__(self, current_settings=None):
        super().__init__()
        self.setWindowTitle("Настройки уведомлений")
        self.resize(400, 300)
        self.setStyleSheet(STYLESHEET_SETTINGS)
        
        self.settings = current_settings if current_settings else DEFAULT_SETTINGS.copy()
        self.result_settings = None # Сюда запишем результат

        layout = QVBoxLayout(self)
        
        title = QLabel("Параметры анализа")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

        form_layout = QFormLayout()
        form_layout.setSpacing(15)

        # 1. Время работы
        self.sb_work = QSpinBox()
        self.sb_work.setRange(1, 240)
        self.sb_work.setValue(self.settings["work_limit_minutes"])
        self.sb_work.setSuffix(" мин")
        form_layout.addRow("Напоминать об отдыхе каждые:", self.sb_work)

        # 2. Осанка (Порог %)
        self.sb_posture = QSpinBox()
        self.sb_posture.setRange(5, 100)
        self.sb_posture.setValue(self.settings["posture_bad_percent"])
        self.sb_posture.setSuffix(" %")
        lbl_info = QLabel("Уведомлять, если время с плохой осанкой превышает этот %:")
        lbl_info.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(lbl_info)
        form_layout.addRow("Чувствительность осанки:", self.sb_posture)

        # 3. Зевки
        self.sb_yawn = QSpinBox()
        self.sb_yawn.setRange(1, 10)
        self.sb_yawn.setValue(self.settings["yawn_limit"])
        form_layout.addRow("Лимит зевков (за 10 мин):", self.sb_yawn)

        layout.addLayout(form_layout)
        layout.addStretch()

        # Кнопки
        btn_save = QPushButton("Сохранить")
        btn_save.clicked.connect(self.save_settings)
        layout.addWidget(btn_save)

    def save_settings(self):
        self.result_settings = {
            "work_limit_minutes": self.sb_work.value(),
            "posture_window_minutes": 3, # Оставляем константой или тоже выносим
            "posture_bad_percent": self.sb_posture.value(),
            "yawn_limit": self.sb_yawn.value(),
            "yawn_window_minutes": 10
        }
        self.accept()

    def get_settings(self):
        return self.result_settings
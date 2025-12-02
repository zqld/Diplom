from gui_stats import StatsWindow # Импорт окна статистики
import sys
import cv2
import time
import datetime
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFrame, QProgressBar, QListWidget)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal, QEvent
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor
#для уведомлений
from src.notifications import NotificationManager, ToastNotification
from gui_settings import SettingsWindow

# Импорты
from src.emotion_detector import EmotionDetector 
from src.face_core import FaceMeshDetector
from src.geometry import calculate_ear, calculate_mar
from src.pose_estimator import HeadPoseEstimator
from src.database import DatabaseManager

# --- НАСТРОЙКИ КАЛИБРОВКИ ---
PITCH_OFFSET = 5.0  # Корректировка угла наклона головы (в градусах) для камеры с учетом ее расположения
PITCH_THRESHOLD = 0.0 # При каком угле считать наклон (чем меньше число, тем сильнее надо наклониться) (нижняя граница)
PITCH_THRESHOLD_TOP = 30.0 # верхняя граница для плохой осанки
POSTURE_TIME_TRIGGER = 1.0 # Сколько секунд надо сидеть криво, чтобы засчиталось событие
COOLDOWN_POSTURE_EVENT = 1.0 #таймер для записи события плохой осанки
COOLDOWN_YAWING_EVENT = 2.0 #таймер для записи события зевка


# --- СТИЛИ ---
STYLESHEET = """
QMainWindow { background-color: #1e1e1e; }
QLabel { color: #ffffff; font-family: 'Segoe UI', sans-serif; }
QFrame#InfoBox { background-color: #2d2d2d; border-radius: 10px; border: 1px solid #3d3d3d; }
QPushButton { background-color: #007acc; color: white; border-radius: 5px; padding: 8px; font-weight: bold; }
QPushButton:hover { background-color: #005c99; }
QPushButton#StopBtn { background-color: #cc0000; }
QProgressBar { border: 1px solid #3d3d3d; border-radius: 5px; text-align: center; background-color: #2d2d2d; }
QProgressBar::chunk { background-color: #007acc; border-radius: 5px; }
QListWidget { background-color: #252526; border: 1px solid #3d3d3d; border-radius: 5px; color: #cccccc; font-family: 'Consolas', 12px; }
"""

# --- ПОТОК ОБРАБОТКИ ВИДЕО ---
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    update_data_signal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.detector = FaceMeshDetector()
        self.pose_estimator = HeadPoseEstimator()
        self.emotion_ai = EmotionDetector("models/emotion_model.hdf5")
        self.db = DatabaseManager("session_data.db")
        
        self.last_save_time = time.time()
        self.frame_counter = 0
        self.current_emotion = "Neutral"
        
        self.posture_start_time = None 

    def run(self):
        cap = cv2.VideoCapture(0)
        
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            image, results = self.detector.process_frame(frame, draw=True)
            
            # Данные по умолчанию (если лица нет или оно не валидно)
            data = {
                "ear": 0.35, "mar": 0.0,  # Ставим "идеальные" значения
                "pitch": 0.0, "emotion": "...",
                "event": None,
                "posture_alert": False
            }
            
            is_face_valid = False # Флаг: можно ли доверять этому кадру?

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                lm_points = landmarks.landmark
                
                # --- ПРОВЕРКА ВАЛИДНОСТИ ЛИЦА (ЗАЩИТА ОТ ВЫХОДА ИЗ КАДРА) ---
                # 1. Собираем все X и Y координаты
                x_coords = [p.x for p in lm_points]
                y_coords = [p.y for p in lm_points]
                
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                
                # 2. Проверяем границы (Safe Zone)
                # Если лицо ближе чем 1% к краю экрана - считаем его невалидным
                # 0.01 = 1%, 0.99 = 99%
                if min_x < 0.01 or max_x > 0.99 or min_y < 0.01 or max_y > 0.99:
                    is_face_valid = False
                    cv2.putText(image, "FACE NOT FULLY VISIBLE - NO REC", (200, 300), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
                else:
                    is_face_valid = True

                # Считаем углы заранее, чтобы проверить Yaw
                raw_pitch, yaw, roll = self.pose_estimator.get_pose(frame, landmarks)

                # --- 2. ПРОВЕРКА ПОВОРОТА ГОЛОВЫ (YAW) ---
                # Если yaw > 35 градусов (сильный поворот вбок) -> НЕВАЛИДНО
                # Если raw_pitch > 40 (слишком сильно задрал/опустил голову вне зоны нормальности) -> НЕВАЛИДНО (не нужно тк тогда при засыфпании бдует думать, что невалидно)
                if abs(yaw) > 40: 
                    is_face_valid = False
                    cv2.putText(image, "LOOKING AWAY - NO REC", (250, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


                # Если лицо полностью в кадре, начинаем анализ
                if is_face_valid:
                    # Вычисления
                    ear = calculate_ear(lm_points)
                    mar = calculate_mar(lm_points)
                    #raw_pitch, yaw, roll = self.pose_estimator.get_pose(frame, landmarks)
                    
                    # 3. Доп. проверка: Если голову сильно повернули вбок (Yaw)
                    # Если поворот > 50 градусов, EAR считается неправильно
                    #if abs(yaw) > 50:
                    #    is_face_valid = False # Снимаем валидность
                    #else:
                    # Всё ок, записываем реальные данные
                    pitch = raw_pitch + PITCH_OFFSET
                    data["ear"] = ear
                    data["mar"] = mar
                    data["pitch"] = pitch
                    
                    # Эмоции
                    if self.frame_counter % 10 == 0:
                        emo, _ = self.emotion_ai.predict_emotion(frame, landmarks)
                        if emo not in ["Error", "No Face"]:
                            self.current_emotion = emo
                    data["emotion"] = self.current_emotion

                    # Логика статусов
                    fatigue_status = "Normal"
                    posture_status = "Good"

                    # А. Усталость
                    if mar > 0.6:
                        fatigue_status = "Yawning"
                        data["event"] = "ЗЕВОК (Усталость)"
                    elif ear < 0.22:
                        fatigue_status = "Eyes Closed"
                        data["event"] = "Закрытие глаз"

                    # Б. Осанка
                    if pitch < PITCH_THRESHOLD or pitch > PITCH_THRESHOLD_TOP:
                        if self.posture_start_time is None:
                            self.posture_start_time = time.time()
                        elif time.time() - self.posture_start_time > POSTURE_TIME_TRIGGER:
                            posture_status = "Bad Posture"
                            data["posture_alert"] = True
                            if data["event"] is None:
                                data["event"] = f"Плохая осанка ({int(pitch)}°)"
                    else:
                        self.posture_start_time = None
                        posture_status = "Good"

                    # Сохранение в БД (ТОЛЬКО ЕСЛИ ЛИЦО ВАЛИДНО)
                    if time.time() - self.last_save_time > 1.0:
                        self.db.save_log(
                            ear=ear, 
                            mar=mar, 
                            pitch=pitch,
                            emotion=self.current_emotion, 
                            fatigue_status=fatigue_status,
                            posture_status=posture_status
                        )
                        self.last_save_time = time.time()
                else:
                    # ЕСЛИ ЛИЦО НЕ ВАЛИДНО (Отвернулся / Край экрана)
                    self.posture_start_time = None # Сбрасываем таймер осанки
                    data["emotion"] = "Paused"     # Пишем в интерфейс "Пауза"
            else:
                # Лица вообще нет
                self.posture_start_time = None
                data["emotion"] = "No Face"
            
            # Если лицо не найдено или не валидно - сбрасываем таймер осанки
            if not results.multi_face_landmarks or not is_face_valid:
                self.posture_start_time = None

            # Конвертация и отправка в GUI
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            p = convert_to_qt_format.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio)
            
            self.change_pixmap_signal.emit(p)
            self.update_data_signal.emit(data)
            
            self.frame_counter += 1

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroFocus: Система контроля состояния")
        self.setGeometry(100, 100, 1100, 700)
        self.setStyleSheet(STYLESHEET)
        
        self.last_event_times = {} 
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Видео
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        self.image_label = QLabel(self)
        self.image_label.resize(800, 600)
        self.image_label.setStyleSheet("background-color: #000; border-radius: 10px;")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_layout.addWidget(self.image_label)
        main_layout.addWidget(video_container, stretch=3)

        # Сайдбар
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setSpacing(15)

        title = QLabel("МОНИТОРИНГ")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(title)
        
        self.emo_box = self.create_info_box("Текущая эмоция", "...")
        sidebar_layout.addWidget(self.emo_box)
        
        fatigue_label = QLabel("Уровень внимания (EAR)")
        sidebar_layout.addWidget(fatigue_label)
        self.fatigue_bar = QProgressBar()
        self.fatigue_bar.setRange(0, 100)
        self.fatigue_bar.setValue(100)
        sidebar_layout.addWidget(self.fatigue_bar)
        
        self.posture_label = QLabel("Осанка: Норма")
        self.posture_label.setStyleSheet("font-size: 14px; color: #00ff00;")
        sidebar_layout.addWidget(self.posture_label)
        
        log_title = QLabel("История событий:")
        log_title.setStyleSheet("color: #aaaaaa; margin-top: 10px;")
        sidebar_layout.addWidget(log_title)
        
        self.event_list = QListWidget()
        self.add_log_event("System started", color="#007acc")
        sidebar_layout.addWidget(self.event_list)

        # Кнопка статистики
        self.btn_stats = QPushButton("📊 Открыть отчет")
        self.btn_stats.clicked.connect(self.open_stats)
        sidebar_layout.addWidget(self.btn_stats)

        # Кнопка Настроек (НОВАЯ)
        self.btn_settings = QPushButton("⚙ Настройки")
        self.btn_settings.clicked.connect(self.open_settings)
        sidebar_layout.addWidget(self.btn_settings)

        # Кнопка выхода
        self.btn_stop = QPushButton("Завершить работу")
        self.btn_stop.setObjectName("StopBtn")
        self.btn_stop.clicked.connect(self.close_app)
        sidebar_layout.addWidget(self.btn_stop)

        # --- СИСТЕМА УВЕДОМЛЕНИЙ ---
        self.notify_manager = NotificationManager() # Подключаемся к БД

        # Таймер проверки (раз в 10 секунд, чтобы не грузить процессор)
        self.notify_timer = QTimer(self)
        self.notify_timer.setInterval(10000) # 10000 мс = 10 сек
        self.notify_timer.timeout.connect(self.check_notifications)
        self.notify_timer.start()

        main_layout.addWidget(sidebar, stretch=1)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_data_signal.connect(self.update_dashboard)
        self.thread.start()

    def create_info_box(self, title_text, value_text):
        frame = QFrame()
        frame.setObjectName("InfoBox")
        layout = QVBoxLayout(frame)
        t_label = QLabel(title_text)
        t_label.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        v_label = QLabel(value_text)
        v_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        v_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frame.value_label = v_label 
        layout.addWidget(t_label)
        layout.addWidget(v_label)
        return frame

    def add_log_event(self, text, color="#ffffff"):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        item_text = f"[{timestamp}] {text}"
        self.event_list.insertItem(0, item_text)
        item = self.event_list.item(0)
        item.setForeground(QColor(color))
        if self.event_list.count() > 50:
            self.event_list.takeItem(50)

    def update_image(self, qt_img):
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def update_dashboard(self, data):
        # 1. Обновляем Эмоции
        self.emo_box.value_label.setText(data["emotion"])
        
        # 2. Обновляем Бодрость
        ear = data["ear"]
        normalized_ear = int((ear - 0.15) / (0.35 - 0.15) * 100)
        normalized_ear = max(0, min(100, normalized_ear))
        self.fatigue_bar.setValue(normalized_ear)
        
        # 3. Обновляем Осанку (текст справа)
        pitch = data["pitch"]
        posture_text = f"ОСАНКА: НОРМА ({int(pitch)}°)"
        posture_style = "color: #00ff00;"
        
        if data["posture_alert"]: 
            posture_text = f"ОСАНКА: ПЛОХАЯ ({int(pitch)}°)"
            posture_style = "color: #ff4444; font-weight: bold;"
            
        self.posture_label.setText(posture_text)
        self.posture_label.setStyleSheet(posture_style)

        # 4. Логика Истории Событий (ИСПРАВЛЕНО)
        current_event = data["event"]
        if current_event:
            now = time.time()
            
            # Создаем "общий ключ" для проверки таймера.
            # Если событие про осанку, мы игнорируем градусы при проверке времени.
            if "осанка" in current_event.lower():
                tracking_key = "posture_any" # Общий ключ для любой кривой осанки
                cooldown = COOLDOWN_POSTURE_EVENT # Писать в лог не чаще чем раз в 2 секунды
                color = "#ff4444"
            else:
                tracking_key = current_event # Зевок
                cooldown = COOLDOWN_YAWING_EVENT
                color = "#ffffff"

            last_time = self.last_event_times.get(tracking_key, 0)
            
            # Если прошло достаточно времени с момента записи такого типа событий
            if now - last_time > cooldown:
                self.add_log_event(current_event, color)
                # Обновляем время для ОБЩЕГО ключа
                self.last_event_times[tracking_key] = now

    def open_stats(self):
        """Открывает окно со статистикой"""
        stats_dialog = StatsWindow()
        stats_dialog.exec() # exec делает окно модальным (поверх основного)
    
    def close_app(self):
        self.thread.stop()
        self.close()

    # функция для открытия настроек и уведомлений

    def open_settings(self):
        """Открывает окно настроек"""
        # Передаем текущие настройки в окно
        dialog = SettingsWindow(self.notify_manager.settings)
        if dialog.exec():
            # Если нажали "Сохранить", забираем новые данные
            new_config = dialog.get_settings()
            if new_config:
                self.notify_manager.update_settings(new_config)
                # Можно показать тост-подтверждение
                self.show_notification("Настройки", "Параметры мониторинга обновлены.")

    def check_notifications(self):
        """Вызывается таймером раз в 10 сек"""
        # Спрашиваем у менеджера, есть ли проблемы
        result = self.notify_manager.check_conditions()
        
        if result:
            title, msg = result
            self.show_notification(title, msg)

    def show_notification(self, title, msg):
        """Показывает всплывающее окно справа внизу"""
        # Важно сохранять ссылку на тост, иначе сборщик мусора удалит его мгновенно
        self.toast = ToastNotification(title, msg)
        self.toast.show_toast()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
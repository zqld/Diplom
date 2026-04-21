from ui.stats import StatsWindow
import os
import sys
import cv2
import time
import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFrame, QProgressBar, QListWidget)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QShortcut, QImage, QPixmap, QFont, QColor, QIcon
from PyQt6.QtWidgets import QSystemTrayIcon, QMenu
from src.notifications import NotificationManager, ToastNotification
from ui.settings import SettingsWindow
from ui.calibration import CalibrationDialog
from src.database import DatabaseManager
from src.calibration_manager import CalibrationManager
from src.sound_manager import sound_manager
from src.logger import logger

PITCH_OFFSET = 5.0
PITCH_THRESHOLD = 0.0
PITCH_THRESHOLD_TOP = 30.0
POSTURE_TIME_TRIGGER = 1.0
COOLDOWN_POSTURE_EVENT = 1.0
COOLDOWN_YAWING_EVENT = 2.0

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
    'good_bg': 'rgba(74, 222, 128, 0.15)',
    'warning_bg': 'rgba(251, 191, 36, 0.15)',
    'danger_bg': 'rgba(248, 113, 113, 0.15)',
}


class ModernButton(QPushButton):
    def __init__(self, text, primary=False, parent=None):
        super().__init__(text, parent)
        self.primary = primary
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(44)
        if primary:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {DARK_COLORS['accent']};
                    color: #FFFFFF;
                    border: none;
                    border-radius: 10px;
                    padding: 10px 20px;
                    font-size: 14px;
                    font-weight: 600;
                    font-family: 'Segoe UI', sans-serif;
                }}
                QPushButton:hover {{
                    background-color: {DARK_COLORS['accent_hover']};
                }}
                QPushButton:pressed {{
                    opacity: 0.85;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {DARK_COLORS['bg_card']};
                    color: {DARK_COLORS['text_secondary']};
                    border: 1px solid {DARK_COLORS['border']};
                    border-radius: 10px;
                    padding: 10px 20px;
                    font-size: 14px;
                    font-weight: 500;
                    font-family: 'Segoe UI', sans-serif;
                }}
                QPushButton:hover {{
                    background-color: {DARK_COLORS['bg_input']};
                    border-color: {DARK_COLORS['border_light']};
                }}
            """)


class ModernPrimaryButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(48)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {DARK_COLORS['accent']};
                color: #FFFFFF;
                border: none;
                border-radius: 12px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
                font-family: 'Segoe UI', sans-serif;
            }}
            QPushButton:hover {{
                background-color: {DARK_COLORS['accent_hover']};
            }}
            QPushButton:pressed {{
                opacity: 0.85;
            }}
        """)


class DangerButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(44)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {DARK_COLORS['danger']};
                border: 1px solid {DARK_COLORS['danger']};
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 500;
                font-family: 'Segoe UI', sans-serif;
            }}
            QPushButton:hover {{
                background-color: {DARK_COLORS['danger_bg']};
            }}
        """)


class MetricCard(QFrame):
    def __init__(self, title, value="--", unit="", status="normal", parent=None):
        super().__init__(parent)
        self.title_text = title
        self.value_text = value
        self.unit_text = unit
        self.status = status
        
        self.setFixedHeight(100)
        self.setMinimumWidth(150)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: transparent;
                border: none;
            }}
            QLabel {{
                background: transparent;
                color: {DARK_COLORS['text_secondary']};
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(4)
        
        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Medium))
        self.title_label.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
        layout.addWidget(self.title_label)
        
        value_row = QHBoxLayout()
        value_row.setSpacing(8)
        
        self.value_label = QLabel(f"{value}{unit}")
        self.value_label.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        self.value_label.setStyleSheet(f"color: {DARK_COLORS['text_primary']};")
        self.value_label.setMinimumWidth(140)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        value_row.addWidget(self.value_label)
        
        self.status_indicator = QLabel()
        self.status_indicator.setFixedSize(8, 8)
        self.status_indicator.setStyleSheet(f"background-color: {DARK_COLORS['text_muted']}; border-radius: 4px;")
        value_row.addWidget(self.status_indicator)
        value_row.addStretch()
        
        layout.addLayout(value_row)
        
        self.update_status(status)
        
    def update_value(self, value, unit=None):
        self.value_label.setText(f"{value}{unit if unit else self.unit_text}")
        
    def update_status(self, status):
        self.status = status
        if status == "good":
            self.status_indicator.setStyleSheet(f"background-color: {DARK_COLORS['good']}; border-radius: 4px;")
        elif status == "warning":
            self.status_indicator.setStyleSheet(f"background-color: {DARK_COLORS['warning']}; border-radius: 4px;")
        elif status == "danger":
            self.status_indicator.setStyleSheet(f"background-color: {DARK_COLORS['danger']}; border-radius: 4px;")
        else:
            self.status_indicator.setStyleSheet(f"background-color: {DARK_COLORS['text_muted']}; border-radius: 4px;")


class ModernProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                background-color: {DARK_COLORS['bg_input']};
                border-radius: 6px;
                height: 8px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {DARK_COLORS['accent']};
                border-radius: 6px;
            }}
        """)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    update_data_signal = pyqtSignal(dict)
    calibration_progress_signal = pyqtSignal(str, int)
    calibration_done_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self._paused = False          # True = анализ заморожен, камера работает
        self._force_close = False
        self._initialized = False
        self._init_processors()
        
        self.session_start_time = time.time()
        self.face_lost_time = None
        self._last_calibration_status = {"face": False, "hand": False}
        self._last_face_state = True
        self._yawn_cooldown_end = 0.0  # debounce зевка: игнорировать до этого времени

        self._error_count = 0
        self._max_errors = 10
        
        logger.info("VideoThread инициализирован")
    
    def _init_processors(self):
        try:
            from src.processors import FaceProcessor, EmotionProcessor, FatigueProcessor, PostureProcessor, HandProcessor

            self.face_processor = FaceProcessor()
            self.emotion_processor = EmotionProcessor()
            # Keep old processors as fallback
            self.fatigue_processor = FatigueProcessor()
            self.posture_processor = PostureProcessor()
            self.hand_processor = None

            self.face_detector = self.face_processor.detector
            self.db = DatabaseManager("session_data.db")

            self.calibration_manager = None

            self.last_save_time = time.time()
            self.frame_counter = 0

            # --- Performance tuning ---
            # Тяжёлые ML-модели (LSTM, Dense posture) — каждые 8 кадров (~4 Hz)
            self._ml_update_interval   = 8
            # Рука/жесты — каждый кадр (30 Hz).
            # model_complexity=0 даёт ~5 ms/frame — укладываемся в бюджет 33 ms.
            # Interval=1 устраняет мигание landmarks (при 2 они пропадали через кадр).
            self._hand_update_interval = 1
            self._last_data = {}  # кэш аналитики для кадров без ML
            self._last_hand_data: dict = {}  # кэш для кадров без hand update

            # --- ML Classifiers (neurofocus) ---
            self._ml_ready = False
            try:
                import os
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
                from neurofocus.ml.fatigue_classifier import FatigueClassifier
                from neurofocus.ml.posture_classifier import PostureClassifier
                from neurofocus.ml.ml_coordinator import MLCoordinator
                from neurofocus.detectors.pose_detector import PoseDetector

                self.fatigue_classifier = FatigueClassifier()
                self.posture_classifier = PostureClassifier(
                    model_path='models/posture_model.keras'
                )
                self.pose_detector = PoseDetector(
                    model_path='models/pose_landmarker_lite.task'
                )

                # ML Coordinator: manages warm-up, personalized thresholds, ML blend
                self.ml_coordinator = MLCoordinator(
                    self.fatigue_classifier, self.posture_classifier
                )

                self._ml_ready = True
                logger.info("ML классификаторы + Online Learning загружены")
            except Exception as ml_err:
                logger.warning(f"ML классификаторы недоступны, используются пороговые значения: {ml_err}")
                self.fatigue_classifier = None
                self.posture_classifier = None
                self.pose_detector = None
                self.ml_coordinator = None

            self._initialized = True

        except Exception as e:
            logger.error(f"Ошибка инициализации процессоров: {e}")
            self._initialized = False
    
    def set_calibration_manager(self, calibration_manager):
        self.calibration_manager = calibration_manager
        try:
            from src.processors import HandProcessor
            self.hand_processor = HandProcessor(calibration_manager=calibration_manager)
        except Exception as e:
            logger.error(f"Ошибка инициализации HandProcessor: {e}")
        # Передаём calibration_manager в PostureProcessor для персонального pitch
        try:
            if hasattr(self, 'posture_processor') and self.posture_processor:
                self.posture_processor.set_calibration_manager(calibration_manager)
        except Exception as e:
            logger.error(f"Ошибка передачи calibration_manager в PostureProcessor: {e}")
    
    def _get_calibration_overlay(self, frame):
        if not self.calibration_manager:
            return None
        
        try:
            if self.calibration_manager._is_calibrating_face:
                progress = len(self.calibration_manager._face_samples)
                pct = int(progress / 20 * 100)
                return {"type": "face", "progress": progress, "pct": pct}

            if self.calibration_manager._is_calibrating_hand:
                progress = len(self.calibration_manager._hand_samples)
                pct = int(progress / 20 * 100)
                return {"type": "hand", "progress": progress, "pct": pct}
        except Exception as e:
            logger.error(f"Ошибка отрисовки калибровки: {e}")
        
        return None

    def run(self):
        if not self._initialized:
            logger.error("VideoThread: процессоры не инициализированы")
            return
        
        cap = None
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                logger.error("VideoThread: не удалось открыть камеру")
                self.error_signal.emit("Не удалось открыть камеру")
                return
            
            # 1280×720 даёт MediaPipe ~2× больше пикселей на руку/лицо →
            # детекция надёжнее. Большинство веб-камер поддерживают 720p @ 30 fps.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)       # 30 fps стабильнее 60 на 720p
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # всегда берём самый свежий кадр

            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"VideoThread: камера запущена {actual_w}x{actual_h} @ {actual_fps:.0f} FPS")
        except Exception as e:
            logger.error(f"VideoThread: ошибка инициализации камеры: {e}")
            self.error_signal.emit(f"Ошибка камеры: {e}")
            return

        # --- Default data for frames before first ML run ---
        default_data = {
            "ear": 0.35, "mar": 0.0, "pitch": 0.0, "emotion": "...",
            "event": None, "posture_alert": False,
            "posture_score": 0, "posture_level": "good",
            "face_detected": False, "hand_detected": False,
            "fatigue_status": "Awake", "fatigue_score": 0,
            "blink_rate": 0, "yawning": False, "microsleep_detected": False,
            "model_used": "geometric", "posture_status": "Good",
            "ml_warmup_progress": 0,
        }
        self._last_data = dict(default_data)

        while self._run_flag:
            frame = None
            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    self._error_count += 1
                    if self._error_count >= self._max_errors:
                        logger.error("VideoThread: слишком много ошибок чтения камеры")
                        break
                    continue

                self._error_count = 0
                frame = cv2.flip(frame, 1)

            except Exception as e:
                logger.error(f"VideoThread: ошибка чтения камеры: {e}")
                continue

            # ---- LIGHT PATH (every frame): face detect → render ----
            try:
                image, results = self.face_detector.process_frame(frame, draw=True)
            except Exception as e:
                logger.error(f"VideoThread: ошибка обнаружения лица: {e}")
                image = frame
                results = None

            face_data = None
            is_face_valid = False
            try:
                face_data = self.face_processor.process(frame, results)
                is_face_valid = face_data['valid']
            except Exception as e:
                logger.error(f"VideoThread: ошибка face_processor: {e}")

            # Determine if this frame should run heavy ML
            do_ml = (self.frame_counter % self._ml_update_interval == 0)

            if do_ml:
                data = self._run_heavy_ml(
                    frame, image, face_data, is_face_valid, default_data
                )
            else:
                # Reuse cached analytics from last full run
                data = dict(self._last_data)
                if face_data and is_face_valid:
                    data["ear"] = face_data.get("ear", self._last_data.get("ear", 0.35))
                    data["mar"] = face_data.get("mar", self._last_data.get("mar", 0.0))
                    data["pitch"] = face_data.get("pitch", self._last_data.get("pitch", 0.0))
                    data["face_detected"] = face_data.get("detected", False)

            # ---- Calibration (every frame, lightweight) ----
            try:
                if self.calibration_manager and is_face_valid:
                    ear_val   = face_data.get("ear",   0.3)
                    mar_val   = face_data.get("mar",   0.15)
                    pitch_val = face_data.get("pitch", 0.0)

                    # Используем реальный размер руки если он уже был измерен
                    hand_size_val = (
                        getattr(self.hand_processor, '_hand_size', None)
                        if self.hand_processor else None
                    )

                    # Авто-калибровка: не мешает ручной, возвращает (face_done, hand_done)
                    was_face_calib = self.calibration_manager.face_calibration["calibrated"]
                    was_hand_calib = self.calibration_manager.hand_calibration["calibrated"]
                    face_done, hand_done = self.calibration_manager.auto_calibrate_if_needed(
                        ear_val, mar_val, pitch_val, hand_size_val
                    )

                    # Сигнал о прогрессе авто-калибровки лица
                    if not was_face_calib and not self.calibration_manager._is_calibrating_face:
                        auto_progress = len(self.calibration_manager._face_samples)
                        if auto_progress > 0:
                            self.calibration_progress_signal.emit("face_auto", auto_progress)
                    if face_done:
                        self.calibration_done_signal.emit("face")
                        logger.info("Авто-калибровка лица завершена")
                    if hand_done:
                        self.calibration_done_signal.emit("hand")
                        logger.info("Авто-калибровка руки завершена")

                    # Ручная калибровка лица
                    if self.calibration_manager._is_calibrating_face:
                        self.calibration_manager.add_face_sample(ear_val, mar_val, pitch_val)
                        progress = len(self.calibration_manager._face_samples)
                        self.calibration_progress_signal.emit("face", progress)

                        if progress >= 20:
                            self.calibration_manager.finish_face_calibration()
                            self.calibration_done_signal.emit("face")
                            logger.info("Калибровка лица завершена")

                    # Ручная калибровка осанки
                    if self.calibration_manager._is_calibrating_posture:
                        self.calibration_manager.add_posture_sample(pitch_val)
                        progress = len(self.calibration_manager._posture_samples)
                        self.calibration_progress_signal.emit("posture", progress)

                        if progress >= 20:
                            self.calibration_manager.finish_posture_calibration()
                            self.calibration_done_signal.emit("posture")
                            logger.info("Калибровка осанки завершена")
            except Exception as e:
                logger.error(f"VideoThread: ошибка калибровки лица: {e}")

            try:
                calib = self.calibration_manager.face_calibration["calibrated"] if self.calibration_manager else False
                if calib != self._last_calibration_status["face"]:
                    self._last_calibration_status["face"] = calib
            except Exception:
                pass

            try:
                calib_info = self._get_calibration_overlay(image)
                if calib_info:
                    data["calibration_info"] = calib_info
            except Exception as e:
                logger.error(f"VideoThread: ошибка отрисовки калибровки: {e}")

            # ---- Hand / Gesture — каждые _hand_update_interval кадров ----
            # Вынесено из _run_heavy_ml: жесты должны обновляться ~15 Hz,
            # а не 4 Hz как тяжёлые ML-модели. Иначе курсор «скачет».
            do_hand = (self.frame_counter % self._hand_update_interval == 0)
            try:
                if self.hand_processor and do_hand:
                    hand_data = self.hand_processor.process(
                        image, frame.shape[1], frame.shape[0]
                    )
                    self._last_hand_data = hand_data

                    data["hand_detected"]   = hand_data['detected']
                    data["current_gesture"] = hand_data.get('current_gesture', 'none')

                    # Наложение жеста на кадр
                    gesture_label = hand_data.get('gesture', 'none')
                    if gesture_label and gesture_label != 'none':
                        cv2.rectangle(image, (5, 5), (200, 35), (0, 0, 0), -1)
                        cv2.putText(image, f"[{gesture_label.upper()}]", (10, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Сбор сэмплов для ручной калибровки руки
                    if (self.calibration_manager
                            and self.calibration_manager._is_calibrating_hand
                            and hand_data.get('hand_size')):
                        self.calibration_manager.add_hand_sample(hand_data['hand_size'])
                        progress = len(self.calibration_manager._hand_samples)
                        self.calibration_progress_signal.emit("hand", progress)
                        if progress >= 20:
                            self.calibration_manager.finish_hand_calibration()
                            self.calibration_done_signal.emit("hand")
                            logger.info("Калибровка руки завершена")

                    # Сбор сэмплов для калибровки активной зоны жестов
                    if (self.calibration_manager
                            and self.calibration_manager._is_calibrating_zone
                            and hand_data.get('palm_x') is not None):
                        px = hand_data['palm_x']
                        py = hand_data['palm_y']
                        cm = self.calibration_manager
                        cm.add_zone_sample(px, py)
                        step = cm._zone_step
                        if step == 'topleft':
                            progress = len(cm._zone_topleft_samples)
                            self.calibration_progress_signal.emit("zone_topleft", progress)
                        elif step == 'bottomright':
                            progress = len(cm._zone_bottomright_samples)
                            self.calibration_progress_signal.emit("zone_bottomright", progress)
                            if progress >= 15:
                                cm.finish_gesture_zone_calibration()
                                self.calibration_done_signal.emit("zone")
                                logger.info("Калибровка зоны жестов завершена")
                                # После смены зоны сбрасываем позицию жестового контроллера,
                                # чтобы outlier-фильтр не заморозил курсор
                                if (self.hand_processor
                                        and self.hand_processor.gesture_controller):
                                    gc = self.hand_processor.gesture_controller
                                    gc.prev_x = gc.screen_width  // 2
                                    gc.prev_y = gc.screen_height // 2
                                    gc._outlier_consecutive = 0
                                    gc._gesture_buf.clear()

                elif self.hand_processor and self._last_hand_data:
                    # Кадр без hand update — используем кэш для UI, жест не двигает мышь
                    data["hand_detected"]   = self._last_hand_data.get('detected', False)
                    data["current_gesture"] = self._last_hand_data.get('current_gesture', 'none')
            except Exception as e:
                logger.error(f"VideoThread: ошибка обработки руки: {e}")

            try:
                h, w, ch = image.shape
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                scaled_image = qt_image.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio)

                self.change_pixmap_signal.emit(scaled_image)
                self.update_data_signal.emit(data)

                self.frame_counter += 1

            except Exception as e:
                logger.error(f"VideoThread: ошибка отправки изображения: {e}")

        if cap:
            try:
                cap.release()
            except Exception:
                pass

        logger.info("VideoThread: завершен")

    def toggle_pause(self):
        """Переключить паузу анализа."""
        self._paused = not self._paused
        return self._paused

    def _run_heavy_ml(self, frame, image, face_data, is_face_valid, default_data):
        """Run heavy ML models (LSTM, pose, posture, emotion) — called every N frames."""
        # Если анализ на паузе — возвращаем кешированные данные
        if self._paused:
            cached = dict(self._last_data)
            cached['analysis_paused'] = True
            return cached
        data = dict(default_data)
        current_time = time.time()
        ear = 0.35
        mar = 0.15
        pitch = 0.0
        emotion = "..."
        pose_landmarks = None

        if not is_face_valid or face_data is None:
            return data

        ear = face_data.get('ear', 0.35)
        mar = face_data.get('mar', 0.15)
        pitch = face_data.get('pitch', 0.0)
        data["ear"] = ear
        data["mar"] = mar
        data["pitch"] = pitch
        data["face_detected"] = face_data.get('detected', False)

        # --- Online Learning: feed features into ML coordinator ---
        # Передаём face_is_visible, чтобы ThresholdAdapter ставил warm-up
        # на паузу при потере лица (не копит мусорные сэмплы).
        if self._ml_ready and self.ml_coordinator is not None:
            self.ml_coordinator.update(
                ear, mar, pitch, current_time,
                face_is_visible=is_face_valid,
            )
            data["ml_warmup_progress"] = self.ml_coordinator.get_calibration_progress()

        # --- Emotion ---
        try:
            emotion = self.emotion_processor.process(frame, face_data['landmarks'])
        except Exception:
            pass
        data["emotion"] = emotion

        # --- Fatigue: ML (LSTM) or threshold fallback ---
        fatigue_data = {}
        if self._ml_ready and self.fatigue_classifier is not None:
            try:
                ml_fatigue = self.fatigue_classifier.predict(face_data['landmarks'], frame)
                _f_status   = ml_fatigue.get('status', 'awake')
                _yawning    = ml_fatigue.get('yawning', False)
                _microsleep = ml_fatigue.get('microsleep_detected', False)

                # Маппинг статуса на уровень (нужен online learning и update_dashboard)
                _level_map = {'awake': 'normal', 'drowsy': 'moderate', 'sleeping': 'severe'}
                _f_level = _level_map.get(_f_status, 'normal')

                # Генерация события (аналог FatigueProcessor, но из ML)
                _f_event = None
                # ── Зевок с debounce: один зевок = одно событие (cooldown 5 сек) ──
                if _yawning and current_time > self._yawn_cooldown_end:
                    _f_event = "Зевок"
                    self._yawn_cooldown_end = current_time + 5.0  # 5 сек cooldown
                elif _microsleep:
                    _f_event = "Сильная усталость"
                elif _f_status == 'sleeping':
                    _f_event = "Сильная усталость"
                elif _f_status == 'drowsy':
                    _f_event = "Усталость"

                fatigue_data = {
                    'fatigue_status': _f_status.capitalize(),
                    'fatigue_score':  ml_fatigue.get('fatigue_score', 0),
                    'fatigue_level':  _f_level,
                    'blink_rate':     ml_fatigue.get('blink_rate', 0),
                    'ear': ear,
                    'mar': mar,
                    'yawning':             _yawning,
                    'microsleep_detected': _microsleep,
                    'model_used':          ml_fatigue.get('model_used', 'geometric'),
                    'event':               _f_event,
                }
            except Exception as ml_e:
                logger.warning(f"ML fatigue error, fallback: {ml_e}")
                fatigue_data = self.fatigue_processor.process(ear, mar, pitch, emotion, current_time)
        else:
            fatigue_data = self.fatigue_processor.process(ear, mar, pitch, emotion, current_time)
        data.update(fatigue_data)

        # --- Online Learning: collect labeled sample for background retraining ---
        # Вызывается каждый heavy-ML кадр (~4 Hz). Фоновый поток запустит
        # дообучение, когда наберётся 500 сэмплов (RETRAIN_THRESHOLD).
        if self._ml_ready and self.ml_coordinator is not None:
            try:
                self.ml_coordinator.collect_sample(
                    ear, mar, pitch,
                    fatigue_data.get('fatigue_level', 'normal'),
                    current_time,
                )
            except Exception as ol_e:
                # Online learning не должен ломать основной цикл
                logger.warning(f"Online learning collect error: {ol_e}")

        # --- Posture: ML (Dense) with Pose or face-mesh fallback ---
        posture_data = {}
        if self._ml_ready and self.posture_classifier is not None and self.pose_detector is not None:
            ml_weight = self.ml_coordinator.get_ml_blend_weight() if self.ml_coordinator else 0.0
            try:
                _, pose_results = self.pose_detector.process_frame(frame, draw=False)
                pose_landmarks = self.pose_detector.get_landmarks(pose_results)

                # Проверяем, что плечи видны (nose=0, l_shoulder=11, r_shoulder=12).
                # get_landmarks() возвращает список, поэтому hasattr(..., 'landmark') = False.
                # Бёдра (23, 24) на вебкамере не видны — не требуем их здесь;
                # extract_pose_features сам откажется от ML если бёдра невидны.
                pose_usable = (
                    pose_landmarks is not None
                    and isinstance(pose_landmarks, list)
                    and len(pose_landmarks) >= 13
                    and getattr(pose_landmarks[11], 'visibility', 0) > 0.3
                    and getattr(pose_landmarks[12], 'visibility', 0) > 0.3
                )

                if pose_usable:
                    ml_posture = self.posture_classifier.predict(pose_landmarks, ml_weight=ml_weight)
                    used = 'ml_progressive' if 0 < ml_weight < 1 else ('ml_pure' if ml_weight >= 1.0 else 'ml_dense')
                    posture_data = {
                        'posture_status': ml_posture.get('status', 'good').capitalize(),
                        'posture_score': int(ml_posture.get('confidence', 0) * 100),
                        'posture_level': ml_posture.get('status', 'good'),
                        'posture_alert': ml_posture.get('status') == 'bad',
                        'model_used_posture': used,
                    }
                else:
                    # Pose landmarks недоступны или неполные —
                    # используем геометрический анализ по Face Mesh
                    # с компенсацией угла камеры из калибровки
                    baseline_pitch = 0.0
                    if self.calibration_manager:
                        baseline_pitch = self.calibration_manager.posture_calibration.get(
                            'baseline_pitch', 0.0
                        )
                    pm = self.posture_classifier.predict_from_face_mesh(
                        face_data['landmarks'], frame.shape[1], frame.shape[0],
                        calibration_baseline_pitch=baseline_pitch,
                        head_pitch=pitch,
                    )
                    posture_data = {
                        'posture_status': pm.get('status', 'good').capitalize(),
                        'posture_score': int(pm.get('confidence', 0) * 100),
                        'posture_level': pm.get('status', 'good'),
                        'posture_alert': pm.get('status') == 'bad',
                        'model_used_posture': 'face_mesh_geometric',
                    }
            except Exception as ml_pe:
                logger.warning(f"ML posture error, fallback: {ml_pe}")
                posture_data = self.posture_processor.process(
                    face_data['landmarks'], frame.shape[1], frame.shape[0], pitch, current_time
                )
        else:
            posture_data = self.posture_processor.process(
                face_data['landmarks'], frame.shape[1], frame.shape[0], pitch, current_time
            )
        data.update(posture_data)

        # Нормализуем posture_alert: ML возвращает разные ключи и регистры.
        # posture_processor (fallback) использует 'is_bad', ML-путь — 'posture_alert',
        # а сравнение статуса должно быть case-insensitive.
        # ИСПРАВЛЕНО: добавлен уровень posture_alert_level ('fair' vs 'bad')
        _ps = posture_data.get('posture_status', '').lower()
        _pl = posture_data.get('posture_level', '').lower()

        data['posture_alert'] = bool(
            posture_data.get('posture_alert', False)
            or posture_data.get('is_bad', False)
            or posture_data.get('pitch_alert', False)
            or _ps in ('bad', 'bad posture', 'fair')
            or _pl in ('bad', 'fair')
        )

        # Новый ключ: уровень серьёзности проблемы с осанкой
        # 'bad' = высокая опасность, 'fair' = предупреждение
        if _ps in ('bad', 'bad posture') or _pl == 'bad':
            data['posture_alert_level'] = 'bad'
        elif _ps == 'fair' or _pl == 'fair':
            data['posture_alert_level'] = 'fair'
        else:
            data['posture_alert_level'] = None

        # Выбор события для отображения в логе.
        # Осанка имеет приоритет (у неё 30 с кулдаун), усталость — 2 с.
        # Если осанка не выдала событие, показываем событие усталости.
        _posture_ev = posture_data.get('event')
        _fatigue_ev = data.get('event')   # уже заполнено из fatigue_data.update()
        if _posture_ev:
            data['event'] = _posture_ev
        elif _fatigue_ev:
            data['event'] = _fatigue_ev
        else:
            data['event'] = None

        # ── DIAGNOSTIC: log first 5 heavy-ML cycles ──────────────
        if self.frame_counter < 40 and self.frame_counter % 8 == 0:
            logger.info(
                f"[DIAG] frame={self.frame_counter} | "
                f"ear={ear:.3f} mar={mar:.3f} pitch={pitch:.2f} | "
                f"fatigue={fatigue_data.get('status', 'N/A')} "
                f"score={fatigue_data.get('fatigue_score', 0):.0f} "
                f"model={fatigue_data.get('model_used', 'N/A')} | "
                f"posture={posture_data.get('posture_status', 'N/A')} "
                f"level={posture_data.get('posture_level', 'N/A')} "
                f"model={posture_data.get('model_used_posture', 'N/A')} | "
                f"blink_rate={fatigue_data.get('blink_rate', 0)}"
            )


        # --- DB save (every ~1s) ---
        if time.time() - self.last_save_time > 1.0:
            posture_raw = posture_data.get('posture_status', 'Good')
            posture_lower = posture_raw.lower()
            if posture_lower == 'bad':
                posture_db = 'Bad Posture'
            elif posture_lower == 'fair':
                posture_db = 'Fair Posture'
            else:
                posture_db = posture_raw

            fatigue_raw = fatigue_data.get('fatigue_status', 'Awake')
            mar_val = fatigue_data.get('mar', 0)
            if mar_val > 0.6:
                fatigue_db = 'Yawning'
            elif fatigue_raw.lower() == 'sleeping':
                fatigue_db = 'Eyes Closed'
            else:
                fatigue_db = fatigue_raw

            # Добавляем информацию об использованной модели в статус
            model_used = fatigue_data.get('model_used', 'geometric')
            posture_model_used = posture_data.get('model_used_posture', 'geometric')
            fatigue_db = f"{fatigue_db} [{model_used}]"
            posture_db = f"{posture_db} [{posture_model_used}]"

            self.db.save_log(
                ear=ear, mar=mar, pitch=pitch, emotion=emotion,
                fatigue_status=fatigue_db, posture_status=posture_db
            )
            self.last_save_time = time.time()

        # Cache the result for lightweight frames
        self._last_data = data
        return data

    def stop(self):
        self._run_flag = False
        self.wait(3000)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.force_close = False
        self.setWindowTitle("NeuroFocus")
        self.setGeometry(80, 80, 1280, 800)
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {DARK_COLORS['bg_main']};
            }}
            QLabel {{
                font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
                color: {DARK_COLORS['text_primary']};
            }}
            QScrollBar:vertical {{
                background: {DARK_COLORS['bg_card']};
                width: 6px;
                border-radius: 3px;
            }}
            QScrollBar::handle:vertical {{
                background: {DARK_COLORS['border_light']};
                border-radius: 3px;
                min-height: 20px;
            }}
        """)

        self.last_event_times = {}
        self._attention_level = 100
        self._yawn_cooldown_end = 0.0  # debounce: до этого времени зевок игнорируется

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(24)

        left_panel = QFrame()
        left_panel.setStyleSheet("background-color: transparent;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(20)

        header_area = QFrame()
        header_area.setStyleSheet("background-color: transparent;")
        header_layout = QHBoxLayout(header_area)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        app_logo = QLabel("◈")
        app_logo.setFont(QFont("Segoe UI", 24))
        app_logo.setStyleSheet(f"color: {DARK_COLORS['accent']};")
        header_layout.addWidget(app_logo)
        
        app_title = QLabel("NeuroFocus")
        app_title.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        app_title.setStyleSheet(f"color: {DARK_COLORS['text_primary']}; letter-spacing: 0.5px;")
        header_layout.addWidget(app_title)
        
        header_layout.addStretch()
        
        self.status_dot = QLabel()
        self.status_dot.setFixedSize(10, 10)
        self.status_dot.setStyleSheet(f"background-color: {DARK_COLORS['good']}; border-radius: 5px;")
        header_layout.addWidget(self.status_dot)
        
        self.status_text = QLabel("Мониторинг активен")
        self.status_text.setFont(QFont("Segoe UI", 12, QFont.Weight.Medium))
        self.status_text.setStyleSheet(f"color: {DARK_COLORS['good']}; margin-left: 8px;")
        header_layout.addWidget(self.status_text)
        
        left_layout.addWidget(header_area)

        video_container = QFrame()
        video_container.setStyleSheet(f"""
            QFrame {{
                background-color: {DARK_COLORS['bg_card']};
                border-radius: 16px;
                border: none;
            }}
        """)
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(0)
        
        video_header = QFrame()
        video_header.setStyleSheet(f"background-color: {DARK_COLORS['bg_card']};")
        video_header.setFixedHeight(48)
        video_header_layout = QHBoxLayout(video_header)
        video_header_layout.setContentsMargins(16, 0, 16, 0)
        
        video_title = QLabel("Видеопоток")
        video_title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        video_title.setStyleSheet(f"color: {DARK_COLORS['text_primary']};")
        video_header_layout.addWidget(video_title)
        
        video_header_layout.addStretch()
        
        video_layout.addWidget(video_header)
        
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet(f"background-color: #121215; border-radius: 0 0 16px 16px;")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_layout.addWidget(self.image_label, stretch=1)

        left_layout.addWidget(video_container, stretch=1)

        right_panel = QFrame()
        right_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {DARK_COLORS['bg_card']};
                border-radius: 16px;
                border: none;
            }}
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(16)

        metrics_header = QLabel("Показатели")
        metrics_header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        metrics_header.setStyleSheet(f"color: {DARK_COLORS['text_muted']}; letter-spacing: 1px;")
        right_layout.addWidget(metrics_header)

        metrics_grid = QHBoxLayout()
        metrics_grid.setSpacing(16)
        
        self.emotion_card = MetricCard("Эмоция", "Нейтрально", "", "normal")
        metrics_grid.addWidget(self.emotion_card)
        
        self.attention_card = MetricCard("Внимание", "100", "%", "good")
        metrics_grid.addWidget(self.attention_card)
        
        right_layout.addLayout(metrics_grid)

        posture_frame = QFrame()
        posture_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {DARK_COLORS['bg_input']};
                border-radius: 12px;
                border: none;
            }}
        """)
        posture_layout = QVBoxLayout(posture_frame)
        posture_layout.setContentsMargins(16, 14, 16, 14)
        posture_layout.setSpacing(8)
        
        posture_title = QLabel("Осанка")
        posture_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Medium))
        posture_title.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
        posture_layout.addWidget(posture_title)
        
        posture_value_layout = QHBoxLayout()
        posture_value_layout.setSpacing(10)
        
        self.posture_icon = QLabel("✓")
        self.posture_icon.setFont(QFont("Segoe UI", 20))
        self.posture_icon.setStyleSheet(f"color: {DARK_COLORS['good']};")
        posture_value_layout.addWidget(self.posture_icon)
        
        self.posture_value = QLabel("Хорошая")
        self.posture_value.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        self.posture_value.setStyleSheet(f"color: {DARK_COLORS['text_primary']};")
        self.posture_value.setMinimumWidth(100)
        posture_value_layout.addWidget(self.posture_value)
        
        posture_value_layout.addStretch()
        
        self.posture_angle = QLabel("0°")
        self.posture_angle.setFont(QFont("Segoe UI", 14))
        self.posture_angle.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
        self.posture_angle.setMinimumWidth(40)
        posture_value_layout.addWidget(self.posture_angle)
        
        posture_layout.addLayout(posture_value_layout)
        
        right_layout.addWidget(posture_frame)

        attention_frame = QFrame()
        attention_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {DARK_COLORS['bg_input']};
                border-radius: 12px;
                border: none;
            }}
        """)
        attention_layout = QVBoxLayout(attention_frame)
        attention_layout.setContentsMargins(16, 14, 16, 14)
        attention_layout.setSpacing(10)
        
        attention_title_layout = QHBoxLayout()
        
        attention_title = QLabel("Уровень внимания")
        attention_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Medium))
        attention_title.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
        attention_title_layout.addWidget(attention_title)
        
        attention_title_layout.addStretch()
        
        self.attention_percent = QLabel("100%")
        self.attention_percent.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.attention_percent.setStyleSheet(f"color: {DARK_COLORS['good']};")
        self.attention_percent.setMinimumWidth(50)
        attention_title_layout.addWidget(self.attention_percent)
        
        attention_layout.addLayout(attention_title_layout)
        
        self.fatigue_bar = ModernProgressBar()
        self.fatigue_bar.setRange(0, 100)
        self.fatigue_bar.setValue(100)
        attention_layout.addWidget(self.fatigue_bar)
        
        right_layout.addWidget(attention_frame)
        
        gesture_status_frame = QFrame()
        gesture_status_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {DARK_COLORS['bg_input']};
                border-radius: 12px;
                border: none;
            }}
        """)
        gesture_status_layout = QHBoxLayout(gesture_status_frame)
        gesture_status_layout.setContentsMargins(16, 10, 16, 10)
        
        gesture_label = QLabel("Мышь жестами")
        gesture_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Medium))
        gesture_label.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
        gesture_status_layout.addWidget(gesture_label)
        
        gesture_status_layout.addStretch()
        
        self.gesture_status = QLabel("Выкл")
        self.gesture_status.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.gesture_status.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
        gesture_status_layout.addWidget(self.gesture_status)
        
        right_layout.addWidget(gesture_status_frame)

        session_frame = QFrame()
        session_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {DARK_COLORS['bg_input']};
                border-radius: 12px;
                border: none;
            }}
        """)
        session_layout = QVBoxLayout(session_frame)
        session_layout.setContentsMargins(16, 12, 16, 12)
        session_layout.setSpacing(8)
        
        session_title = QLabel("Статистика сессии")
        session_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Medium))
        session_title.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
        session_layout.addWidget(session_title)
        
        session_row = QHBoxLayout()
        session_row.setSpacing(16)
        
        time_label = QLabel("Время работы:")
        time_label.setFont(QFont("Segoe UI", 11))
        time_label.setStyleSheet(f"color: {DARK_COLORS['text_secondary']};")
        session_row.addWidget(time_label)
        
        self.session_time_label = QLabel("00:00:00")
        self.session_time_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.session_time_label.setStyleSheet(f"color: {DARK_COLORS['accent']};")
        session_row.addWidget(self.session_time_label)
        
        session_row.addStretch()
        
        events_label = QLabel("Событий:")
        events_label.setFont(QFont("Segoe UI", 11))
        events_label.setStyleSheet(f"color: {DARK_COLORS['text_secondary']};")
        session_row.addWidget(events_label)
        
        self.session_events_label = QLabel("0")
        self.session_events_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.session_events_label.setStyleSheet(f"color: {DARK_COLORS['warning']};")
        session_row.addWidget(self.session_events_label)
        
        session_layout.addLayout(session_row)
        
        right_layout.addWidget(session_frame)

        events_header = QLabel("События")
        events_header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        events_header.setStyleSheet(f"color: {DARK_COLORS['text_muted']}; letter-spacing: 1px;")
        right_layout.addWidget(events_header)

        self.event_list = QListWidget()
        self.event_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {DARK_COLORS['bg_input']};
                border: none;
                border-radius: 12px;
                padding: 12px;
                font-size: 12px;
                color: {DARK_COLORS['text_secondary']};
            }}
            QListWidget::item {{
                padding: 6px 4px;
                border-bottom: 1px solid {DARK_COLORS['border']};
            }}
            QListWidget::item:last-child {{
                border-bottom: none;
            }}
        """)
        self.event_list.setFixedHeight(130)
        self.add_log_event("Система запущена", color=DARK_COLORS['text_muted'])
        right_layout.addWidget(self.event_list)

        right_layout.addStretch()

        button_group = QFrame()
        button_group.setStyleSheet("background-color: transparent;")
        button_layout = QVBoxLayout(button_group)
        button_layout.setSpacing(10)
        
        self.btn_stats = ModernPrimaryButton("📊  Аналитика")
        self.btn_stats.clicked.connect(self.open_stats)
        button_layout.addWidget(self.btn_stats)
        
        self.btn_progress = ModernButton("📈  Прогресс")
        self.btn_progress.clicked.connect(self.open_progress)
        button_layout.addWidget(self.btn_progress)
        
        self.btn_pomodoro = ModernButton("🍅  Pomodoro")
        self.btn_pomodoro.clicked.connect(self.open_pomodoro)
        button_layout.addWidget(self.btn_pomodoro)
        
        self.btn_pause = ModernButton("⏸  Пауза анализа")
        self.btn_pause.clicked.connect(self.toggle_analysis_pause)
        button_layout.addWidget(self.btn_pause)

        self.btn_gesture = ModernButton("🖱️  Управление мышью")
        self.btn_gesture.clicked.connect(self.toggle_gesture)
        button_layout.addWidget(self.btn_gesture)
        
        self.btn_help = ModernButton("❓  Жесты")
        self.btn_help.clicked.connect(self.show_gesture_help)
        button_layout.addWidget(self.btn_help)

        self.btn_settings = ModernButton("⚙  Настройки")
        self.btn_settings.clicked.connect(self.open_settings)
        button_layout.addWidget(self.btn_settings)

        self.btn_stop = DangerButton("Завершить мониторинг")
        self.btn_stop.clicked.connect(self.close_app)
        button_layout.addWidget(self.btn_stop)
        
        right_layout.addWidget(button_group)

        main_layout.addWidget(left_panel, stretch=3)
        main_layout.addWidget(right_panel, stretch=1)

        self.notify_manager = NotificationManager()
        self.session_start_time = time.time()
        self.face_lost_count = 0
        self.face_lost_time = None
        self._auto_pause_triggered = False
        self._last_face_state = True

        self.notify_timer = QTimer(self)
        self.notify_timer.setInterval(10000)
        self.notify_timer.timeout.connect(self.check_notifications)
        self.notify_timer.start()

        self.session_timer = QTimer(self)
        self.session_timer.setInterval(1000)
        self.session_timer.timeout.connect(self.update_session_stats)
        self.session_timer.start()

        # Периодически агрегируем данные в daily_progress (каждые 5 минут)
        self.progress_sync_timer = QTimer(self)
        self.progress_sync_timer.setInterval(5 * 60 * 1000)  # 5 мин
        self.progress_sync_timer.timeout.connect(self._sync_daily_progress)
        self.progress_sync_timer.start()

        self.calibration_manager = CalibrationManager()
        
        self.video_thread = VideoThread()
        self.video_thread.set_calibration_manager(self.calibration_manager)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.update_data_signal.connect(self.update_dashboard)
        self.video_thread.calibration_progress_signal.connect(self.on_calibration_progress)
        self.video_thread.calibration_done_signal.connect(self.on_calibration_done)
        self.video_thread.start()
        
        self.shortcut_escape = QShortcut(Qt.Key.Key_Escape, self)
        self.shortcut_escape.activated.connect(self.on_escape_pressed)
        
        self.shortcut_g = QShortcut(Qt.Key.Key_G, self)
        self.shortcut_g.activated.connect(self.toggle_gesture)
        
        self.setup_tray()
        
        logger.info("Приложение запущено успешно")
    
    def setup_tray(self):
        self.tray_icon = QSystemTrayIcon(self)
        
        tray_menu = QMenu()
        tray_menu.setStyleSheet(f"""
            QMenu {{
                background-color: {DARK_COLORS['bg_card']};
                color: {DARK_COLORS['text_primary']};
                border: 1px solid {DARK_COLORS['border']};
                border-radius: 8px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 8px 16px;
                border-radius: 4px;
            }}
            QMenu::item:selected {{
                background-color: {DARK_COLORS['bg_input']};
            }}
        """)
        
        show_action = tray_menu.addAction("Показать NeuroFocus")
        show_action.triggered.connect(self.show)
        show_action.triggered.connect(self.activateWindow)
        
        tray_menu.addSeparator()
        
        quit_action = tray_menu.addAction("Выход")
        quit_action.triggered.connect(self.close_app)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.tray_icon_activated)
        self.tray_icon.setToolTip("NeuroFocus - Мониторинг активен")
        
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        if os.path.exists(icon_path):
            self.tray_icon.setIcon(QIcon(icon_path))
        else:
            pixmap = QPixmap(64, 64)
            pixmap.fill(QColor(DARK_COLORS['accent']))
            self.tray_icon.setIcon(QIcon(pixmap))
        
        self.tray_icon.show()
    
    def tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show()
            self.activateWindow()
    
    def on_calibration_progress(self, calib_type, progress):
        pct = int(progress / 20 * 100)
        # Звук — только один раз при первом сэмпле ручной калибровки
        if calib_type == "face" and progress == 1:
            sound_manager.calibration_start()
        # face_auto — авто-калибровка лица, не дублируем лог каждый кадр
        if calib_type == "face_auto":
            if progress in (10, 20, 30):   # показываем только на 33% / 66% / 100%
                self.add_log_event(f"Авто-калибровка лица: {int(progress/30*100)}%",
                                   DARK_COLORS['text_muted'])
        else:
            self.add_log_event(f"Калибровка {calib_type}: {pct}%", DARK_COLORS['text_muted'])

    def on_calibration_done(self, calib_type):
        labels = {
            "face":    "лица",
            "hand":    "руки",
            "posture": "осанки",
            "zone":    "зоны управления",
        }
        label = labels.get(calib_type, calib_type)
        self.add_log_event(f"Калибровка {label} завершена ✓", DARK_COLORS['good'])
        self.show_notification("Калибровка", f"Калибровка {label} успешно завершена")

    def add_log_event(self, text, color="#6A6A7A"):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        item_text = f"{timestamp}  {text}"
        self.event_list.insertItem(0, item_text)
        item = self.event_list.item(0)
        if item:
            item.setForeground(QColor(color))
        if self.event_list.count() > 25:
            self.event_list.takeItem(25)

    def update_image(self, qt_img):
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def update_dashboard(self, data):
        emotion = data["emotion"]
        self.emotion_card.update_value(emotion)
        
        face_detected = data.get("face_detected", True)
        
        if face_detected:
            if not self._last_face_state:
                self.face_lost_time = None
                self._auto_pause_triggered = False
                self.add_log_event("Лицо найдено ✓", DARK_COLORS['good'])
            
            self.status_dot.setStyleSheet(f"background-color: {DARK_COLORS['good']}; border-radius: 5px;")
            self.status_text.setText("Мониторинг активен")
            self.status_text.setStyleSheet(f"color: {DARK_COLORS['good']}; margin-left: 8px;")
            self._last_face_state = True
        else:
            if self._last_face_state:
                self.face_lost_time = time.time()
            
            self.status_dot.setStyleSheet(f"background-color: {DARK_COLORS['warning']}; border-radius: 5px;")
            self.status_text.setText("Лицо не найдено")
            self.status_text.setStyleSheet(f"color: {DARK_COLORS['warning']}; margin-left: 8px;")
            self._last_face_state = False
            
            if self.face_lost_time and not self._auto_pause_triggered:
                lost_duration = time.time() - self.face_lost_time
                if lost_duration > 5:
                    self._auto_pause_triggered = True
                    self.face_lost_count += 1
                    sound_manager.warning()
                    self.add_log_event("⚠️ Лицо потеряно >5 сек", DARK_COLORS['warning'])
                    logger.warning(f"Лицо пользователя потеряно на {lost_duration:.1f} секунд")
        
        fatigue_level = data.get("fatigue_level", "normal")
        
        if emotion in ["Нейтрально", "Neutral", "Спокойствие"]:
            self.emotion_card.update_status("good")
        elif emotion in ["Счастье", "Happy"]:
            self.emotion_card.update_status("good")
        elif emotion in ["Усталость", "Tired", "Грусть", "Sad"]:
            self.emotion_card.update_status("warning")
        elif fatigue_level in ["mild", "moderate", "severe"]:
            self.emotion_card.update_status("warning")
        else:
            self.emotion_card.update_status("normal")

        fatigue_score = data.get("fatigue_score", 0)
        attention_level = int(max(0, min(100, 100 - fatigue_score)))
        
        self._attention_level = attention_level
        self.fatigue_bar.setValue(attention_level)
        self.attention_percent.setText(f"{attention_level}%")
        
        if attention_level >= 70:
            self.attention_card.update_status("good")
            self.attention_percent.setStyleSheet(f"color: {DARK_COLORS['good']};")
        elif attention_level >= 40:
            self.attention_card.update_status("warning")
            self.attention_percent.setStyleSheet(f"color: {DARK_COLORS['warning']};")
        else:
            self.attention_card.update_status("danger")
            self.attention_percent.setStyleSheet(f"color: {DARK_COLORS['danger']};")
            
        self.attention_card.update_value(f"{attention_level}", "%")

        pitch = data["pitch"]

        # ИСПРАВЛЕНО: posture_alert теперь имеет уровни 'fair' и 'bad'
        posture_alert_level = data.get('posture_alert_level')

        if posture_alert_level == 'bad':
            # Плохая осанка — красный, восклицательный знак
            self.posture_value.setText("Плохая")
            self.posture_angle.setText(f"{int(pitch)}°")
            self.posture_icon.setText("!")
            self.posture_icon.setStyleSheet(f"color: {DARK_COLORS['danger']}; font-weight: bold;")
            self.posture_value.setStyleSheet(f"color: {DARK_COLORS['danger']};")
        elif posture_alert_level == 'fair':
            # Среднее состояние — жёлтый предупреждающий
            self.posture_value.setText("Средняя")
            self.posture_angle.setText(f"{int(pitch)}°")
            self.posture_icon.setText("⚠")
            self.posture_icon.setStyleSheet(f"color: {DARK_COLORS['warning']}; font-weight: bold;")
            self.posture_value.setStyleSheet(f"color: {DARK_COLORS['warning']};")
        else:
            self.posture_value.setText("Хорошая")
            self.posture_angle.setText(f"{int(pitch)}°")
            self.posture_icon.setText("✓")
            self.posture_icon.setStyleSheet(f"color: {DARK_COLORS['good']};")
            self.posture_value.setStyleSheet(f"color: {DARK_COLORS['text_primary']};")

        current_event = data["event"]
        if current_event:
            now = time.time()
            ev_lower = current_event.lower()

            _posture_keywords = ("осанка", "наклон", "отклонение", "голова", "поза")
            if any(kw in ev_lower for kw in _posture_keywords):
                tracking_key = "posture_any"
                # ИСПРАВЛЕНО: разные кулдауны для 'fair' и 'bad'
                posture_level = data.get('posture_alert_level')
                if posture_level == 'bad':
                    cooldown = 30.0   # плохая осанка — чаще
                    color = DARK_COLORS['danger']
                else:
                    cooldown = 45.0   # средняя — реже, чтобы не спамить
                    color = DARK_COLORS['warning']
            else:
                tracking_key = current_event
                cooldown = COOLDOWN_YAWING_EVENT
                color = DARK_COLORS['warning']

            last_time = self.last_event_times.get(tracking_key, 0)

            if now - last_time > cooldown:
                self.add_log_event(current_event, color)
                self.last_event_times[tracking_key] = now
                # Звуковой сигнал: осанка — attention, усталость — warning
                try:
                    if tracking_key == "posture_any":
                        sound_manager.attention()
                    else:
                        sound_manager.warning()
                except Exception:
                    pass

    def open_stats(self):
        stats_dialog = StatsWindow()
        stats_dialog.exec()
    
    def _sync_daily_progress(self):
        """Фоновая агрегация daily_progress каждые 5 минут."""
        try:
            from src.progress_tracker import ProgressTracker
            ProgressTracker().update_daily_progress()
        except Exception:
            pass

    def open_progress(self):
        # Обновляем дневной прогресс перед открытием окна
        self._sync_daily_progress()
        from ui.progress import ProgressWindow
        progress_dialog = ProgressWindow(self)
        progress_dialog.exec()
    
    def open_pomodoro(self):
        from ui.pomodoro import PomodoroTimer
        # Переиспользуем существующий экземпляр — не создаём новый при каждом открытии,
        # чтобы таймер не останавливался при закрытии окна
        if not hasattr(self, '_pomodoro') or self._pomodoro is None:
            self._pomodoro = PomodoroTimer(self)
            # Подключаем сигнал завершения цикла к системе уведомлений
            self._pomodoro.pomodoro_finished.connect(
                lambda title, msg: self.show_notification(title, msg, accent='#4ADE80')
            )
        self._pomodoro.show()
        self._pomodoro.raise_()
        self._pomodoro.activateWindow()

    def close_app(self):
        logger.info("Завершение работы приложения...")

        self.force_close = True

        self.tray_icon.hide()

        if hasattr(self, 'notify_timer'):
            self.notify_timer.stop()

        if hasattr(self, 'session_timer'):
            self.session_timer.stop()

        if hasattr(self, 'progress_sync_timer'):
            self.progress_sync_timer.stop()

        if hasattr(self, 'video_thread'):
            self.video_thread.stop()
            self.video_thread.wait(3000)

        # Gracefully flush async DB queue
        if hasattr(self, 'db') and self.db:
            try:
                self.db.stop(timeout=5.0)
            except Exception as e:
                logger.error(f"Ошибка остановки БД: {e}")

        try:
            from src.progress_tracker import ProgressTracker
            tracker = ProgressTracker()
            tracker.update_daily_progress()
            logger.info("Ежедневный прогресс обновлен")
        except Exception as e:
            logger.error(f"Ошибка обновления прогресса: {e}")

        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QCoreApplication
        logger.info("Приложение завершено")
        QApplication.quit()
    
    def closeEvent(self, event):
        if not self.force_close:
            event.ignore()
            self.hide()
            self.tray_icon.showMessage(
                "NeuroFocus",
                "Мониторинг продолжается в фоне. Нажмите на иконку для возврата.",
                QSystemTrayIcon.MessageIcon.Information,
                2000
            )
        else:
            event.accept()

    def open_calibration(self):
        """Открыть минималистичный диалог калибровки."""
        dlg = CalibrationDialog(self.calibration_manager, parent=self)
        dlg.exec()
        # После закрытия — уведомление если хоть что-то откалибровано
        cm = self.calibration_manager
        if cm and (cm.face_calibration.get("calibrated") or cm.hand_calibration.get("calibrated")):
            self.show_notification("Калибровка", "Параметры сохранены ✓")

    def open_settings(self):
        dialog = SettingsWindow(self.notify_manager.settings, self.calibration_manager)
        dialog.exec()
        if dialog.result_settings:
            self.notify_manager.update_settings(dialog.result_settings)
            # Применяем чувствительность жестов к работающему контроллеру
            sensitivity = dialog.result_settings.get('gesture_sensitivity', 1.0)
            hp = getattr(self.video_thread, 'hand_processor', None)
            if hp and hasattr(hp, 'gesture_controller') and hp.gesture_controller:
                hp.gesture_controller.set_sensitivity(sensitivity)
            self.show_notification("Настройки", "Параметры обновлены")
    
    def toggle_analysis_pause(self):
        paused = self.video_thread.toggle_pause()
        if paused:
            self.btn_pause.setText("▶  Возобновить")
            self.btn_pause.setStyleSheet(f"""
                QPushButton {{
                    background-color: {DARK_COLORS['warning_bg']};
                    color: {DARK_COLORS['warning']};
                    border: 1px solid {DARK_COLORS['warning']};
                    border-radius: 10px;
                    padding: 10px 20px;
                    font-size: 14px;
                    font-weight: 600;
                    font-family: 'Segoe UI', sans-serif;
                }}
                QPushButton:hover {{
                    background-color: rgba(251,191,36,0.25);
                }}
            """)
            self.show_notification("Пауза", "Анализ лица и осанки приостановлен", accent='#FBBF24')
        else:
            self.btn_pause.setText("⏸  Пауза анализа")
            self.btn_pause.setStyleSheet("")
            # Восстанавливаем стиль ModernButton
            self.btn_pause.setStyleSheet(f"""
                QPushButton {{
                    background-color: {DARK_COLORS['bg_card']};
                    color: {DARK_COLORS['text_secondary']};
                    border: 1px solid {DARK_COLORS['border']};
                    border-radius: 10px;
                    padding: 10px 20px;
                    font-size: 14px;
                    font-weight: 500;
                    font-family: 'Segoe UI', sans-serif;
                }}
                QPushButton:hover {{
                    background-color: {DARK_COLORS['bg_input']};
                    border-color: {DARK_COLORS['border_light']};
                }}
            """)
            self.show_notification("Анализ", "Анализ возобновлён")

    def toggle_gesture(self):
        hp = getattr(self.video_thread, 'hand_processor', None)
        if hp is not None:
            enabled = hp.toggle_gesture_control()
            if enabled:
                self.gesture_status.setText("Вкл")
                self.gesture_status.setStyleSheet(f"color: {DARK_COLORS['good']};")
                self.btn_gesture.setText("🖱️  Мышь ВКЛ")
            else:
                self.gesture_status.setText("Выкл")
                self.gesture_status.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
                self.btn_gesture.setText("🖱️  Управление мышью")
        else:
            self.show_notification("Жесты", "Управление жестами недоступно")
    
    def show_gesture_help(self):
        from ui.help import GestureHelpWindow
        help_dialog = GestureHelpWindow(self)
        help_dialog.exec()
    
    def on_escape_pressed(self):
        hp = getattr(self.video_thread, 'hand_processor', None)
        if hp is not None and hp.is_enabled():
            hp.disable_gesture_control()
            self.gesture_status.setText("Выкл")
            self.gesture_status.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
            self.btn_gesture.setText("🖱️  Управление мышью")
            self.show_notification("Мышь", "Управление выключено")
    
    def update_session_stats(self):
        elapsed = int(time.time() - self.session_start_time)
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        self.session_time_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

        if hasattr(self, 'event_list'):
            self.session_events_label.setText(str(self.event_list.count()))

        # Обновляем дневной прогресс каждые 5 минут
        if elapsed > 0 and elapsed % 300 == 0:
            try:
                from src.progress_tracker import ProgressTracker
                ProgressTracker().update_daily_progress()
            except Exception:
                pass

    def check_notifications(self):
        result = self.notify_manager.check_conditions()
        if result:
            title, msg = result
            # Подбираем цвет акцента по теме уведомления
            if 'осанк' in title.lower() or 'осанк' in msg.lower():
                accent = '#F87171'   # danger — осанка
            elif 'устал' in title.lower() or 'зева' in msg.lower():
                accent = '#FBBF24'  # warning — усталость
            else:
                accent = '#6B8AFE'  # accent — перерыв/общее
            self.show_notification(title, msg, accent=accent)

    def show_notification(self, title, msg, accent=None):
        self.toast = ToastNotification(title, msg, accent_color=accent)
        self.toast.show_toast()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

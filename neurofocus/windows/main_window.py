from neurofocus.windows.stats import StatsWindow
import os
import sys

tensorflow_available = False
try:
    import tensorflow as tf
    tensorflow_available = True
except (ImportError, Exception):
    pass

import cv2
import time
import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFrame, QProgressBar, QListWidget, QSlider)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QShortcut, QImage, QPixmap, QFont, QColor, QIcon
from PyQt6.QtWidgets import QSystemTrayIcon, QMenu

from neurofocus.detectors import FaceDetector, HandDetector, PoseDetector
from neurofocus.analyzers import EmotionAnalyzer as EmotionDetector
from neurofocus.ml import FatigueClassifier, PostureClassifier, TrainingDataCollector, UserProfileManager
from neurofocus.utils import LandmarkUtils, landmarks_utils
from neurofocus.utils.geometry import calculate_ear, calculate_mar
from neurofocus.utils.pose_estimator import HeadPoseEstimator
from neurofocus.utils.database import DatabaseManager
from neurofocus.utils.logger import logger
from neurofocus.utils.sound import sound_manager
from neurofocus.controls.gesture import GestureController
from neurofocus.controls.calibration import CalibrationManager
from neurofocus.core.constants import (
    PITCH_OFFSET, PITCH_THRESHOLD, PITCH_THRESHOLD_TOP,
    POSTURE_TIME_TRIGGER, COOLDOWN_POSTURE_EVENT, COOLDOWN_YAWING_EVENT
)
from neurofocus.core.settings_manager import settings_manager
from neurofocus.ui.theme import theme_manager
from neurofocus.ui.fatique_graph import MiniFatigueGraph

from neurofocus.utils.notifications import NotificationManager, ToastNotification

try:
    from neurofocus.windows.settings import SettingsWindow
except:
    SettingsWindow = None


class ModernButton(QPushButton):
    def __init__(self, text, primary=False, parent=None):
        super().__init__(text, parent)
        self.primary = primary
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(44)
        if primary:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {theme_manager.colors['accent']};
                    color: #FFFFFF;
                    border: none;
                    border-radius: 10px;
                    padding: 10px 20px;
                    font-size: 14px;
                    font-weight: 600;
                    font-family: 'Segoe UI', sans-serif;
                }}
                QPushButton:hover {{
                    background-color: {theme_manager.colors['accent_hover']};
                }}
                QPushButton:pressed {{
                    opacity: 0.85;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {theme_manager.colors['bg_card']};
                    color: {theme_manager.colors['text_secondary']};
                    border: 1px solid {theme_manager.colors['border']};
                    border-radius: 10px;
                    padding: 10px 20px;
                    font-size: 14px;
                    font-weight: 500;
                    font-family: 'Segoe UI', sans-serif;
                }}
                QPushButton:hover {{
                    background-color: {theme_manager.colors['bg_input']};
                    border-color: {theme_manager.colors['border_light']};
                }}
            """)


class ModernPrimaryButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(48)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {theme_manager.colors['accent']};
                color: #FFFFFF;
                border: none;
                border-radius: 12px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
                font-family: 'Segoe UI', sans-serif;
            }}
            QPushButton:hover {{
                background-color: {theme_manager.colors['accent_hover']};
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
                color: {theme_manager.colors['danger']};
                border: 1px solid {theme_manager.colors['danger']};
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 500;
                font-family: 'Segoe UI', sans-serif;
            }}
            QPushButton:hover {{
                background-color: {theme_manager.colors['danger_bg']};
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
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
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
        
        self._apply_card_style()
        self.update_status(status)
        
    def update_value(self, value, unit=None):
        self.value_label.setText(f"{value}{unit if unit else self.unit_text}")
        
    def update_status(self, status):
        self.status = status
        if status == "good":
            self.status_indicator.setStyleSheet(f"background-color: {theme_manager.colors['good']}; border-radius: 4px;")
        elif status == "warning":
            self.status_indicator.setStyleSheet(f"background-color: {theme_manager.colors['warning']}; border-radius: 4px;")
        elif status == "danger":
            self.status_indicator.setStyleSheet(f"background-color: {theme_manager.colors['danger']}; border-radius: 4px;")
        else:
            self.status_indicator.setStyleSheet(f"background-color: {theme_manager.colors['text_muted']}; border-radius: 4px;")
    
    def update_style(self):
        c = theme_manager.colors
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {c['bg_card']};
                border: 1px solid {c['border']};
                border-radius: 12px;
            }}
            QLabel {{
                background: transparent;
                color: {c['text_secondary']};
            }}
        """)
        self.title_label.setStyleSheet(f"color: {c['text_muted']}; background: transparent;")
        self.value_label.setStyleSheet(f"color: {c['text_primary']}; background: transparent;")
        self.update_status(self.status)
    
    def _apply_card_style(self):
        self.update_style()


class ModernProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                background-color: {theme_manager.colors['bg_input']};
                border-radius: 6px;
                height: 8px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {theme_manager.colors['accent']};
                border-radius: 6px;
            }}
        """)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    update_data_signal = pyqtSignal(dict)
    calibration_progress_signal = pyqtSignal(str, int)
    calibration_done_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    
    pause_signal = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self._paused = False
        self._is_minimized = False
        self._low_power_mode = False
        self._initialized = False
        self._init_processors()
        
        self.session_start_time = time.time()
        self.face_lost_time = None
        self._last_calibration_status = {"face": False, "hand": False}
        self._last_face_state = True
        
        self._error_count = 0
        self._max_errors = 10
        
        self._frame_skip = 1
        self._frame_counter = 0
        
        logger.info("VideoThread инициализирован")
    
    def pause(self):
        self._paused = True
        self.pause_signal.emit(True)
        logger.info("VideoThread приостановлен")
    
    def resume(self):
        self._paused = False
        self.pause_signal.emit(False)
        logger.info("VideoThread возобновлен")
    
    def toggle_pause(self):
        if self._paused:
            self.resume()
        else:
            self.pause()
        return self._paused
    
    def set_minimized(self, minimized: bool):
        self._is_minimized = minimized
        if minimized:
            self._low_power_mode = True
            self._frame_skip = 5
        else:
            self._low_power_mode = False
            self._frame_skip = 1
    
    def _init_processors(self):
        try:
            self.face_detector = FaceDetector()
            self.hand_detector = HandDetector()
            self.pose_detector = PoseDetector()
            
            # User profile manager FIRST (needed by FatigueClassifier)
            self.user_profile_manager = UserProfileManager()
            self.user_profile_manager.load_profile('default')
            
            # ML classifiers for fatigue and posture
            self.fatigue_classifier = FatigueClassifier(user_profile_manager=self.user_profile_manager)
            self.posture_classifier = PostureClassifier(use_tf_hub=True)
            
            self.emotion_analyzer = None
            
            self.emotion_model_path = 'models/emotion_model.hdf5'
            if tensorflow_available and os.path.exists(self.emotion_model_path):
                try:
                    self.emotion_analyzer = EmotionDetector(self.emotion_model_path)
                except Exception as e:
                    logger.warning(f"Не удалось загрузить модель эмоций: {e}")
            
            self.db = DatabaseManager("session_data.db")
            self.calibration_manager = None
            
            # Training data collector for ML models
            self.training_collector = TrainingDataCollector()
            
            # Collect data every N frames
            self._collect_data_interval = 30
            self._frame_since_last_collect = 0
            
            self.last_save_time = time.time()
            self.frame_counter = 0
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации процессоров: {e}")
            self._initialized = False
    
    def set_calibration_manager(self, calibration_manager):
        self.calibration_manager = calibration_manager
        if self.hand_detector:
            try:
                self.hand_detector.set_calibration_manager(calibration_manager)
            except AttributeError:
                pass
    
    def _get_calibration_overlay(self, frame):
        if not self.calibration_manager:
            return None
        
        h, w = frame.shape[:2]
        
        try:
            if self.calibration_manager._is_calibrating_face:
                progress = len(self.calibration_manager._face_samples)
                pct = int(progress / 20 * 100)
                
                bar_width = 300
                bar_height = 30
                bar_x = (w - bar_width) // 2
                bar_y = h - 80
                
                cv2.rectangle(frame, (bar_x - 2, bar_y - 2), (bar_x + bar_width + 2, bar_y + bar_height + 2), (50, 50, 50), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (80, 80, 80), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * pct / 100), bar_y + bar_height), (100, 200, 100), -1)
                
                cv2.putText(frame, f"Калибровка лица: {pct}%", (bar_x, bar_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 100), 2)
                cv2.putText(frame, "Смотрите в камеру и не двигайтесь", (bar_x, bar_y + bar_height + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                return {"type": "face", "progress": progress, "pct": pct}
            
            if self.calibration_manager._is_calibrating_hand:
                progress = len(self.calibration_manager._hand_samples)
                pct = int(progress / 20 * 100)
                
                bar_width = 300
                bar_height = 30
                bar_x = (w - bar_width) // 2
                bar_y = h - 80
                
                cv2.rectangle(frame, (bar_x - 2, bar_y - 2), (bar_x + bar_width + 2, bar_y + bar_height + 2), (50, 50, 50), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (80, 80, 80), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * pct / 100), bar_y + bar_height), (100, 150, 220), -1)
                
                cv2.putText(frame, f"Калибровка руки: {pct}%", (bar_x, bar_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 150, 220), 2)
                cv2.putText(frame, "Покажите руку на экране", (bar_x, bar_y + bar_height + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
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
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.error("VideoThread: не удалось открыть камеру")
                self.error_signal.emit("Не удалось открыть камеру")
                return
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            logger.info("VideoThread: камера запущена")
        except Exception as e:
            logger.error(f"VideoThread: ошибка инициализации камеры: {e}")
            self.error_signal.emit(f"Ошибка камеры: {e}")
            return

        last_frame_time = 0
        normal_fps = 30
        low_power_fps = 10
        
        while self._run_flag:
            while self._paused:
                if not self._run_flag:
                    break
                self.msleep(100)
                continue
            
            current_time = time.time()
            target_fps = low_power_fps if self._low_power_mode else normal_fps
            frame_interval = 1.0 / target_fps
            
            if current_time - last_frame_time < frame_interval:
                self.msleep(int((frame_interval - (current_time - last_frame_time)) * 1000))
                continue
            
            last_frame_time = current_time
            
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

            self._frame_counter += 1
            if self._frame_counter % self._frame_skip != 0:
                continue

            try:
                image, results = self.face_detector.process_frame(frame, draw=True)
            except Exception as e:
                logger.error(f"VideoThread: ошибка обнаружения лица: {e}")
                image = frame
                results = None

            data = {
                "ear": 0.35, "mar": 0.0,
                "pitch": 0.0, "emotion": "...",
                "event": None,
                "posture_alert": False,
                "posture_score": 0,
                "posture_level": "good",
                "face_detected": False,
                "hand_detected": False,
                "paused": self._paused,
            }

            current_time = time.time()
            ear = 0.35
            mar = 0.15
            pitch = 0.0
            is_face_valid = False
            emotion = "..."

            try:
                face_landmarks = self.face_detector.get_landmarks(results)
                is_face_valid = face_landmarks is not None
                data["face_detected"] = self.face_detector.is_face_detected(results)
                
                if is_face_valid:
                    ear = calculate_ear(face_landmarks)
                    mar = calculate_mar(face_landmarks)
                    pitch = 0.0
                    
                    if hasattr(self, 'head_pose_estimator') and self.head_pose_estimator:
                        try:
                            pitch, _, _ = self.head_pose_estimator.estimate(face_landmarks, frame.shape[1], frame.shape[0])
                        except:
                            pitch = 0.0
                    
                    data["ear"] = ear
                    data["mar"] = mar
                    data["pitch"] = pitch
                    
                    if self.emotion_analyzer:
                        emotion, _ = self.emotion_analyzer.predict_emotion(frame, face_landmarks)
                    data["emotion"] = emotion
                    
                    # ML-based fatigue classification
                    fatigue_result = self.fatigue_classifier.predict(face_landmarks, frame)
                    fatigue_data = {
                        'fatigue_status': self.fatigue_classifier.get_status_text(fatigue_result['status']),
                        'fatigue_score': fatigue_result.get('fatigue_score', 0),
                        'fatigue_level': fatigue_result.get('fatigue_level', 'normal'),
                        'ear': fatigue_result.get('ear', 0.35),
                        'mar': fatigue_result.get('mar', 0.0),
                        'blink_rate': fatigue_result.get('blink_rate', 0),
                        'trend': fatigue_result.get('trend', 'stable'),
                    }
                    data.update(fatigue_data)
                    
                    # ML-based posture classification (use face mesh as fallback)
                    posture_result = self.posture_classifier.predict_from_face_mesh(
                        face_landmarks, frame.shape[1], frame.shape[0]
                    )
                    posture_data = {
                        'posture_status': posture_result.get('status', 'good'),
                        'posture_score': posture_result.get('confidence', 0) * 100,
                        'is_bad': posture_result.get('status') == 'bad',
                    }
                    data.update(posture_data)
                    
                    # Collect training data periodically
                    self._frame_since_last_collect += 1
                    if self._frame_since_last_collect >= self._collect_data_interval:
                        self._frame_since_last_collect = 0
                        # Add fatigue sample
                        self.training_collector.add_fatigue_sample(
                            features={'ear': ear, 'mar': mar},
                            predicted_label=fatigue_result['status'],
                            confidence=fatigue_result['confidence']
                        )
                        # Add posture sample (if we have pose data)
                        if pose_landmarks:
                            from neurofocus.ml.preprocessing import extract_pose_features
                            pose_features = extract_pose_features(pose_landmarks)
                            if pose_features is not None:
                                self.training_collector.add_posture_sample(
                                    features={
                                        'shoulder_angle': float(pose_features[0]),
                                        'shoulder_diff': float(pose_features[1]),
                                        'forward_lean': float(pose_features[3]),
                                        'torso_tilt': float(pose_features[4]),
                                    },
                                    predicted_label=posture_result['status'],
                                    confidence=posture_result['confidence']
                                )
                    
                    # Generate events based on ML predictions
                    event = None
                    if posture_result['status'] == 'bad' and posture_result['confidence'] > 0.7:
                        event = "Плохая осанка"
                    elif fatigue_result['status'] in ['drowsy', 'sleeping'] and fatigue_result['confidence'] > 0.7:
                        event = "Усталость"
                    if event:
                        data["event"] = event
                    if posture_result['status'] == 'bad':
                        data["posture_alert"] = True
                    
                    if time.time() - self.last_save_time > 1.0:
                        self.db.save_log(
                            ear=ear,
                            mar=mar,
                            pitch=pitch,
                            emotion=emotion,
                            fatigue_status=fatigue_data.get('fatigue_status', 'Normal'),
                            posture_status=posture_data.get('posture_status', 'Good')
                        )
                        self.last_save_time = time.time()
                        
            except Exception as e:
                logger.error(f"VideoThread: ошибка обработки лица: {e}")

            # Initialize posture_data with default values
            posture_data = {
                'posture_status': 'unknown',
                'posture_score': 0,
                'is_bad': False,
            }

            # Pose detection for better posture classification
            pose_landmarks = None
            try:
                # Try TensorFlow Hub MoveNet first (more accurate)
                if self.posture_classifier._tf_hub_estimator and self.posture_classifier._tf_hub_estimator.is_available:
                    tf_hub_result = self.posture_classifier.predict_from_frame(frame)
                    if tf_hub_result['status'] != 'unknown':
                        posture_data['posture_status'] = tf_hub_result['status']
                        posture_data['posture_score'] = tf_hub_result['confidence'] * 100
                        posture_data['is_bad'] = tf_hub_result['status'] == 'bad'
                        data['posture_status'] = tf_hub_result['status']
                        data['posture_score'] = tf_hub_result['confidence'] * 100
                        data['posture_level'] = tf_hub_result['status']
                        data['posture_alert'] = tf_hub_result['status'] == 'bad'
                # Fallback to MediaPipe
                elif self.pose_detector and self.pose_detector.is_available:
                    _, pose_results = self.pose_detector.process_frame(frame, draw=False)
                    if pose_results:
                        pose_landmarks = self.pose_detector.get_landmarks(pose_results)
                        if pose_landmarks and len(pose_landmarks) >= 13:
                            pose_posture_result = self.posture_classifier.predict(pose_landmarks)
                            if pose_posture_result['status'] != 'unknown':
                                posture_data['posture_status'] = pose_posture_result['status']
                                posture_data['posture_score'] = pose_posture_result['confidence'] * 100
                                posture_data['is_bad'] = pose_posture_result['status'] == 'bad'
                                data['posture_status'] = pose_posture_result['status']
                                data['posture_score'] = pose_posture_result['confidence'] * 100
                                data['posture_level'] = pose_posture_result['status']
                                data['posture_alert'] = pose_posture_result['status'] == 'bad'
            except Exception as e:
                logger.error(f"VideoThread: ошибка обработки позы: {e}")

            try:
                if self.hand_detector and self.hand_detector.is_available:
                    _, hand_results = self.hand_detector.process_frame(image, draw=True)
                    hand_detected = self.hand_detector.is_hand_detected(hand_results)
                    data["hand_detected"] = hand_detected
                    data["current_gesture"] = 'none'
                    
                    if hand_detected and hand_results:
                        hand_landmarks = hand_results.hand_landmarks[0]
                        fingers = self.hand_detector.get_fingers_up(hand_landmarks)
                        fingers_count = sum(fingers)
                        
                        gesture_map = {
                            0: 'fist',
                            1: 'one',
                            2: 'peace',
                            3: 'three',
                            4: 'four',
                            5: 'open'
                        }
                        gesture = gesture_map.get(fingers_count, 'none')
                        data["current_gesture"] = gesture
                        
                        if gesture != 'none' and gesture != 'open':
                            cv2.rectangle(image, (5, 5), (200, 35), (0, 0, 0), -1)
                            cv2.putText(image, f"[{gesture.upper()}]", (10, 28), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        if hasattr(self, 'gesture_controller') and self.gesture_controller:
                            try:
                                processed_gesture = self.gesture_controller.process_hand(
                                    hand_landmarks,
                                    image.shape[1],
                                    image.shape[0],
                                    fingers
                                )
                                data["gesture_action"] = processed_gesture
                            except Exception as gest_err:
                                pass
            except Exception as e:
                logger.error(f"VideoThread: ошибка обработки руки: {e}")

            try:
                if self.calibration_manager and is_face_valid:
                    ear_val = ear if is_face_valid else 0.3
                    mar_val = mar if is_face_valid else 0.15
                    pitch_val = pitch if is_face_valid else 0.0
                    
                    self.calibration_manager.auto_calibrate_if_needed(ear_val, mar_val, pitch_val, 0.25)
                    
                    if self.calibration_manager._is_calibrating_face:
                        self.calibration_manager.add_face_sample(ear_val, mar_val, pitch_val)
                        progress = len(self.calibration_manager._face_samples)
                        self.calibration_progress_signal.emit("face", progress)
                        
                        if progress >= 20:
                            self.calibration_manager.finish_face_calibration()
                            self.calibration_done_signal.emit("face")
                            logger.info("Калибровка лица завершена")
            except Exception as e:
                logger.error(f"VideoThread: ошибка калибровки лица: {e}")

            try:
                calib = self.calibration_manager.face_calibration["calibrated"] if self.calibration_manager else False
                if calib != self._last_calibration_status["face"]:
                    self._last_calibration_status["face"] = calib
            except Exception as e:
                pass

            try:
                calib_info = self._get_calibration_overlay(image)
                if calib_info:
                    data["calibration_info"] = calib_info
            except Exception as e:
                logger.error(f"VideoThread: ошибка отрисовки калибровки: {e}")

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

    def stop(self):
        self._run_flag = False
        self.wait(3000)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        theme_manager.set_theme('dark')
        
        self.setWindowTitle("NeuroFocus")
        self.setGeometry(80, 80, 1280, 800)
        self.setStyleSheet(theme_manager.get_complete_stylesheet())

        self.last_event_times = {}
        self._attention_level = 100

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
        app_logo.setStyleSheet(f"color: {theme_manager.colors['accent']};")
        header_layout.addWidget(app_logo)
        
        app_title = QLabel("NeuroFocus")
        app_title.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        app_title.setStyleSheet(f"color: {theme_manager.colors['text_primary']}; letter-spacing: 0.5px;")
        header_layout.addWidget(app_title)
        
        header_layout.addStretch()
        
        self.status_dot = QLabel()
        self.status_dot.setFixedSize(10, 10)
        self.status_dot.setStyleSheet(f"background-color: {theme_manager.colors['good']}; border-radius: 5px;")
        header_layout.addWidget(self.status_dot)
        
        self.status_text = QLabel("Мониторинг активен")
        self.status_text.setFont(QFont("Segoe UI", 12, QFont.Weight.Medium))
        self.status_text.setStyleSheet(f"color: {theme_manager.colors['good']}; margin-left: 8px;")
        header_layout.addWidget(self.status_text)
        
        left_layout.addWidget(header_area)

        video_container = QFrame()
        video_container.setStyleSheet(f"""
            QFrame {{
                background-color: {theme_manager.colors['bg_card']};
                border-radius: 16px;
                border: none;
            }}
        """)
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(0)
        
        video_header = QFrame()
        video_header.setStyleSheet(f"background-color: {theme_manager.colors['bg_card']};")
        video_header.setFixedHeight(48)
        video_header_layout = QHBoxLayout(video_header)
        video_header_layout.setContentsMargins(16, 0, 16, 0)
        
        video_title = QLabel("Видеопоток")
        video_title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        video_title.setStyleSheet(f"color: {theme_manager.colors['text_primary']};")
        video_header_layout.addWidget(video_title)
        
        video_header_layout.addStretch()
        
        video_layout.addWidget(video_header)
        
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._apply_video_label_style()
        video_layout.addWidget(self.image_label, stretch=1)

        left_layout.addWidget(video_container, stretch=1)

        right_panel = QFrame()
        right_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {theme_manager.colors['bg_card']};
                border-radius: 16px;
                border: none;
            }}
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(16)

        metrics_header = QLabel("Показатели")
        metrics_header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        metrics_header.setStyleSheet(f"color: {theme_manager.colors['text_muted']}; letter-spacing: 1px;")
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
                background-color: {theme_manager.colors['bg_input']};
                border-radius: 12px;
                border: none;
            }}
        """)
        posture_layout = QVBoxLayout(posture_frame)
        posture_layout.setContentsMargins(16, 14, 16, 14)
        posture_layout.setSpacing(8)
        
        posture_title = QLabel("Осанка")
        posture_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Medium))
        posture_title.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
        posture_layout.addWidget(posture_title)
        
        posture_value_layout = QHBoxLayout()
        posture_value_layout.setSpacing(10)
        
        self.posture_icon = QLabel("✓")
        self.posture_icon.setFont(QFont("Segoe UI", 20))
        self.posture_icon.setStyleSheet(f"color: {theme_manager.colors['good']};")
        posture_value_layout.addWidget(self.posture_icon)
        
        self.posture_value = QLabel("Хорошая")
        self.posture_value.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        self.posture_value.setStyleSheet(f"color: {theme_manager.colors['text_primary']};")
        self.posture_value.setMinimumWidth(100)
        posture_value_layout.addWidget(self.posture_value)
        
        posture_value_layout.addStretch()
        
        self.posture_angle = QLabel("0°")
        self.posture_angle.setFont(QFont("Segoe UI", 14))
        self.posture_angle.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
        self.posture_angle.setMinimumWidth(40)
        posture_value_layout.addWidget(self.posture_angle)
        
        posture_layout.addLayout(posture_value_layout)
        
        right_layout.addWidget(posture_frame)

        attention_frame = QFrame()
        attention_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {theme_manager.colors['bg_input']};
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
        attention_title.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
        attention_title_layout.addWidget(attention_title)
        
        attention_title_layout.addStretch()
        
        self.attention_percent = QLabel("100%")
        self.attention_percent.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.attention_percent.setStyleSheet(f"color: {theme_manager.colors['good']};")
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
                background-color: {theme_manager.colors['bg_input']};
                border-radius: 12px;
                border: none;
            }}
        """)
        gesture_status_layout = QHBoxLayout(gesture_status_frame)
        gesture_status_layout.setContentsMargins(16, 10, 16, 10)
        
        gesture_label = QLabel("Мышь жестами")
        gesture_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Medium))
        gesture_label.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
        gesture_status_layout.addWidget(gesture_label)
        
        gesture_status_layout.addStretch()
        
        self.gesture_status = QLabel("Выкл")
        self.gesture_status.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.gesture_status.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
        gesture_status_layout.addWidget(self.gesture_status)
        
        right_layout.addWidget(gesture_status_frame)

        session_frame = QFrame()
        session_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {theme_manager.colors['bg_input']};
                border-radius: 12px;
                border: none;
            }}
        """)
        session_layout = QVBoxLayout(session_frame)
        session_layout.setContentsMargins(16, 12, 16, 12)
        session_layout.setSpacing(8)
        
        session_title = QLabel("Статистика сессии")
        session_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Medium))
        session_title.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
        session_layout.addWidget(session_title)
        
        session_row = QHBoxLayout()
        session_row.setSpacing(16)
        
        time_label = QLabel("Время работы:")
        time_label.setFont(QFont("Segoe UI", 11))
        time_label.setStyleSheet(f"color: {theme_manager.colors['text_secondary']};")
        session_row.addWidget(time_label)
        
        self.session_time_label = QLabel("00:00:00")
        self.session_time_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.session_time_label.setStyleSheet(f"color: {theme_manager.colors['accent']};")
        session_row.addWidget(self.session_time_label)
        
        session_row.addStretch()
        
        events_label = QLabel("Событий:")
        events_label.setFont(QFont("Segoe UI", 11))
        events_label.setStyleSheet(f"color: {theme_manager.colors['text_secondary']};")
        session_row.addWidget(events_label)
        
        self.session_events_label = QLabel("0")
        self.session_events_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.session_events_label.setStyleSheet(f"color: {theme_manager.colors['warning']};")
        session_row.addWidget(self.session_events_label)
        
        session_layout.addLayout(session_row)
        
        right_layout.addWidget(session_frame)

        events_header = QLabel("События")
        events_header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        events_header.setStyleSheet(f"color: {theme_manager.colors['text_muted']}; letter-spacing: 1px;")
        right_layout.addWidget(events_header)

        self.event_list = QListWidget()
        self.event_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {theme_manager.colors['bg_input']};
                border: none;
                border-radius: 12px;
                padding: 12px;
                font-size: 12px;
                color: {theme_manager.colors['text_secondary']};
            }}
            QListWidget::item {{
                padding: 6px 4px;
                border-bottom: 1px solid {theme_manager.colors['border']};
            }}
            QListWidget::item:last-child {{
                border-bottom: none;
            }}
        """)
        self.event_list.setFixedHeight(130)
        self.add_log_event("Система запущена", color=theme_manager.colors['text_muted'])
        right_layout.addWidget(self.event_list)
        
        self.graph_frame = QFrame()
        self.graph_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {theme_manager.colors['bg_card']};
                border: 1px solid {theme_manager.colors['border']};
                border-radius: 12px;
                padding: 8px;
            }}
        """)
        graph_layout = QVBoxLayout(self.graph_frame)
        graph_layout.setContentsMargins(8, 8, 8, 8)
        
        graph_title = QLabel("Уровень усталости")
        graph_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
        graph_title.setStyleSheet(f"color: {theme_manager.colors['text_muted']}; background: transparent;")
        graph_layout.addWidget(graph_title)
        
        self.fatigue_graph = MiniFatigueGraph(max_points=60)
        graph_layout.addWidget(self.fatigue_graph)
        
        right_layout.addWidget(self.graph_frame)

        right_layout.addStretch()

        button_group = QFrame()
        button_group.setStyleSheet("background-color: transparent;")
        button_layout = QVBoxLayout(button_group)
        button_layout.setSpacing(10)
        
        self.btn_stats = ModernPrimaryButton("📊  Аналитика")
        self.btn_stats.clicked.connect(self.open_stats)
        self.btn_stats.setStyleSheet(theme_manager.get_button_style(custom_color="#22C55E"))
        button_layout.addWidget(self.btn_stats)
        
        self.btn_progress = ModernButton("📈  Прогресс")
        self.btn_progress.clicked.connect(self.open_progress)
        button_layout.addWidget(self.btn_progress)
        
        self.btn_pomodoro = ModernButton("🍅  Pomodoro")
        self.btn_pomodoro.clicked.connect(self.open_pomodoro)
        button_layout.addWidget(self.btn_pomodoro)
        
        self.btn_pause = ModernButton("⏸  Пауза")
        self.btn_pause.clicked.connect(self.toggle_pause)
        button_layout.addWidget(self.btn_pause)
        
        self.btn_gesture = ModernButton("🖱️  Управление мышью")
        self.btn_gesture.clicked.connect(self.toggle_gesture)
        button_layout.addWidget(self.btn_gesture)
        
        self.btn_help = ModernButton("❓  Жесты")
        self.btn_help.clicked.connect(self.show_gesture_help)
        button_layout.addWidget(self.btn_help)

        self.btn_settings = ModernButton("⚙  Настройки")
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_settings.setStyleSheet(theme_manager.get_button_style(custom_color="#3B82F6"))
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

        self.calibration_manager = CalibrationManager()
        
        self.video_thread = VideoThread()
        self.video_thread.set_calibration_manager(self.calibration_manager)
        
        self.gesture_controller = GestureController(calibration_manager=self.calibration_manager)
        self.video_thread.gesture_controller = self.gesture_controller
        
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
        
        self.pomodoro_dialog = None
        
        self.fatigue_warning_cooldown = 0
        
        logger.info("Приложение запущено успешно")
    
    def setup_tray(self):
        self.tray_icon = QSystemTrayIcon(self)
        
        tray_menu = QMenu()
        tray_menu.setStyleSheet(f"""
            QMenu {{
                background-color: {theme_manager.colors['bg_card']};
                color: {theme_manager.colors['text_primary']};
                border: 1px solid {theme_manager.colors['border']};
                border-radius: 8px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 8px 16px;
                border-radius: 4px;
            }}
            QMenu::item:selected {{
                background-color: {theme_manager.colors['bg_input']};
            }}
        """)
        
        show_action = tray_menu.addAction("Показать NeuroFocus")
        show_action.triggered.connect(self.show_from_tray)
        show_action.triggered.connect(self.activateWindow)
        
        tray_menu.addSeparator()
        
        minimize_action = tray_menu.addAction("Свернуть в трей")
        minimize_action.triggered.connect(self.minimize_to_tray)
        
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
            pixmap.fill(QColor(theme_manager.colors['accent']))
            self.tray_icon.setIcon(QIcon(pixmap))
        
        self.tray_icon.show()
    
    def tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show_from_tray()
            self.activateWindow()
    
    def minimize_to_tray(self):
        self.hide()
        if hasattr(self, 'video_thread'):
            self.video_thread.set_minimized(True)
        if self.tray_icon:
            self.tray_icon.showMessage(
                "NeuroFocus",
                "Приложение свернуто в трей. Работа продолжается.",
                QSystemTrayIcon.MessageIcon.Information,
                2000
            )
    
    def show_from_tray(self):
        self.show()
        self.activateWindow()
        if hasattr(self, 'video_thread'):
            self.video_thread.set_minimized(False)
    
    def toggle_pause(self):
        if hasattr(self, 'video_thread'):
            is_paused = self.video_thread.toggle_pause()
            if is_paused:
                self.btn_pause.setText("▶  Возобновить")
                self.add_log_event("⏸  Мониторинг приостановлен", color=theme_manager.colors['warning'])
            else:
                self.btn_pause.setText("⏸  Пауза")
                self.add_log_event("▶  Мониторинг возобновлен", color=theme_manager.colors['good'])
    
    def _apply_video_label_style(self):
        c = theme_manager.colors
        video_bg = c.get('bg_input', '#2D2D3A')
        if hasattr(self, 'image_label'):
            self.image_label.setStyleSheet(f"background-color: {video_bg}; border-radius: 0 0 16px 16px;")
    
    def apply_theme(self):
        c = theme_manager.colors
        
        self.setStyleSheet(theme_manager.get_complete_stylesheet())
        
        self._apply_video_label_style()
        
        for widget in [self.btn_stats, self.btn_progress, self.btn_pomodoro, self.btn_pause, 
                       self.btn_gesture, self.btn_help, self.btn_settings, self.btn_stop]:
            if hasattr(widget, '_update_style'):
                widget._update_style()
            else:
                widget.setStyleSheet(theme_manager.get_button_style())
        
        if hasattr(self, 'app_logo'):
            self.app_logo.setStyleSheet(f"color: {c['accent']};")
        if hasattr(self, 'app_title'):
            self.app_title.setStyleSheet(f"color: {c['text_primary']}; letter-spacing: 0.5px;")
        
        if hasattr(self, 'status_dot'):
            self.status_dot.setStyleSheet(f"background-color: {c['good']}; border-radius: 5px;")
        if hasattr(self, 'status_text'):
            self.status_text.setStyleSheet(f"color: {c['good']}; margin-left: 8px;")
        
        if hasattr(self, 'video_panel'):
            self.video_panel.setStyleSheet(f"background-color: {c['bg_card']}; border-radius: 16px;")
        
        if hasattr(self, 'fatigue_graph'):
            self.fatigue_graph.update_colors()
        
        if hasattr(self, 'graph_frame'):
            self.graph_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {c['bg_card']};
                    border: 1px solid {c['border']};
                    border-radius: 12px;
                    padding: 8px;
                }}
            """)
        
        if hasattr(self, 'event_list'):
            self.event_list.setStyleSheet(f"""
                QListWidget {{
                    background-color: {c['bg_input']};
                    border: none;
                    border-radius: 8px;
                    padding: 4px;
                    color: {c['text_secondary']};
                }}
                QListWidget::item {{
                    padding: 6px;
                    border-bottom: 1px solid {c['border']};
                }}
                QListWidget::item:last-child {{
                    border-bottom: none;
                }}
            """)
        
        if hasattr(self, 'emotion_card') and hasattr(self.emotion_card, 'update_style'):
            self.emotion_card.update_style()
        if hasattr(self, 'attention_card') and hasattr(self.attention_card, 'update_style'):
            self.attention_card.update_style()
        
        if hasattr(self, 'posture_icon'):
            self.posture_icon.setStyleSheet(f"color: {c['good']};")
        if hasattr(self, 'posture_value'):
            self.posture_value.setStyleSheet(f"color: {c['text_primary']};")
        if hasattr(self, 'posture_angle'):
            self.posture_angle.setStyleSheet(f"color: {c['text_muted']};")
        
        if hasattr(self, 'attention_percent'):
            self.attention_percent.setStyleSheet(f"color: {c['good']};")
        
        if hasattr(self, 'gesture_status'):
            self.gesture_status.setStyleSheet(f"color: {c['text_muted']};")
        
        if hasattr(self, 'session_time_label'):
            self.session_time_label.setStyleSheet(f"color: {c['accent']};")
        
        if hasattr(self, 'session_events_label'):
            self.session_events_label.setStyleSheet(f"color: {c['warning']};")
        
        if hasattr(self, 'statusBar'):
            self.statusBar().setStyleSheet(f"background-color: {c['bg_card']}; color: {c['text_secondary']};")
        
        if hasattr(self, 'video_thread'):
            self.video_thread.set_minimized(self.isMinimized())
    
    def on_calibration_progress(self, calib_type, progress):
        pct = int(progress / 20 * 100)
        if calib_type == "face":
            sound_manager.calibration_start()
        self.add_log_event(f"Калибровка {calib_type}: {pct}%", theme_manager.colors['text_muted'])
    
    def on_calibration_done(self, calib_type):
        self.add_log_event(f"Калибровка {calib_type} завершена ✓", theme_manager.colors['good'])

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
                self.add_log_event("Лицо найдено ✓", theme_manager.colors['good'])
            
            self.status_dot.setStyleSheet(f"background-color: {theme_manager.colors['good']}; border-radius: 5px;")
            self.status_text.setText("Мониторинг активен")
            self.status_text.setStyleSheet(f"color: {theme_manager.colors['good']}; margin-left: 8px;")
            self._last_face_state = True
        else:
            if self._last_face_state:
                self.face_lost_time = time.time()
            
            self.status_dot.setStyleSheet(f"background-color: {theme_manager.colors['warning']}; border-radius: 5px;")
            self.status_text.setText("Лицо не найдено")
            self.status_text.setStyleSheet(f"color: {theme_manager.colors['warning']}; margin-left: 8px;")
            self._last_face_state = False
            
            if self.face_lost_time and not self._auto_pause_triggered:
                lost_duration = time.time() - self.face_lost_time
                if lost_duration > 5:
                    self._auto_pause_triggered = True
                    self.face_lost_count += 1
                    sound_manager.warning()
                    self.add_log_event("⚠️ Лицо потеряно >5 сек", theme_manager.colors['warning'])
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
        
        if hasattr(self, 'fatigue_graph'):
            self.fatigue_graph.update_data(fatigue_score)
        
        if attention_level >= 70:
            self.attention_card.update_status("good")
            self.attention_percent.setStyleSheet(f"color: {theme_manager.colors['good']};")
        elif attention_level >= 40:
            self.attention_card.update_status("warning")
            self.attention_percent.setStyleSheet(f"color: {theme_manager.colors['warning']};")
        else:
            self.attention_card.update_status("danger")
            self.attention_percent.setStyleSheet(f"color: {theme_manager.colors['danger']};")
            
        self.attention_card.update_value(f"{attention_level}", "%")
        
        if self.pomodoro_dialog and self.pomodoro_dialog.is_work:
            self.fatigue_warning_cooldown -= 1
            if self.fatigue_warning_cooldown <= 0:
                if fatigue_level == "severe":
                    self.pomodoro_dialog.suggest_break("Сильная усталость")
                    self.fatigue_warning_cooldown = 60
                    sound_manager.attention()
                elif fatigue_level == "moderate":
                    self.pomodoro_dialog.suggest_break("Умеренная усталость")
                    self.fatigue_warning_cooldown = 120
                elif fatigue_level == "mild" and data.get("trend") == "decreasing":
                    self.pomodoro_dialog.suggest_break("Усталость нарастает")
                    self.fatigue_warning_cooldown = 180

        pitch = data["pitch"]

        if data["posture_alert"]:
            self.posture_value.setText("Плохая")
            self.posture_angle.setText(f"{int(pitch)}°")
            self.posture_icon.setText("!")
            self.posture_icon.setStyleSheet(f"color: {theme_manager.colors['danger']}; font-weight: bold;")
            self.posture_value.setStyleSheet(f"color: {theme_manager.colors['danger']};")
        else:
            self.posture_value.setText("Хорошая")
            self.posture_angle.setText(f"{int(pitch)}°")
            self.posture_icon.setText("✓")
            self.posture_icon.setStyleSheet(f"color: {theme_manager.colors['good']};")
            self.posture_value.setStyleSheet(f"color: {theme_manager.colors['text_primary']};")

        current_event = data["event"]
        if current_event:
            now = time.time()

            if "осанка" in current_event.lower():
                tracking_key = "posture_any"
                cooldown = COOLDOWN_POSTURE_EVENT
                color = theme_manager.colors['danger']
            else:
                tracking_key = current_event
                cooldown = COOLDOWN_YAWING_EVENT
                color = theme_manager.colors['warning']

            last_time = self.last_event_times.get(tracking_key, 0)

            if now - last_time > cooldown:
                self.add_log_event(current_event, color)
                self.last_event_times[tracking_key] = now

    def open_stats(self):
        stats_dialog = StatsWindow()
        stats_dialog.exec()
    
    def open_progress(self):
        from neurofocus.windows.progress import ProgressWindow
        progress_dialog = ProgressWindow(self)
        progress_dialog.exec()
    
    def open_pomodoro(self):
        from neurofocus.windows.pomodoro import PomodoroTimer
        if not hasattr(self, 'pomodoro_dialog') or self.pomodoro_dialog is None:
            self.pomodoro_dialog = PomodoroTimer(self)
        self.pomodoro_dialog.show()
        self.pomodoro_dialog.activateWindow()

    def close_app(self):
        logger.info("Завершение работы приложения...")
        
        if hasattr(self, 'tray_icon') and self.tray_icon:
            self.tray_icon.hide()
        
        if hasattr(self, 'notify_timer') and self.notify_timer:
            self.notify_timer.stop()
        
        if hasattr(self, 'session_timer') and self.session_timer:
            self.session_timer.stop()
        
        if hasattr(self, 'video_thread'):
            self.video_thread.stop()
            self.video_thread.wait(3000)
        
        if hasattr(self, 'pomodoro_dialog') and self.pomodoro_dialog:
            try:
                self.pomodoro_dialog.close()
            except:
                pass
        
        try:
            from neurofocus.utils.progress_tracker import ProgressTracker
            tracker = ProgressTracker()
            tracker.update_daily_progress()
            logger.info("Ежедневный прогресс обновлен")
        except Exception as e:
            logger.error(f"Ошибка обновления прогресса: {e}")
        
        logger.info("Приложение завершено")
        self.close()
        
        QApplication.quit()
    
    def closeEvent(self, event):
        minimize_to_tray = self.notify_manager.settings.get("minimize_to_tray", True)
        if minimize_to_tray:
            event.ignore()
            self.minimize_to_tray()
        else:
            self.close_app()
            event.accept()

    def open_settings(self):
        dialog = SettingsWindow(self.notify_manager.settings, self.calibration_manager)
        dialog.exec()
        if dialog.result_settings:
            self.notify_manager.update_settings(dialog.result_settings)
            self._refresh_button_colors()
            self.show_notification("Настройки", "Параметры обновлены")
    
    def _refresh_button_colors(self):
        if hasattr(self, 'btn_stats'):
            self.btn_stats.setStyleSheet(theme_manager.get_button_style(custom_color="#22C55E"))
        if hasattr(self, 'btn_settings'):
            self.btn_settings.setStyleSheet(theme_manager.get_button_style(custom_color="#3B82F6"))
    
    def toggle_gesture(self):
        if hasattr(self.video_thread, 'gesture_controller'):
            enabled = self.video_thread.gesture_controller.toggle()
            if enabled:
                self.gesture_status.setText("Вкл")
                self.gesture_status.setStyleSheet(f"color: {theme_manager.colors['good']};")
                self.btn_gesture.setText("🖱️  Мышь ВКЛ")
            else:
                self.gesture_status.setText("Выкл")
                self.gesture_status.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
                self.btn_gesture.setText("🖱️  Управление мышью")
    
    def show_gesture_help(self):
        from neurofocus.windows.help import GestureHelpWindow
        help_dialog = GestureHelpWindow(self)
        help_dialog.exec()
    
    def on_escape_pressed(self):
        if hasattr(self.video_thread, 'gesture_controller'):
            if self.video_thread.gesture_controller.enabled:
                self.video_thread.gesture_controller.disable()
                self.gesture_status.setText("Выкл")
                self.gesture_status.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
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

    def check_notifications(self):
        result = self.notify_manager.check_conditions()
        if result:
            title, msg = result
            self.show_notification(title, msg)

    def show_notification(self, title, msg):
        self.toast = ToastNotification(title, msg)
        self.toast.show_toast()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

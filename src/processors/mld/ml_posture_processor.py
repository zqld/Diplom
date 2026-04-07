from typing import Optional
from src.config_manager import config_manager
from src.logger import logger
import numpy as np
from collections import deque
import os
import time


class MLPostureProcessor:
    """
    ML-версия процессора осанки с использованием LSTM модели.
    Обеспечивает более точное распознавание нарушений осанки на основе
    временных рядов данных о положении головы и плеч.
    """
    
    def __init__(self):
        self._config = config_manager.posture
        self._window_size = self._config.get('window_size_seconds', 30)
        
        self._last_event_time = 0
        self._cooldown = 5.0
        
        self._init_models()
        
        self._last_alert_time = 0
        self._alert_cooldown = 10.0
        
    def _init_models(self):
        """Инициализировать ML модели."""
        self.lstm_model = None
        self._model_loaded = False
        self._tf_available = False
        
        # Проверяем доступность TensorFlow
        try:
            import tensorflow as tf
            self._tf_available = True
        except Exception as e:
            logger.warning(f"TensorFlow недоступен: {e}")
            self._model_loaded = False
            return
        
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        models_dir = os.path.join(base_path, 'models')
        
        lstm_path = os.path.join(models_dir, 'posture_lstm.keras')
        
        try:
            if os.path.exists(lstm_path):
                try:
                    self.lstm_model = tf.keras.models.load_model(lstm_path, compile=False)
                    logger.info(f"LSTM модель осанки загружена: {lstm_path}")
                    self._model_loaded = True
                except Exception as e:
                    logger.warning(f"Не удалось загрузить LSTM модель осанки: {e}")
                    
        except Exception as e:
            logger.warning(f"Ошибка при загрузке моделей: {e}")
            self._model_loaded = False
        
        self.sequence_buffer = deque(maxlen=30)
        self.prediction_history = deque(maxlen=10)
        
        self.classes = ['good', 'fair', 'bad']
    
    def process(self, landmarks, frame_width: int, frame_height: int, pitch: float, current_time: float) -> dict:
        result = {
            'is_bad': False,
            'posture_score': 0,
            'posture_level': 'good',
            'posture_status': 'Good',
            'head_tilt': 0,
            'head_forward': 0,
            'event': None,
            'pitch_alert': False,
            'model_used': 'classic'
        }
        
        if not landmarks:
            return result
        
        # Проверяем что landmarks валидны
        try:
            if hasattr(landmarks, 'landmark'):
                lm = landmarks.landmark
            elif isinstance(landmarks, list):
                lm = landmarks
            else:
                return result
            
            if not lm or len(lm) < 15:
                return result
                
            # Проверяем ключевые точки
            if len(lm) > 1 and not hasattr(lm[1], 'x'):
                return result
                
        except:
            return result
        
        try:
            head_tilt, head_forward, shoulder_diff = self._extract_features(landmarks, frame_width, frame_height)
            
            result['head_tilt'] = head_tilt
            result['head_forward'] = head_forward
            
            # Используем pitch из face_processor как основной показатель осанки
            # Это более стабильный метод чем расчёт по landmarks
            if self._model_loaded:
                result = self._process_ml(head_tilt, head_forward, shoulder_diff, pitch, current_time)
            else:
                result = self._process_fallback(head_tilt, head_forward, shoulder_diff, pitch, current_time)
            
            if result.get('event') and current_time - self._last_event_time > self._cooldown:
                self._last_event_time = current_time
            elif result.get('event'):
                result['event'] = None
            
            if abs(pitch) > self._config.get('pitch_threshold', 25):
                result['pitch_alert'] = True
                if current_time - self._last_alert_time > self._alert_cooldown:
                    result['event'] = f"Наклон головы ({int(pitch)}°)"
                    self._last_alert_time = current_time
            
            logger.debug(f"Posture (ML): level={result['posture_level']}, tilt={head_tilt:.1f}, model={result.get('model_used', 'classic')}")
            
        except Exception as e:
            logger.error(f"Ошибка в ML PostureProcessor: {e}")
            result = self._process_fallback(0, 0, 0, pitch, current_time)
        
        return result
    
    def _extract_features(self, landmarks, frame_width: int, frame_height: int):
        """Извлечь признаки для ML модели."""
        try:
            if hasattr(landmarks, 'landmark'):
                lm = landmarks.landmark
            elif isinstance(landmarks, list):
                lm = landmarks
            else:
                return 0, 0, 0
            
            if len(lm) < 1:
                return 0, 0, 0
            
            nose = lm[1] if len(lm) > 1 else None
            left_shoulder = lm[11] if len(lm) > 11 else None
            right_shoulder = lm[12] if len(lm) > 12 else None
            
            head_tilt = 0
            if left_shoulder and right_shoulder and left_shoulder.y and right_shoulder.y:
                shoulder_diff_y = abs(right_shoulder.y - left_shoulder.y) * frame_height
                shoulder_diff_x = abs(right_shoulder.x - left_shoulder.x) * frame_width
                
                if shoulder_diff_x > 5:
                    head_tilt = np.degrees(np.arctan2(shoulder_diff_y, shoulder_diff_x + 1e-6))
                else:
                    head_tilt = 0
            
            head_forward = 0
            if nose and left_shoulder and right_shoulder and nose.y:
                mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
                head_forward = abs(nose.y - mid_shoulder_y)
            
            shoulder_diff = 0
            if left_shoulder and right_shoulder and left_shoulder.y and right_shoulder.y:
                shoulder_diff = abs(left_shoulder.y - right_shoulder.y) * frame_height
            
            head_tilt = max(-30, min(30, head_tilt))
            head_forward = max(0, min(1, head_forward))
            shoulder_diff = max(0, min(100, shoulder_diff))
            
            return head_tilt, head_forward, shoulder_diff
            
        except Exception as e:
            logger.debug(f"Error extracting posture features: {e}")
            return 0, 0, 0
    
    def _process_ml(self, head_tilt: float, head_forward: float, shoulder_diff: float, pitch: float, current_time: float) -> dict:
        """Обработка с использованием ML модели + fallback по pitch."""
        
        features = [
            head_tilt / 45.0,
            head_forward * 10,
            shoulder_diff / 100,
            pitch / 90.0
        ]
        
        self.sequence_buffer.append(features)
        
        # Если нет достаточно данных для LSTM - используем классический метод
        if len(self.sequence_buffer) < 10:
            return self._process_fallback(head_tilt, head_forward, shoulder_diff, pitch, current_time)
        
        # Проверяемpitch - если очень большой наклон, сразу помечаем как плохой
        abs_pitch = abs(pitch)
        if abs_pitch > 40:
            return {
                'is_bad': True,
                'posture_score': 70,
                'posture_level': 'bad',
                'posture_status': 'Bad',
                'head_tilt': head_tilt,
                'head_forward': head_forward,
                'event': "Плохая осанка",
                'pitch_alert': True,
                'model_used': 'lstm'
            }
        
        sequence = np.array(list(self.sequence_buffer))
        sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
        
        prediction = None
        try:
            if self.lstm_model:
                pred = self.lstm_model.predict(sequence, verbose=0)
                if len(pred.shape) > 1:
                    pred = pred[0]
                prediction = pred
        except Exception as e:
            logger.debug(f"LSTM posture prediction error: {e}")
        
        if prediction is not None:
            self.prediction_history.append(pred)
            
            avg_pred = np.mean(self.prediction_history, axis=0)
            class_idx = np.argmax(avg_pred)
            confidence = avg_pred[class_idx]
            
            status = self.classes[class_idx] if class_idx < len(self.classes) else 'good'
            
            # Если pitch умеренно большой, повышаем вероятность bad
            if abs_pitch > 25 and status == 'fair':
                status = 'bad'
            
            is_bad = status == 'bad'
            posture_score = int((1 - confidence) * 100) if confidence < 0.5 else int((1 - confidence) * 40)
            posture_score = max(0, min(100, posture_score))
            
            # Дополнительно повышаем score при большом pitch
            if abs_pitch > 30:
                posture_score = max(posture_score, 60)
            elif abs_pitch > 20:
                posture_score = max(posture_score, 40)
            
            return {
                'is_bad': is_bad,
                'posture_score': posture_score,
                'posture_level': status,
                'posture_status': status.capitalize(),
                'head_tilt': head_tilt,
                'head_forward': head_forward,
                'event': f"Плохая осанка" if is_bad else None,
                'pitch_alert': abs(pitch) > 25,
                'model_used': 'lstm'
            }
        else:
            return self._process_fallback(head_tilt, head_forward, shoulder_diff, pitch, current_time)
    
    def _process_fallback(self, head_tilt: float, head_forward: float, shoulder_diff: float, pitch: float, current_time: float) -> dict:
        """Классический метод - используем pitch как основной показатель."""
        
        is_bad = False
        posture_score = 0
        
        # Pitch - основной показатель (наклон головы вперёд/назад)
        # pitch > 0 = голова наклонена вниз (смотрит вниз)
        # pitch < 0 = голова запрокинута назад
        abs_pitch = abs(pitch)
        if abs_pitch > 40:
            posture_score += 50
            is_bad = True
        elif abs_pitch > 30:
            posture_score += 35
            is_bad = True
        elif abs_pitch > 20:
            posture_score += 20
        
        # Tilt (наклон вбок)
        if abs(head_tilt) > 25:
            posture_score += 30
            is_bad = True
        elif abs(head_tilt) > 15:
            posture_score += 15
        
        # Forward (голова вперёд)
        if head_forward > 0.20:
            posture_score += 25
            is_bad = True
        elif head_forward > 0.12:
            posture_score += 10
        
        # Shoulder diff (перекос плеч)
        if shoulder_diff > 50:
            posture_score += 20
            is_bad = True
        elif shoulder_diff > 30:
            posture_score += 10
        
        posture_score = min(100, posture_score)
        
        if posture_score >= 50:
            level = 'bad'
        elif posture_score >= 25:
            level = 'fair'
        else:
            level = 'good'
        
        event = None
        if is_bad:
            event = "Плохая осанка"
        
        return {
            'is_bad': is_bad,
            'posture_score': posture_score,
            'posture_level': level,
            'posture_status': level.capitalize(),
            'head_tilt': head_tilt,
            'head_forward': head_forward,
            'event': event,
            'pitch_alert': abs(pitch) > 30,
            'model_used': 'classic'
        }
    
    def reset(self):
        self.sequence_buffer.clear()
        self.prediction_history.clear()
        self._last_event_time = 0
        self._last_alert_time = 0
    
    def reload_config(self):
        self._config = config_manager.posture
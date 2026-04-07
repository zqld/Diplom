from typing import Optional
from src.config_manager import config_manager
from src.logger import logger
import numpy as np
from collections import deque
import os
import time


class MLFatigueProcessor:
    """
    ML-версия процессора усталости с использованием LSTM модели.
    Обеспечивает более точное распознавание усталости на основе 
    временных рядов данных.
    """
    
    def __init__(self):
        self._config = config_manager.fatigue
        self._window_size = self._config.get('window_size_seconds', 30)
        
        self._last_event_time = 0
        self._cooldown = 2.0
        
        self._init_models()
        
        self._last_event_time = 0
        self._cooldown = 2.0
        
    def _init_models(self):
        """Инициализировать ML модели."""
        self.lstm_model = None
        self.cnn_model = None
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
        
        lstm_path = os.path.join(models_dir, 'fatigue_lstm_best.keras')
        cnn_path = os.path.join(models_dir, 'fatigue_cnn.keras')
        
        try:
            if os.path.exists(lstm_path):
                try:
                    self.lstm_model = tf.keras.models.load_model(lstm_path, compile=False)
                    logger.info(f"LSTM модель усталости загружена: {lstm_path}")
                except Exception as e:
                    logger.warning(f"Не удалось загрузить LSTM модель: {e}")
            
            if os.path.exists(cnn_path):
                try:
                    self.cnn_model = tf.keras.models.load_model(cnn_path, compile=False)
                    logger.info(f"CNN модель усталости загружена: {cnn_path}")
                except Exception as e:
                    logger.warning(f"Не удалось загрузить CNN модель: {e}")
            
            self._model_loaded = self.lstm_model is not None or self.cnn_model is not None
            
        except Exception as e:
            logger.warning(f"Ошибка при загрузке моделей: {e}")
            self._model_loaded = False
        
        self.sequence_buffer = deque(maxlen=30)
        self.prediction_history = deque(maxlen=10)
        
        self.classes = ['awake', 'drowsy', 'sleeping']
    
    def process(self, ear: float, mar: float, pitch: float, emotion: str, current_time: float) -> dict:
        result = {
            'fatigue_score': 0,
            'fatigue_level': 'normal',
            'fatigue_status': 'Normal',
            'event': None,
            'cooldown_active': False,
            'model_used': 'classic'
        }
        
        try:
            if self._model_loaded:
                result = self._process_ml(ear, mar, pitch, current_time)
            else:
                result = self._process_fallback(ear, mar, pitch, emotion, current_time)
            
            if current_time - self._last_event_time < self._cooldown:
                result['cooldown_active'] = True
            
            if mar > self._config.get('mar_threshold_yawn', 0.6):
                result['event'] = "Зевок"
                result['fatigue_status'] = "Yawning"
                if current_time - self._last_event_time >= self._cooldown:
                    self._last_event_time = current_time
                    
            elif result['fatigue_level'] == "severe":
                result['event'] = "Сильная усталость"
                result['fatigue_status'] = "Fatigued"
                if current_time - self._last_event_time >= self._cooldown:
                    self._last_event_time = current_time
                    
            elif result['fatigue_level'] == "moderate":
                result['event'] = "Усталость"
                result['fatigue_status'] = "Tired"
                
            elif result['fatigue_level'] == "mild":
                result['event'] = "Лёгкая усталость"
                result['fatigue_status'] = "Mild"
            
            logger.debug(f"Fatigue (ML): level={result['fatigue_level']}, score={result['fatigue_score']}, model={result.get('model_used', 'classic')}")
            
        except Exception as e:
            logger.error(f"Ошибка в ML FatigueProcessor: {e}")
            result = self._process_fallback(ear, mar, pitch, emotion, current_time)
        
        return result
    
    def _process_ml(self, ear: float, mar: float, pitch: float, current_time: float) -> dict:
        """Обработка с использованием ML моделей."""
        
        features = [ear, mar, pitch / 90.0]
        
        if len(self.sequence_buffer) > 0:
            last_features = list(self.sequence_buffer)[-1]
            features.extend(last_features)
        else:
            features.extend([ear, mar, pitch / 90.0])
        
        self.sequence_buffer.append(features[:3])
        
        input_sequence = self._prepare_sequence()
        
        prediction = None
        model_used = None
        
        if self.lstm_model is not None and input_sequence is not None:
            try:
                pred = self.lstm_model.predict(input_sequence, verbose=0)
                if len(pred.shape) > 1:
                    pred = pred[0]
                prediction = pred
                model_used = 'lstm'
            except Exception as e:
                logger.debug(f"LSTM prediction error: {e}")
        
        if prediction is None and self.cnn_model is not None:
            try:
                input_features = np.array(features[:16]).reshape(1, -1)
                pred = self.cnn_model.predict(input_features, verbose=0)
                if len(pred.shape) > 1:
                    pred = pred[0]
                prediction = pred
                model_used = 'cnn'
            except Exception as e:
                logger.debug(f"CNN prediction error: {e}")
        
        if prediction is not None:
            self.prediction_history.append(pred)
            
            avg_pred = np.mean(self.prediction_history, axis=0)
            class_idx = np.argmax(avg_pred)
            confidence = avg_pred[class_idx]
            
            status = self.classes[class_idx] if class_idx < len(self.classes) else 'awake'
            
            fatigue_score = int((1 - confidence) * 100) if confidence < 0.5 else int((1 - confidence) * 50)
            fatigue_score = max(0, min(100, fatigue_score))
            
            level = self._status_to_level(status)
            
            return {
                'fatigue_score': fatigue_score,
                'fatigue_level': level,
                'fatigue_status': status.capitalize(),
                'event': None,
                'cooldown_active': False,
                'model_used': model_used,
                'confidence': float(confidence)
            }
        else:
            return self._process_fallback(ear, mar, pitch, None, current_time)
    
    def _prepare_sequence(self):
        """Подготовить последовательность для LSTM модели."""
        if len(self.sequence_buffer) < 10:
            return None
        
        sequence = np.array(list(self.sequence_buffer))
        
        if len(sequence.shape) == 2:
            sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
        
        return sequence
    
    def _status_to_level(self, status: str) -> str:
        """Преобразовать статус модели в уровень."""
        status = status.lower()
        if status == 'sleeping':
            return 'severe'
        elif status == 'drowsy':
            return 'moderate'
        else:
            return 'normal'
    
    def _process_fallback(self, ear: float, mar: float, pitch: float, emotion: str, current_time: float) -> dict:
        """Классический метод при недоступности моделей."""
        
        fatigue_score = 0
        
        if ear < 0.18:
            fatigue_score += 60
        elif ear < 0.22:
            fatigue_score += 40
        elif ear < 0.25:
            fatigue_score += 20
        
        if mar > 0.5:
            fatigue_score += 25
        elif mar > 0.4:
            fatigue_score += 10
        
        if abs(pitch) > 30:
            fatigue_score += 15
        
        fatigue_score = min(100, fatigue_score)
        
        if fatigue_score >= 70:
            level = 'severe'
        elif fatigue_score >= 40:
            level = 'moderate'
        elif fatigue_score >= 20:
            level = 'mild'
        else:
            level = 'normal'
        
        return {
            'fatigue_score': fatigue_score,
            'fatigue_level': level,
            'fatigue_status': level.capitalize(),
            'event': None,
            'cooldown_active': False,
            'model_used': 'classic'
        }
    
    def reset(self):
        self.sequence_buffer.clear()
        self.prediction_history.clear()
        self._last_event_time = 0
    
    def reload_config(self):
        self._config = config_manager.fatigue
        self._window_size = self._config.get('window_size_seconds', 30)
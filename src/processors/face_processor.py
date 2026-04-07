from typing import Optional, Tuple
import numpy as np
from src.face_core import FaceMeshDetector
from src.pose_estimator import HeadPoseEstimator
from src.geometry import calculate_ear, calculate_mar
from src.config_manager import config_manager
from src.logger import logger


class FaceProcessor:
    def __init__(self):
        self.detector = FaceMeshDetector()
        self.pose_estimator = HeadPoseEstimator()
        
        self._config = config_manager.face
        self._yaw_threshold = self._config.get('yaw_threshold', 40)
        self._pitch_offset = self._config.get('pitch_offset', 5.0)
        self._pitch_min = self._config.get('pitch_threshold_min', 0.0)
        self._pitch_max = self._config.get('pitch_threshold_max', 30.0)
    
    def process(self, frame, results) -> dict:
        data = {
            'detected': False,
            'valid': False,
            'landmarks': None,
            'ear': 0.35,
            'mar': 0.15,
            'pitch': 0.0,
            'yaw': 0.0,
            'roll': 0.0,
            'emotion': 'No Face'
        }
        
        try:
            if results is None or not hasattr(results, 'multi_face_landmarks') or not results.multi_face_landmarks:
                return data
            
            landmarks = results.multi_face_landmarks[0]
            lm_points = landmarks.landmark
            
            x_coords = [p.x for p in lm_points]
            y_coords = [p.y for p in lm_points]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            is_detected = not (min_x < 0.01 or max_x > 0.99 or min_y < 0.01 or max_y > 0.99)
            
            raw_pitch, yaw, roll = self.pose_estimator.get_pose(frame, landmarks)
            pitch = raw_pitch + self._pitch_offset
            
            is_face_valid = is_detected and abs(yaw) <= self._yaw_threshold
            
            data['detected'] = is_detected
            data['valid'] = is_face_valid
            data['landmarks'] = landmarks
            data['pitch'] = pitch
            data['yaw'] = yaw
            data['roll'] = roll
            
            if is_face_valid:
                ear = calculate_ear(lm_points)
                mar = calculate_mar(lm_points)
                data['ear'] = ear
                data['mar'] = mar
            
            logger.debug(f"Face processed: detected={is_detected}, valid={is_face_valid}")
            
        except Exception as e:
            logger.error(f"Error in FaceProcessor: {e}")
        
        return data
    
    def reload_config(self):
        self._config = config_manager.face
        self._yaw_threshold = self._config.get('yaw_threshold', 40)
        self._pitch_offset = self._config.get('pitch_offset', 5.0)
        self._pitch_min = self._config.get('pitch_threshold_min', 0.0)
        self._pitch_max = self._config.get('pitch_threshold_max', 30.0)


class EmotionProcessor:
    def __init__(self):
        self.emotion_ai = None
        self._current_emotion = 'Neutral'
        self._frame_counter = 0
        self._init_model()
    
    def _init_model(self):
        try:
            from src.emotion_detector import EmotionDetector
            model_path = "models/emotion_model.hdf5"
            self.emotion_ai = EmotionDetector(model_path)
            logger.info("Emotion detector initialized")
        except Exception as e:
            logger.warning(f"Could not initialize emotion detector: {e}")
            self.emotion_ai = None
    
    def process(self, frame, landmarks) -> str:
        if self.emotion_ai is None:
            return self._current_emotion
        
        try:
            self._frame_counter += 1
            
            if self._frame_counter % 10 == 0:
                emo, _ = self.emotion_ai.predict_emotion(frame, landmarks)
                if emo and emo not in ["Error", "No Face"]:
                    self._current_emotion = emo
            
        except Exception as e:
            logger.error(f"Error in EmotionProcessor: {e}")
        
        return self._current_emotion
    
    def reset(self):
        self._current_emotion = 'Neutral'
        self._frame_counter = 0

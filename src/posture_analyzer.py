import numpy as np
from collections import deque
import time
import math


class PostureAnalyzer:
    """
    Анализатор осанки на основе оценки положения головы и плеч.
    Использует face mesh landmarks для оценки осанки ( когда полный Pose недоступен).
    
    Метрики:
    - Forward head: наклон головы вперёд (по положению носа относительно центра)
    - Head tilt: наклон головы вбок (по symmetry лица)
    - Face confidence: насколько лицо находится в центре кадра
    """
    
    def __init__(self, window_size_seconds=5):
        self.window_size = window_size_seconds
        
        self.head_tilt_history = deque(maxlen=50)
        self.head_forward_history = deque(maxlen=50)
        self.face_position_history = deque(maxlen=50)
        self.timestamps = deque(maxlen=50)
        
        self.last_posture_event_time = 0
        
        self._tilt_threshold = 12
        self._forward_threshold = 0.08
        self._face_center_threshold = 0.15
        
    def update_from_face_mesh(self, face_landmarks, frame_width=640, frame_height=480, current_time=None):
        """
        Обновить данные об осанке на основе landmarks лица.
        
        Args:
            face_landmarks: MediaPipe Face Mesh landmarks
            frame_width, frame_height: размеры кадра
            current_time: текущее время
        
        Returns:
            dict с метриками осанки
        """
        if face_landmarks is None:
            return self._get_default_metrics()
        
        try:
            nose = face_landmarks.landmark[1]
            left_ear = face_landmarks.landmark[234]
            right_ear = face_landmarks.landmark[454]
            forehead = face_landmarks.landmark[10]
            chin = face_landmarks.landmark[152]
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            
            face_center_x = (left_ear.x + right_ear.x) / 2
            face_center_y = (nose.y + chin.y + forehead.y) / 3
            
            head_tilt = self._calculate_head_tilt(left_ear, right_ear, left_eye, right_eye)
            head_forward = self._calculate_head_forward(nose, forehead, chin)
            face_position_score = self._calculate_face_position(face_center_x, face_center_y)
            
            self.head_tilt_history.append(head_tilt)
            self.head_forward_history.append(head_forward)
            self.face_position_history.append(face_position_score)
            if current_time:
                self.timestamps.append(current_time)
            
            posture_score = self._calculate_posture_score(head_tilt, head_forward, face_position_score)
            posture_level, is_bad = self._analyze_posture_state(head_tilt, head_forward, face_position_score)
            
            return {
                'posture_score': posture_score,
                'posture_level': posture_level,
                'head_tilt': head_tilt,
                'head_forward': head_forward,
                'face_position': face_position_score,
                'is_bad': is_bad,
            }
            
        except (AttributeError, IndexError) as e:
            return self._get_default_metrics()
    
    def _calculate_head_tilt(self, left_ear, right_ear, left_eye, right_eye):
        """
        Рассчитать наклон головы вбок.
        Использует асимметрию глаз и ушей.
        """
        eye_center_x = (left_eye.x + right_eye.x) / 2
        ear_center_x = (left_ear.x + right_ear.x) / 2
        
        dx = (right_ear.x - left_ear.x)
        
        if dx < 0.03:
            return 0
        
        tilt = (eye_center_x - ear_center_x) / dx * 100
        
        return max(-30, min(30, tilt * 15))
    
    def _calculate_head_forward(self, nose, forehead, chin):
        """
        Рассчитать выдвижение головы вперёд.
        Использует соотношение высоты лица к его ширине.
        """
        face_height = chin.y - forehead.y
        nose_x_deviation = abs(nose.x - 0.5)
        
        forward_score = nose_x_deviation + (0.4 - face_height) * 0.3
        
        return max(0, forward_score)
    
    def _calculate_face_position(self, face_center_x, face_center_y):
        """
        Насколько лицо смещено от центра кадра.
        """
        dx = abs(face_center_x - 0.5)
        dy = abs(face_center_y - 0.4)
        
        return math.sqrt(dx**2 + dy**2)
    
    def _calculate_posture_score(self, head_tilt, head_forward, face_position):
        """
        Рассчитать общий скор осанки (0-100, где 100 = очень плохая осанка).
        """
        score = 0
        
        abs_tilt = abs(head_tilt)
        if abs_tilt > 25:
            score += 50
        elif abs_tilt > 18:
            score += 35
        elif abs_tilt > 12:
            score += 20
        elif abs_tilt > 8:
            score += 10
        
        if head_forward > 0.12:
            score += 50
        elif head_forward > 0.08:
            score += 35
        elif head_forward > 0.05:
            score += 20
        elif head_forward > 0.03:
            score += 10
        
        if face_position > 0.25:
            score += 40
        elif face_position > 0.18:
            score += 25
        elif face_position > 0.12:
            score += 15
        elif face_position > 0.08:
            score += 5
        
        return min(100, score)
    
    def _analyze_posture_state(self, head_tilt, head_forward, face_position):
        """
        Проанализировать состояние осанки.
        """
        score = self._calculate_posture_score(head_tilt, head_forward, face_position)
        
        if score >= 60:
            level = 'bad'
            is_bad = True
        elif score >= 30:
            level = 'fair'
            is_bad = True
        else:
            level = 'good'
            is_bad = False
        
        return level, is_bad
    
    def _get_default_metrics(self):
        """Вернуть метрики по умолчанию."""
        return {
            'posture_score': 0,
            'posture_level': 'unknown',
            'head_tilt': 0,
            'head_forward': 0,
            'face_position': 0,
            'is_bad': False,
        }
    
    def update(self, pose_landmarks=None, ear_x=None, ear_y=None, frame_width=640, frame_height=480, current_time=None):
        """
        Универсальный метод - поддерживает как Pose, так и Face Mesh.
        """
        return self.update_from_face_mesh(pose_landmarks, frame_width, frame_height, current_time)
    
    def get_posture_event(self, current_time):
        """
        Проверить и вернуть событие осанки, если нужно.
        """
        cooldown = 5.0
        
        if current_time - self.last_posture_event_time < cooldown:
            return None
        
        if len(self.head_tilt_history) < 10:
            return None
        
        recent_tilt = list(self.head_tilt_history)[-10:]
        recent_forward = list(self.head_forward_history)[-10:]
        
        avg_tilt = np.mean([abs(t) for t in recent_tilt])
        avg_forward = np.mean(recent_forward)
        
        if avg_tilt > 15 or avg_forward > 0.10:
            self.last_posture_event_time = current_time
            return "Плохая осанка"
        
        return None
    
    def get_status_text(self):
        """Получить текстовое описание текущего состояния осанки."""
        if len(self.head_tilt_history) < 5:
            return "Анализ..."
        
        recent_tilt = list(self.head_tilt_history)[-5:]
        recent_forward = list(self.head_forward_history)[-5:]
        
        avg_tilt = np.mean([abs(t) for t in recent_tilt])
        avg_forward = np.mean(recent_forward)
        
        issues = []
        
        if avg_tilt > 10:
            issues.append("наклон")
        if avg_forward > 0.07:
            issues.append("вперёд")
        
        if not issues:
            return "Хорошая"
        else:
            return "Проблемы: " + ", ".join(issues)
    
    def reset(self):
        """Сбросить все данные."""
        self.head_tilt_history.clear()
        self.head_forward_history.clear()
        self.face_position_history.clear()
        self.timestamps.clear()
        self.last_posture_event_time = 0

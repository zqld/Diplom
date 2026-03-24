import numpy as np
from collections import deque
import time


class FatigueAnalyzer:
    """
    Анализатор усталости на основе комбинированного скора и анализа трендов.
    Использует множественные признаки с лица и отслеживает динамику во времени.
    """
    
    def __init__(self, window_size_seconds=30):
        self.window_size = window_size_seconds
        
        self.ear_history = deque(maxlen=100)
        self.mar_history = deque(maxlen=100)
        self.pitch_history = deque(maxlen=100)
        self.emotion_history = deque(maxlen=50)
        self.timestamps = deque(maxlen=100)
        
        self.blink_count = 0
        self.blink_timestamps = deque(maxlen=30)
        self.last_ear = 0.35
        self.was_eyes_closed = False
        
        self.fatigue_events = []
        self.last_fatigue_event_time = 0
        
        self._ear_threshold_low = 0.25
        self._ear_threshold_high = 0.30
        self._mar_threshold = 0.5
        
    def update(self, ear, mar, pitch, emotion, current_time):
        """Обновить данные и получить текущий уровень усталости."""
        
        self.ear_history.append(ear)
        self.mar_history.append(mar)
        self.pitch_history.append(pitch)
        self.emotion_history.append(emotion)
        self.timestamps.append(current_time)
        
        self._detect_blink(ear, current_time)
        
        fatigue_score = self._calculate_fatigue_score()
        fatigue_level, trend = self._analyze_fatigue_state(fatigue_score)
        
        return {
            'fatigue_score': fatigue_score,
            'fatigue_level': fatigue_level,
            'trend': trend,
            'ear_trend': self._get_ear_trend(),
            'blink_rate': self._get_blink_rate(),
            'avg_ear': self._get_avg_ear(),
            'avg_mar': self._get_avg_mar(),
        }
    
    def _detect_blink(self, ear, current_time):
        """Определение моргания по резкому изменению EAR."""
        if self.last_ear > 0.28 and ear < 0.22:
            if not self.was_eyes_closed:
                self.blink_count += 1
                self.blink_timestamps.append(current_time)
                self.was_eyes_closed = True
        elif ear > 0.25:
            self.was_eyes_closed = False
        self.last_ear = ear
    
    def _calculate_fatigue_score(self):
        """
        Рассчитать общий скор усталости (0-100, где 100 = очень устал).
        Основан на комбинации нескольких факторов.
        """
        score = 0
        weights = {
            'ear': 0.35,
            'mar': 0.15,
            'blink': 0.20,
            'emotion': 0.15,
            'trend': 0.15
        }
        
        ear_score = self._get_ear_score()
        mar_score = self._get_mar_score()
        blink_score = self._get_blink_score()
        emotion_score = self._get_emotion_score()
        trend_score = self._get_trend_score()
        
        score = (
            weights['ear'] * ear_score +
            weights['mar'] * mar_score +
            weights['blink'] * blink_score +
            weights['emotion'] * emotion_score +
            weights['trend'] * trend_score
        )
        
        return min(100, max(0, score))
    
    def _get_ear_score(self):
        """Скор на основе EAR (открытость глаз)."""
        if len(self.ear_history) < 5:
            return 0
        
        recent_ear = list(self.ear_history)[-10:]
        avg_ear = np.mean(recent_ear)
        
        if avg_ear < 0.18:
            return 100
        elif avg_ear < 0.22:
            return 80
        elif avg_ear < 0.25:
            return 50
        elif avg_ear < 0.30:
            return 20
        return 0
    
    def _get_mar_score(self):
        """Скор на основе MAR (открытость рта - зевание)."""
        if len(self.mar_history) < 5:
            return 0
        
        recent_mar = list(self.mar_history)[-10:]
        max_mar = max(recent_mar)
        avg_mar = np.mean(recent_mar)
        
        if max_mar > 0.65:
            return 100
        elif max_mar > 0.55:
            return 70
        elif max_mar > 0.45:
            return 40
        elif avg_mar > 0.35:
            return 20
        return 0
    
    def _get_blink_score(self):
        """Скор на основе частоты морганий."""
        blink_rate = self._get_blink_rate()
        
        if blink_rate > 40:
            return 80
        elif blink_rate > 30:
            return 60
        elif blink_rate > 20:
            return 30
        return 0
    
    def _get_emotion_score(self):
        """Скор на основе эмоционального состояния."""
        if len(self.emotion_history) < 3:
            return 0
        
        emotions = list(self.emotion_history)[-10:]
        
        fatigue_emotions = ['Усталость', 'Tired', 'Грусть', 'Sad', 'Сонливость', 'Drowsy']
        neutral_emotions = ['Нейтрально', 'Neutral', 'Спокойствие', 'Calm']
        happy_emotions = ['Счастье', 'Happy', 'Радость', 'Joy']
        
        fatigue_count = sum(1 for e in emotions if e in fatigue_emotions)
        happy_count = sum(1 for e in emotions if e in happy_emotions)
        
        if fatigue_count >= 5:
            return 70
        elif fatigue_count >= 3:
            return 40
        elif happy_count >= 5:
            return 0
        return 10
    
    def _get_trend_score(self):
        """Скор на основе тренда (динамики изменения EAR)."""
        if len(self.ear_history) < 15:
            return 0
        
        recent = list(self.ear_history)[-15:]
        first_half = np.mean(recent[:7])
        second_half = np.mean(recent[7:])
        
        change = first_half - second_half
        
        if change > 0.08:
            return 100
        elif change > 0.05:
            return 70
        elif change > 0.03:
            return 40
        elif change > 0.01:
            return 20
        return 0
    
    def _get_ear_trend(self):
        """Получить направление тренда EAR."""
        if len(self.ear_history) < 15:
            return 'stable'
        
        recent = list(self.ear_history)[-15:]
        first_half = np.mean(recent[:7])
        second_half = np.mean(recent[7:])
        
        change = second_half - first_half
        
        if change < -0.03:
            return 'decreasing'
        elif change > 0.03:
            return 'increasing'
        return 'stable'
    
    def _analyze_fatigue_state(self, fatigue_score):
        """
        Проанализировать текущее состояние усталости.
        Возвращает уровень (normal, mild, moderate, severe) и тренд.
        """
        trend = self._get_ear_trend()
        
        if fatigue_score >= 70:
            level = 'severe'
        elif fatigue_score >= 50:
            level = 'moderate'
        elif fatigue_score >= 30:
            level = 'mild'
        else:
            level = 'normal'
        
        if trend == 'decreasing' and fatigue_score > 30:
            level = self._worsen_level(level)
        
        return level, trend
    
    def _worsen_level(self, level):
        """Ухудшить уровень при негативном тренде."""
        levels = ['normal', 'mild', 'moderate', 'severe']
        try:
            idx = levels.index(level)
            return levels[min(idx + 1, 3)]
        except ValueError:
            return level
    
    def _get_blink_rate(self):
        """Получить частоту морганий в минуту."""
        if len(self.blink_timestamps) < 2:
            return 0
        
        recent_blinks = [t for t in self.blink_timestamps if time.time() - t < 60]
        
        if len(recent_blinks) < 2:
            return 0
        
        time_span = recent_blinks[-1] - recent_blinks[0]
        if time_span < 1:
            time_span = 1
            
        return int(len(recent_blinks) / time_span * 60)
    
    def _get_avg_ear(self):
        """Среднее значение EAR за последнее время."""
        if not self.ear_history:
            return 0.35
        return np.mean(list(self.ear_history)[-20:])
    
    def _get_avg_mar(self):
        """Среднее значение MAR за последнее время."""
        if not self.mar_history:
            return 0.0
        return np.mean(list(self.mar_history)[-20:])
    
    def get_fatigue_event(self, current_time):
        """
        Проверить и вернуть событие усталости, если нужно.
        С учётом cooldown периода.
        """
        cooldown = 5.0
        
        if current_time - self.last_fatigue_event_time < cooldown:
            return None
        
        level, trend = self._analyze_fatigue_state(self._calculate_fatigue_score())
        
        if level == 'severe':
            self.last_fatigue_event_time = current_time
            return "Сильная усталость"
        elif level == 'moderate':
            self.last_fatigue_event_time = current_time
            return "Умеренная усталость"
        elif level == 'mild' and trend == 'decreasing':
            self.last_fatigue_event_time = current_time
            return "Лёгкая усталость (снижение)"
        
        return None
    
    def get_status_text(self):
        """Получить текстовое описание текущего состояния."""
        if len(self.ear_history) < 5:
            return "Анализ..."
        
        score = self._calculate_fatigue_score()
        level, trend = self._analyze_fatigue_state(score)
        
        status_map = {
            'normal': 'Бодрое',
            'mild': 'Лёгкая усталость',
            'moderate': 'Усталость',
            'severe': 'Сильная усталость'
        }
        
        status = status_map.get(level, 'Бодрое')
        
        if trend == 'decreasing':
            status += ' ↓'
        elif trend == 'increasing':
            status += ' ↑'
            
        return status
    
    def reset(self):
        """Сбросить все данные."""
        self.ear_history.clear()
        self.mar_history.clear()
        self.pitch_history.clear()
        self.emotion_history.clear()
        self.timestamps.clear()
        self.blink_count = 0
        self.blink_timestamps.clear()
        self.fatigue_events.clear()
        self.last_fatigue_event_time = 0

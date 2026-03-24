"""
Auto-labeling System for Fatigue Data.
"""

import time
from collections import deque


class FatigueAutoLabeler:
    """
    Automatically labels fatigue states based on behavioral patterns.
    """
    
    def __init__(self):
        self.closed_eye_duration = 0
        self.closed_eye_start = None
        self.last_state = 'awake'
        self.microsleep_threshold = 0.5
        self.sleeping_threshold = 3.0
        self.state_history = deque(maxlen=100)
        
    def update(self, ear: float, mar: float, blink_rate: float,
               head_droop: float = 0, current_time: float = None) -> dict:
        if current_time is None:
            current_time = time.time()
        
        if ear < 0.18:
            if self.closed_eye_start is None:
                self.closed_eye_start = current_time
            self.closed_eye_duration = current_time - self.closed_eye_start
        else:
            self.closed_eye_start = None
            self.closed_eye_duration = 0
        
        if self.closed_eye_duration >= self.sleeping_threshold:
            label = 'sleeping'
            confidence = min(1.0, self.closed_eye_duration / 5.0)
        elif self.closed_eye_duration >= self.microsleep_threshold:
            label = 'drowsy'
            confidence = 0.8
        elif ear < 0.22:
            label = 'drowsy'
            confidence = 0.6
        else:
            label = 'awake'
            confidence = 0.9
        
        if mar > 0.5:
            label = 'drowsy'
            confidence = max(confidence, 0.7)
        
        if head_droop > 25:
            if label == 'awake':
                label = 'drowsy'
                confidence = 0.5
        
        self.last_state = label
        self.state_history.append({'label': label, 'confidence': confidence})
        
        return {
            'label': label,
            'confidence': confidence,
            'closed_eye_duration': self.closed_eye_duration,
            'is_microsleep': self.closed_eye_duration >= self.microsleep_threshold,
            'is_sleeping': self.closed_eye_duration >= self.sleeping_threshold
        }
    
    def get_consensus_label(self, window_size: int = 30) -> dict:
        if len(self.state_history) < 5:
            return {'label': 'awake', 'confidence': 0.5}
        
        recent = list(self.state_history)[-window_size:]
        label_counts = {'awake': 0, 'drowsy': 0, 'sleeping': 0}
        total_conf = {'awake': 0, 'drowsy': 0, 'sleeping': 0}
        
        for item in recent:
            label_counts[item['label']] += 1
            total_conf[item['label']] += item['confidence']
        
        best_label = max(label_counts, key=lambda x: label_counts[x] * total_conf[x])
        confidence = total_conf[best_label] / max(1, label_counts[best_label])
        consistency = label_counts[best_label] / len(recent)
        confidence *= consistency
        
        return {
            'label': best_label,
            'confidence': min(1.0, confidence),
            'consistency': consistency
        }
    
    def reset(self):
        self.closed_eye_duration = 0
        self.closed_eye_start = None
        self.last_state = 'awake'
        self.state_history.clear()


class PostureAutoLabeler:
    """
    Automatically labels posture states based on body position.
    """
    
    def __init__(self):
        self.bad_posture_duration = 0
        self.bad_posture_start = None
        self.last_state = 'good'
        self.bad_threshold = 5.0
        self.state_history = deque(maxlen=100)
        
    def update(self, posture_status: str, posture_score: float,
               shoulder_angle: float = 0, head_tilt: float = 0,
               current_time: float = None) -> dict:
        if current_time is None:
            current_time = time.time()
        
        is_bad = posture_status in ['bad', 'poor'] or posture_score < 50
        is_bad = is_bad or abs(shoulder_angle) > 15 or abs(head_tilt) > 20
        
        if is_bad:
            if self.bad_posture_start is None:
                self.bad_posture_start = current_time
            self.bad_posture_duration = current_time - self.bad_posture_start
            label = 'bad'
        else:
            self.bad_posture_start = None
            self.bad_posture_duration = 0
            label = 'good'
        
        if self.bad_posture_duration >= self.bad_threshold:
            label = 'bad'
            confidence = min(1.0, self.bad_posture_duration / 10.0)
        elif is_bad:
            confidence = 0.7
        else:
            confidence = 0.9
        
        self.last_state = label
        self.state_history.append({'label': label, 'confidence': confidence})
        
        return {
            'label': label,
            'confidence': confidence,
            'bad_posture_duration': self.bad_posture_duration
        }
    
    def get_consensus_label(self, window_size: int = 30) -> dict:
        if len(self.state_history) < 5:
            return {'label': 'good', 'confidence': 0.5}
        
        recent = list(self.state_history)[-window_size:]
        label_counts = {'good': 0, 'bad': 0}
        
        for item in recent:
            label_counts[item['label']] += 1
        
        best_label = max(label_counts, key=label_counts.get)
        consistency = label_counts[best_label] / len(recent)
        
        return {
            'label': best_label,
            'confidence': consistency,
            'label_counts': label_counts
        }
    
    def reset(self):
        self.bad_posture_duration = 0
        self.bad_posture_start = None
        self.last_state = 'good'
        self.state_history.clear()

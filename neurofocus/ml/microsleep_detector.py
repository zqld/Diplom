"""
Microsleep Detector - Detects dangerous microsleep episodes.
"""

import time
from collections import deque


class MicrosleepDetector:
    """
    Detects microsleep episodes - brief periods of reduced alertness
    that can be dangerous during work.
    
    Microsleep is defined as eye closure for 500ms or longer.
    """
    
    def __init__(self):
        self.closed_start_time = None
        self.events = deque(maxlen=100)  # Keep last 100 events
        
        # Thresholds
        self.ear_closed_threshold = 0.18  # Eyes considered closed
        self.microsleep_duration = 0.5   # 500ms = minimum microsleep
        
        # Statistics
        self.total_microsleeps = 0
        
    def update(self, ear: float, current_time: float = None) -> dict:
        """
        Check for microsleep episode.
        
        Args:
            ear: Current Eye Aspect Ratio
            current_time: Current timestamp
            
        Returns:
            dict with detection info
        """
        if current_time is None:
            current_time = time.time()
            
        result = {
            'microsleep_detected': False,
            'is_eyes_closed': False,
            'closed_duration': 0.0,
            'microsleep_count_last_minute': 0
        }
        
        # Check if eyes are closed
        if ear < self.ear_closed_threshold:
            result['is_eyes_closed'] = True
            
            if self.closed_start_time is None:
                # Start tracking closed eyes
                self.closed_start_time = current_time
            else:
                # Eyes have been closed
                closed_duration = current_time - self.closed_start_time
                result['closed_duration'] = closed_duration
                
                # Check if this is a microsleep (duration > threshold)
                if closed_duration >= self.microsleep_duration:
                    # Check if this is a NEW microsleep (not continuing)
                    if not hasattr(self, '_last_microsleep_time') or \
                       current_time - self._last_microsleep_time > 1.0:
                        
                        # Record new microsleep
                        self.events.append({
                            'start': self.closed_start_time,
                            'duration': closed_duration,
                            'timestamp': current_time
                        })
                        self.total_microsleeps += 1
                        self._last_microsleep_time = current_time
                        
                        result['microsleep_detected'] = True
        else:
            # Eyes are open, reset tracking
            self.closed_start_time = None
            
        # Count microsleeps in last minute
        recent_count = self.get_microsleeps_per_minute()
        result['microsleep_count_last_minute'] = recent_count
        
        return result
    
    def get_danger_level(self) -> str:
        """
        Get danger level based on microsleep frequency.
        
        Returns:
            'danger' (2+ per minute), 'warning' (1 per minute), 'normal'
        """
        recent_count = self.get_microsleeps_per_minute()
        
        if recent_count >= 2:
            return 'danger'
        elif recent_count >= 1:
            return 'warning'
        return 'normal'
    
    def get_microsleeps_per_minute(self) -> int:
        """Get number of microsleeps in the last 60 seconds."""
        if not self.events:
            return 0
            
        now = time.time()
        recent = [e for e in self.events if now - e['timestamp'] < 60]
        return len(recent)
    
    def get_total_duration(self) -> float:
        """Get total time spent in microsleep (seconds)."""
        return sum(e['duration'] for e in self.events)
    
    def get_average_duration(self) -> float:
        """Get average microsleep duration."""
        if not self.events:
            return 0.0
        return sum(e['duration'] for e in self.events) / len(self.events)
    
    def get_statistics(self) -> dict:
        """Get comprehensive microsleep statistics."""
        return {
            'total_microsleeps': self.total_microsleeps,
            'microsleeps_per_minute': self.get_microsleeps_per_minute(),
            'danger_level': self.get_danger_level(),
            'total_duration': self.get_total_duration(),
            'avg_duration': self.get_average_duration(),
            'is_eyes_closed': self.closed_start_time is not None,
            'closed_duration': time.time() - self.closed_start_time if self.closed_start_time else 0
        }
    
    def reset(self):
        """Reset detector state."""
        self.closed_start_time = None
        self.events.clear()
        self.total_microsleeps = 0
        if hasattr(self, '_last_microsleep_time'):
            delattr(self, '_last_microsleep_time')


# Alias
MicrosleepDetector = MicrosleepDetector
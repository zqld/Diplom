"""
Blink Tracker - Real-time blink detection from EAR history.
"""

import time
from collections import deque
import numpy as np


class BlinkTracker:
    """
    Tracks blinks based on EAR (Eye Aspect Ratio) changes.
    Detects when eyes close and open, counting as one blink.
    """
    
    def __init__(self, history_size: int = 300, fps: int = 30):
        """
        Args:
            history_size: Maximum number of frames to keep in history
            fps: Frames per second for time calculations
        """
        self.history_size = history_size
        self.fps = fps
        
        # EAR history with timestamps: [(ear, timestamp), ...]
        self.ear_history = deque(maxlen=history_size)
        
        # Blink tracking
        self.blink_count = 0
        self.in_blink = False
        self.blink_start_time = None
        self.blink_start_ear = 1.0
        
        # Blink events for analysis
        self.blink_events = deque(maxlen=50)
        
        # Thresholds
        self.ear_closed_threshold = 0.22
        self.ear_open_threshold = 0.28
        
    def update(self, ear: float, current_time: float = None) -> dict:
        """
        Update with new EAR value.
        
        Args:
            ear: Current Eye Aspect Ratio value
            current_time: Current timestamp (defaults to time.time())
            
        Returns:
            dict with blink detection info
        """
        if current_time is None:
            current_time = time.time()
            
        # Add to history
        self.ear_history.append((ear, current_time))
        
        result = {
            'new_blink': False,
            'in_blink': self.in_blink,
            'blink_count': self.blink_count
        }
        
        # Detect blink start (EAR drops below threshold)
        if ear < self.ear_closed_threshold and not self.in_blink:
            self.in_blink = True
            self.blink_start_time = current_time
            self.blink_start_ear = ear
            
        # Detect blink end (EAR returns above threshold)
        elif ear >= self.ear_open_threshold and self.in_blink:
            blink_duration = current_time - self.blink_start_time
            
            # Only count if it was a real blink (not just noise)
            if blink_duration > 0.05:  # Minimum 50ms
                self.blink_count += 1
                
                # Record blink event
                self.blink_events.append({
                    'start': self.blink_start_time,
                    'duration': blink_duration,
                    'min_ear': self.blink_start_ear
                })
                
                result['new_blink'] = True
                
            self.in_blink = False
            self.blink_start_time = None
            
        return result
    
    def get_blink_rate_per_minute(self) -> int:
        """
        Calculate blinks per minute based on recent history.
        
        Returns:
            Integer blinks per minute
        """
        if len(self.ear_history) < 30:  # Need at least 1 second
            return 0
            
        # Get time span of history
        oldest = self.ear_history[0][1]
        newest = self.ear_history[-1][1]
        time_span_minutes = (newest - oldest) / 60.0
        
        if time_span_minutes < 0.1:  # Less than 6 seconds
            return 0
            
        # Count blinks in recent history (last N seconds)
        recent_blinks = [e for e in self.blink_events 
                        if newest - e['start'] < time_span_minutes * 60]
        
        return int(len(recent_blinks) / time_span_minutes)
    
    def get_average_blink_duration(self) -> float:
        """Get average blink duration in seconds."""
        if not self.blink_events:
            return 0.0
            
        recent = list(self.blink_events)[-10:]  # Last 10 blinks
        return sum(e['duration'] for e in recent) / len(recent)
    
    def get_blink_statistics(self) -> dict:
        """Get comprehensive blink statistics."""
        # Recent time window (last 60 seconds)
        now = time.time()
        recent_events = [e for e in self.blink_events if now - e['start'] < 60]
        
        return {
            'blink_count_total': self.blink_count,
            'blink_rate_per_minute': self.get_blink_rate_per_minute(),
            'avg_blink_duration': self.get_average_blink_duration(),
            'blinks_last_60s': len(recent_events),
            'in_blink': self.in_blink,
            'current_ear': self.ear_history[-1][0] if self.ear_history else 0.3
        }
    
    def reset(self):
        """Reset all counters and history."""
        self.ear_history.clear()
        self.blink_count = 0
        self.in_blink = False
        self.blink_start_time = None
        self.blink_events.clear()
    
    def get_ear_rolling_stats(self, window_seconds: int = 5) -> dict:
        """
        Get rolling statistics for EAR over a time window.
        
        Args:
            window_seconds: Time window in seconds
            
        Returns:
            dict with mean, std, min, max of EAR
        """
        if not self.ear_history:
            return {'mean': 0.3, 'std': 0, 'min': 0.3, 'max': 0.3}
            
        now = self.ear_history[-1][1]
        cutoff = now - window_seconds
        
        recent_ears = [e[0] for e in self.ear_history if e[1] > cutoff]
        
        if len(recent_ears) < 5:
            return {'mean': 0.3, 'std': 0, 'min': 0.3, 'max': 0.3}
            
        return {
            'mean': float(np.mean(recent_ears)),
            'std': float(np.std(recent_ears)),
            'min': float(np.min(recent_ears)),
            'max': float(np.max(recent_ears)),
            'count': len(recent_ears)
        }


# Alias for backwards compatibility
BlinkDetector = BlinkTracker
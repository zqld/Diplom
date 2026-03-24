"""
Temporal Feature Extractor - Extracts features from sliding window of frames.
"""

import time
import numpy as np
from collections import deque


class TemporalFeatureExtractor:
    """
    Extracts temporal features from a sliding window of frames.
    Used to feed LSTM model with meaningful features.
    """
    
    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Number of frames in sliding window (30 = 1 sec at 30fps)
        """
        self.window_size = window_size
        
        # Data windows
        self.ear_window = deque(maxlen=window_size)
        self.mar_window = deque(maxlen=window_size)
        self.pitch_window = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        # Extended history for trend calculation
        self.ear_history = deque(maxlen=300)  # 10 seconds
        self.mar_history = deque(maxlen=300)
        
    def update(self, ear: float, mar: float, pitch: float = 0.0, 
               head_droop: float = 0.0, head_tilt: float = 0.0,
               current_time: float = None) -> dict:
        """
        Add new frame data to the temporal buffer.
        
        Args:
            ear: Eye Aspect Ratio
            mar: Mouth Aspect Ratio
            pitch: Head pitch angle (degrees)
            head_droop: Head forward droop (degrees)
            head_tilt: Head side tilt (degrees)
            current_time: Current timestamp
            
        Returns:
            dict with current feature values
        """
        if current_time is None:
            current_time = time.time()
            
        # Add to windows
        self.ear_window.append(ear)
        self.mar_window.append(mar)
        self.pitch_window.append(pitch)
        self.timestamps.append(current_time)
        
        # Add to extended history
        self.ear_history.append(ear)
        self.mar_history.append(mar)
        
        # Return current features
        return self.get_current_features()
    
    def get_current_features(self) -> dict:
        """
        Get all temporal features for current window.
        
        Returns:
            dict with extracted features
        """
        features = self._calculate_window_features()
        
        # Add extended history features
        features.update(self._calculate_extended_features())
        
        return features
    
    def _calculate_window_features(self) -> dict:
        """Calculate features from current window."""
        if len(self.ear_window) < 5:
            return self._default_features()
            
        ear_arr = np.array(list(self.ear_window))
        mar_arr = np.array(list(self.mar_window))
        
        # Basic statistics
        features = {
            # EAR features
            'ear_mean': float(np.mean(ear_arr)),
            'ear_std': float(np.std(ear_arr)),
            'ear_min': float(np.min(ear_arr)),
            'ear_max': float(np.max(ear_arr)),
            'ear_median': float(np.median(ear_arr)),
            
            # MAR features  
            'mar_mean': float(np.mean(mar_arr)),
            'mar_std': float(np.std(mar_arr)),
            'mar_max': float(np.max(mar_arr)),
            
            # Trend (linear regression slope)
            'ear_trend': self._calculate_trend(ear_arr),
            'mar_trend': self._calculate_trend(mar_arr),
            
            # Eye closure detection
            'eyes_closed_ratio': float(np.sum(ear_arr < 0.22) / len(ear_arr)),
            'eyes_very_closed_ratio': float(np.sum(ear_arr < 0.18) / len(ear_arr)),
            
            # Yawn detection (sustained high MAR)
            'yawning': bool(np.sum(mar_arr > 0.5) > 3),
            'yawn_intensity': float(np.max(mar_arr)),
        }
        
        return features
    
    def _calculate_extended_features(self) -> dict:
        """Calculate features from extended history (last 10 seconds)."""
        if len(self.ear_history) < 30:
            return {}
            
        ear_arr = np.array(list(self.ear_history))
        
        # Variance over time
        # Split into chunks and get variance of means
        chunk_size = 30  # 1 second chunks
        chunk_means = []
        for i in range(0, len(ear_arr) - chunk_size, chunk_size):
            chunk_means.append(np.mean(ear_arr[i:i+chunk_size]))
        
        features = {
            'ear_variance_long': float(np.var(ear_arr)),
            'ear_stability': float(1.0 / (np.std(chunk_means) + 0.001)) if chunk_means else 1.0,
        }
        
        # Blink rate (estimated from eye closures)
        # Count transitions from open to closed
        transitions = 0
        prev_open = True
        for ear in ear_arr:
            if ear < 0.22:  # Closed
                if prev_open:
                    transitions += 1
                    prev_open = False
            else:
                prev_open = True
                
        time_span = len(self.ear_history) / 30.0  # seconds
        features['estimated_blink_rate'] = int(transitions / time_span * 60) if time_span > 0 else 0
        
        return features
    
    def _calculate_trend(self, arr: np.ndarray) -> float:
        """
        Calculate linear trend (slope) of array.
        Positive = increasing, Negative = decreasing.
        """
        if len(arr) < 2:
            return 0.0
            
        x = np.arange(len(arr))
        try:
            slope, _ = np.polyfit(x, arr, 1)
            return float(slope)
        except:
            return 0.0
    
    def _default_features(self) -> dict:
        """Return default features when insufficient data."""
        return {
            'ear_mean': 0.3,
            'ear_std': 0.0,
            'ear_min': 0.3,
            'ear_max': 0.3,
            'ear_median': 0.3,
            'mar_mean': 0.15,
            'mar_std': 0.0,
            'mar_max': 0.15,
            'ear_trend': 0.0,
            'mar_trend': 0.0,
            'eyes_closed_ratio': 0.0,
            'eyes_very_closed_ratio': 0.0,
            'yawning': False,
            'yawn_intensity': 0.0,
            'ear_variance_long': 0.0,
            'ear_stability': 1.0,
            'estimated_blink_rate': 0,
        }
    
    def get_lstm_features(self) -> np.ndarray:
        """
        Get features formatted for LSTM input.
        
        Returns:
            numpy array of shape (window_size, num_features)
        """
        if len(self.ear_window) < self.window_size:
            # Pad with default features
            default = self._default_features()
            features = [list(default.values())] * self.window_size
            return np.array(features)
        
        # For each frame in window, create feature vector
        features = []
        for i in range(len(self.ear_window)):
            # Get history up to this point
            ear_hist = list(self.ear_window)[:i+1]
            mar_hist = list(self.mar_window)[:i+1]
            
            if len(ear_hist) < 5:
                frame_features = list(self._default_features().values())
            else:
                frame_features = [
                    np.mean(ear_hist),
                    np.std(ear_hist),
                    np.min(ear_hist),
                    np.max(ear_hist),
                    np.mean(mar_hist),
                    np.max(mar_hist),
                    self._calculate_trend(np.array(ear_hist)),
                    self._calculate_trend(np.array(mar_hist)),
                    # Add more features as needed
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # padding
                ]
            
            features.append(frame_features)
        
        # Pad to window_size if needed
        while len(features) < self.window_size:
            features.append([0.0] * 16)
            
        return np.array(features[:self.window_size])
    
    def reset(self):
        """Reset all buffers."""
        self.ear_window.clear()
        self.mar_window.clear()
        self.pitch_window.clear()
        self.timestamps.clear()
        self.ear_history.clear()
        self.mar_history.clear()


# Alias
TemporalFeatures = TemporalFeatureExtractor
"""
User Profile Manager - Manages user profiles with calibration data.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from collections import deque


class UserProfileManager:
    """
    Manages user profiles for personalization.
    Supports calibration, adaptive thresholds, and training data collection.
    """
    
    def __init__(self, profiles_dir: str = "data/user_profiles"):
        """
        Args:
            profiles_dir: Directory to store user profile files
        """
        self.profiles_dir = profiles_dir
        os.makedirs(profiles_dir, exist_ok=True)
        
        self.current_user = None
        self.current_profile = None
        self.default_profile_name = "default"
        
    def create_profile(self, user_name: str) -> dict:
        """
        Create new user profile with default values.
        
        Args:
            user_name: Name identifier for the user
            
        Returns:
            Created profile dictionary
        """
        profile = {
            'name': user_name,
            'created_at': datetime.now().isoformat(),
            'last_used': datetime.now().isoformat(),
            'calibration': {
                'completed': False,
                'duration_seconds': 60,
                'samples_collected': 0,
                'ear_samples': [],
                'mar_samples': [],
            },
            'baseline': {
                'ear': 0.30,
                'mar': 0.15,
                'blink_rate': 15,
                'attention_baseline': 90,
                'pitch': 0.0,
            },
            'adaptive_thresholds': {
                'ear_drowsy': 0.22,
                'ear_sleeping': 0.15,
                'mar_yawn': 0.5,
                'blink_rate_low': 10,
                'blink_rate_high': 30,
            },
            'training_data': [],
            'session_stats': {
                'total_sessions': 0,
                'total_time_minutes': 0,
                'avg_fatigue_score': 0,
                'microsleep_count': 0,
                'sessions_data': [],  # Last 30 sessions
            },
            'settings': {
                'auto_calibrate': False,
                'save_training_data': True,
            }
        }
        
        profile_path = os.path.join(self.profiles_dir, f"{user_name}.json")
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
            
        print(f"Created new profile for user: {user_name}")
        return profile
    
    def load_profile(self, user_name: str = None) -> dict:
        """
        Load user profile, creating if doesn't exist.
        
        Args:
            user_name: Name of user to load (uses default if None)
            
        Returns:
            Loaded profile dictionary
        """
        if user_name is None:
            user_name = self.default_profile_name
            
        profile_path = os.path.join(self.profiles_dir, f"{user_name}.json")
        
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                self.current_profile = json.load(f)
                self.current_user = user_name
                self.current_profile['last_used'] = datetime.now().isoformat()
                self._save_profile()
                print(f"Loaded profile for user: {user_name}")
                return self.current_profile
        else:
            print(f"Profile not found, creating new: {user_name}")
            return self.create_profile(user_name)
    
    def get_current_profile(self) -> dict:
        """Get current profile, loading default if not loaded."""
        if self.current_profile is None:
            self.load_profile()
        return self.current_profile
    
    def start_calibration(self):
        """Initialize calibration data collection."""
        profile = self.get_current_profile()
        profile['calibration']['ear_samples'] = []
        profile['calibration']['mar_samples'] = []
        profile['calibration']['samples_collected'] = 0
        print("Calibration started - collect 60 seconds of data")
    
    def add_calibration_sample(self, ear: float, mar: float):
        """Add sample during calibration."""
        profile = self.get_current_profile()
        
        # Add samples
        profile['calibration']['ear_samples'].append(float(ear))
        profile['calibration']['mar_samples'].append(float(mar))
        profile['calibration']['samples_collected'] += 1
        
        # Keep only last 60 seconds worth (at 30fps = 1800 samples)
        max_samples = 1800
        if len(profile['calibration']['ear_samples']) > max_samples:
            profile['calibration']['ear_samples'] = \
                profile['calibration']['ear_samples'][-max_samples:]
            profile['calibration']['mar_samples'] = \
                profile['calibration']['mar_samples'][-max_samples:]
    
    def finish_calibration(self) -> bool:
        """
        Complete calibration and calculate baseline values.
        
        Returns:
            True if successful, False if insufficient samples
        """
        import numpy as np
        
        profile = self.get_current_profile()
        samples = profile['calibration']['samples_collected']
        
        if samples < 300:  # Need at least 10 seconds of data
            print(f"Insufficient calibration samples: {samples} (need 300+)")
            return False
        
        # Calculate baseline from median (more robust than mean)
        ear_samples = profile['calibration']['ear_samples']
        mar_samples = profile['calibration']['mar_samples']
        
        profile['baseline']['ear'] = float(np.median(ear_samples))
        profile['baseline']['mar'] = float(np.median(mar_samples))
        
        # Estimate blink rate (rough)
        # Count eye closures (EAR < 0.22)
        closures = sum(1 for e in ear_samples if e < 0.22)
        time_seconds = len(ear_samples) / 30.0
        profile['baseline']['blink_rate'] = int(closures / time_seconds * 60)
        
        # Update calibration status
        profile['calibration']['completed'] = True
        
        # Calculate adaptive thresholds based on user's baseline
        # These are personalized to the user's normal state
        profile['adaptive_thresholds']['ear_drowsy'] = \
            profile['baseline']['ear'] * 0.75  # 25% below baseline
        profile['adaptive_thresholds']['ear_sleeping'] = \
            profile['baseline']['ear'] * 0.5   # 50% below baseline
        profile['adaptive_thresholds']['mar_yawn'] = \
            profile['baseline']['mar'] * 2.0   # 2x baseline
        
        self._save_profile()
        
        print(f"Calibration completed:")
        print(f"  Baseline EAR: {profile['baseline']['ear']:.3f}")
        print(f"  Baseline MAR: {profile['baseline']['mar']:.3f}")
        print(f"  Blink rate: {profile['baseline']['blink_rate']}/min")
        
        return True
    
    def get_adaptive_thresholds(self) -> dict:
        """Get personalized thresholds for fatigue detection."""
        profile = self.get_current_profile()
        
        if profile['calibration']['completed']:
            return profile['adaptive_thresholds']
        else:
            # Return default thresholds if not calibrated
            return {
                'ear_drowsy': 0.22,
                'ear_sleeping': 0.15,
                'mar_yawn': 0.5,
                'blink_rate_low': 10,
                'blink_rate_high': 30,
            }
    
    def get_baseline(self) -> dict:
        """Get user's baseline values."""
        return self.get_current_profile()['baseline']
    
    def apply_personalization(self, ear: float, mar: float, 
                              blink_rate: int) -> dict:
        """
        Apply personalization adjustments to detection results.
        
        Args:
            ear: Current EAR value
            mar: Current MAR value
            blink_rate: Current blink rate
            
        Returns:
            Adjusted values and flags
        """
        profile = self.get_current_profile()
        thresholds = profile['adaptive_thresholds']
        baseline = profile['baseline']
        
        result = {
            'is_drowsy': ear < thresholds['ear_drowsy'],
            'is_sleeping': ear < thresholds['ear_sleeping'],
            'is_yawning': mar > thresholds['mar_yawn'],
            'is_blink_rate_low': blink_rate < thresholds['blink_rate_low'],
            'is_blink_rate_high': blink_rate > thresholds['blink_rate_high'],
            
            # Normalized values (how far from baseline)
            'ear_normalized': (ear - baseline['ear']) / (baseline['ear'] + 0.001),
            'mar_normalized': (mar - baseline['mar']) / (baseline['mar'] + 0.001),
            
            # Personal fatigue score adjustment
            'personal_fatigue_factor': 1.0
        }
        
        # Adjust fatigue score based on deviation from baseline
        if result['ear_normalized'] < -0.2:
            result['personal_fatigue_factor'] = 1.2
        elif result['ear_normalized'] < -0.1:
            result['personal_fatigue_factor'] = 1.1
            
        return result
    
    def add_training_sample(self, features: dict, predicted_label: str):
        """Add training data sample for this user."""
        if not self.get_current_profile()['settings']['save_training_data']:
            return
            
        sample = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'label': predicted_label
        }
        
        profile = self.get_current_profile()
        profile['training_data'].append(sample)
        
        # Keep only last 1000 samples
        if len(profile['training_data']) > 1000:
            profile['training_data'] = profile['training_data'][-1000:]
            
        self._save_profile()
    
    def update_session_stats(self, session_duration_minutes: float, 
                            avg_fatigue_score: float, microsleep_count: int):
        """Update session statistics after a work session."""
        profile = self.get_current_profile()
        
        stats = profile['session_stats']
        
        # Update totals
        stats['total_sessions'] += 1
        stats['total_time_minutes'] += int(session_duration_minutes)
        stats['microsleep_count'] += microsleep_count
        
        # Update average
        n = stats['total_sessions']
        stats['avg_fatigue_score'] = \
            (stats['avg_fatigue_score'] * (n-1) + avg_fatigue_score) / n
        
        # Add session to history (keep last 30)
        stats['sessions_data'].append({
            'date': datetime.now().isoformat(),
            'duration_minutes': int(session_duration_minutes),
            'avg_fatigue_score': avg_fatigue_score,
            'microsleep_count': microsleep_count
        })
        if len(stats['sessions_data']) > 30:
            stats['sessions_data'] = stats['sessions_data'][-30:]
        
        self._save_profile()
    
    def get_training_data(self) -> tuple:
        """
        Get training data for model retraining.
        
        Returns:
            Tuple of (features_array, labels_array)
        """
        import numpy as np
        
        profile = self.get_current_profile()
        training_data = profile['training_data']
        
        if not training_data:
            return None, None
        
        # Convert to arrays
        features_list = []
        labels = []
        
        label_map = {'awake': 0, 'drowsy': 1, 'sleeping': 2}
        
        for sample in training_data:
            if 'features' in sample and 'label' in sample:
                # Flatten features dict to array
                feat = sample['features']
                feature_vec = [
                    feat.get('ear_mean', 0.3),
                    feat.get('ear_std', 0),
                    feat.get('mar_mean', 0.15),
                    feat.get('mar_max', 0.15),
                    feat.get('yawning', False),
                ]
                features_list.append(feature_vec)
                
                if sample['label'] in label_map:
                    labels.append(label_map[sample['label']])
        
        if features_list:
            return np.array(features_list), np.array(labels)
        return None, None
    
    def reset_calibration(self):
        """Reset calibration to defaults."""
        profile = self.get_current_profile()
        profile['calibration'] = {
            'completed': False,
            'duration_seconds': 60,
            'samples_collected': 0,
            'ear_samples': [],
            'mar_samples': [],
        }
        profile['baseline'] = {
            'ear': 0.30,
            'mar': 0.15,
            'blink_rate': 15,
            'attention_baseline': 90,
            'pitch': 0.0,
        }
        self._save_profile()
        print("Calibration reset")
    
    def delete_profile(self, user_name: str = None):
        """Delete user profile."""
        if user_name is None:
            user_name = self.current_user
            
        profile_path = os.path.join(self.profiles_dir, f"{user_name}.json")
        if os.path.exists(profile_path):
            os.remove(profile_path)
            print(f"Deleted profile: {user_name}")
            
        if self.current_user == user_name:
            self.current_user = None
            self.current_profile = None
    
    def list_profiles(self) -> list:
        """List all available user profiles."""
        profiles = []
        for filename in os.listdir(self.profiles_dir):
            if filename.endswith('.json'):
                profiles.append(filename[:-5])  # Remove .json
        return profiles
    
    def _save_profile(self):
        """Save current profile to disk."""
        if self.current_user and self.current_profile:
            profile_path = os.path.join(self.profiles_dir, f"{self.current_user}.json")
            with open(profile_path, 'w') as f:
                json.dump(self.current_profile, f, indent=2)


# Alias
UserProfileManager = UserProfileManager
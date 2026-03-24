import os
import json
from typing import Dict, Any


class SettingsManager:
    """Manages application settings with persistence to JSON file."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self.config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
        os.makedirs(self.config_dir, exist_ok=True)
        
        self.settings_file = os.path.join(self.config_dir, "settings.json")
        self.calibration_file = os.path.join(self.config_dir, "calibration.json")
        
        self._settings = self._load_settings()
        self._calibration = self._load_calibration()
    
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from JSON file."""
        default = {
            "work_limit_minutes": 45,
            "posture_window_minutes": 3,
            "posture_bad_percent": 30,
            "yawn_limit": 4,
            "yawn_window_minutes": 10,
            "minimize_to_tray": True,
            "sound_volume": 0.5,
            "sound_enabled": True,
            "theme": "dark",
        }
        
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    default.update(loaded)
        except Exception:
            pass
        
        return default
    
    def _load_calibration(self) -> Dict[str, Any]:
        """Load calibration data from JSON file."""
        default = {
            "sensitivity": 1.0,
            "auto_calibrate": True,
            "face_calibrated": False,
            "hand_calibrated": False,
            "face_ear_threshold": 0.25,
            "face_mar_threshold": 0.5,
            "face_pitch_offset": 0.0,
        }
        
        try:
            if os.path.exists(self.calibration_file):
                with open(self.calibration_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    default.update(loaded)
        except Exception:
            pass
        
        return default
    
    def save_settings(self):
        """Save settings to JSON file."""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self._settings, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save settings: {e}")
    
    def save_calibration(self):
        """Save calibration data to JSON file."""
        try:
            with open(self.calibration_file, 'w', encoding='utf-8') as f:
                json.dump(self._calibration, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save calibration: {e}")
    
    @property
    def settings(self) -> Dict[str, Any]:
        return self._settings
    
    @settings.setter
    def settings(self, value: Dict[str, Any]):
        self._settings.update(value)
        self.save_settings()
    
    @property
    def calibration(self) -> Dict[str, Any]:
        return self._calibration
    
    @calibration.setter
    def calibration(self, value: Dict[str, Any]):
        self._calibration.update(value)
        self.save_calibration()
    
    def get(self, key: str, default=None):
        return self._settings.get(key, default)
    
    def set(self, key: str, value: Any):
        self._settings[key] = value
        self.save_settings()
    
    def get_calibration(self, key: str, default=None):
        return self._calibration.get(key, default)
    
    def set_calibration(self, key: str, value: Any):
        self._calibration[key] = value
        self.save_calibration()


settings_manager = SettingsManager()

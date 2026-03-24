import os
import json
from typing import Any, Dict, Optional


class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: str = None):
        if self._initialized:
            return
        
        self._initialized = True
        
        if config_path is None:
            self.config_path = os.path.join(
                os.path.dirname(__file__), 
                "..", 
                "config.json"
            )
        else:
            self.config_path = config_path
        
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self):
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
            else:
                self._config = self._get_default_config()
                self._save_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'camera': {
                'index': 0,
                'width': 640,
                'height': 480,
                'fps': 30
            },
            'face': {
                'min_detection_confidence': 0.5,
                'yaw_threshold': 40,
                'pitch_offset': 5.0,
                'pitch_threshold_min': 0.0,
                'pitch_threshold_max': 30.0
            },
            'fatigue': {
                'window_size_seconds': 30,
                'ear_threshold_closed': 0.2,
                'ear_threshold_normal': 0.35,
                'mar_threshold_yawn': 0.6,
                'fatigue_severe_threshold': 80,
                'fatigue_moderate_threshold': 60,
                'fatigue_mild_threshold': 40
            },
            'posture': {
                'window_size_seconds': 5,
                'bad_posture_threshold': 0.15,
                'head_forward_threshold': 0.08,
                'posture_time_trigger': 1.0
            },
            'gesture': {
                'enabled': False,
                'sensitivity': 1.0,
                'min_detection_confidence': 0.7,
                'max_hands': 1
            },
            'calibration': {
                'auto_calibrate': True,
                'samples_required': 20,
                'face_calibrated': False,
                'hand_calibrated': False
            },
            'notifications': {
                'work_limit_minutes': 45,
                'posture_window_minutes': 3,
                'posture_bad_percent': 30,
                'yawn_limit': 3,
                'yawn_window_minutes': 10,
                'sound_enabled': True,
                'sound_volume': 0.5
            },
            'ui': {
                'theme': 'dark',
                'window_width': 1280,
                'window_height': 800
            },
            'logging': {
                'level': 'INFO',
                'file_enabled': True,
                'max_file_size_mb': 5,
                'backup_count': 5
            },
            'pomodoro': {
                'work_minutes': 25,
                'break_minutes': 5,
                'enabled': False
            }
        }
    
    def _save_config(self):
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self._save_config()
    
    def get_section(self, section: str) -> Dict[str, Any]:
        return self._config.get(section, {})
    
    def set_section(self, section: str, values: Dict[str, Any]):
        self._config[section] = values
        self._save_config()
    
    def reload(self):
        self._load_config()
    
    @property
    def camera(self) -> Dict[str, Any]:
        return self._config.get('camera', {})
    
    @property
    def face(self) -> Dict[str, Any]:
        return self._config.get('face', {})
    
    @property
    def fatigue(self) -> Dict[str, Any]:
        return self._config.get('fatigue', {})
    
    @property
    def posture(self) -> Dict[str, Any]:
        return self._config.get('posture', {})
    
    @property
    def gesture(self) -> Dict[str, Any]:
        return self._config.get('gesture', {})
    
    @property
    def calibration(self) -> Dict[str, Any]:
        return self._config.get('calibration', {})
    
    @property
    def notifications(self) -> Dict[str, Any]:
        return self._config.get('notifications', {})
    
    @property
    def ui(self) -> Dict[str, Any]:
        return self._config.get('ui', {})
    
    @property
    def logging(self) -> Dict[str, Any]:
        return self._config.get('logging', {})
    
    @property
    def pomodoro(self) -> Dict[str, Any]:
        return self._config.get('pomodoro', {})


config_manager = ConfigManager()

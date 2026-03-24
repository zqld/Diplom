import json
import os
from collections import deque
import time


class CalibrationManager:
    """
    Менеджер калибровки для лица и руки.
    Управляет настройками калибровки и их сохранением/загрузкой.
    """
    
    def __init__(self, config_path="data/calibration.json"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Face calibration data
        self.face_calibration = {
            "baseline_ear": 0.30,
            "baseline_mar": 0.15,
            "baseline_pitch": 0.0,
            "calibrated": False
        }
        
        # Hand calibration data
        self.hand_calibration = {
            "baseline_hand_size": 0.25,
            "calibrated": False
        }
        
        # Apply loaded config
        if self.config.get("face_calibrated", False):
            self.face_calibration = {
                "baseline_ear": self.config.get("face_ear", 0.30),
                "baseline_mar": self.config.get("face_mar", 0.15),
                "baseline_pitch": self.config.get("face_pitch", 0.0),
                "calibrated": True
            }
        
        if self.config.get("hand_calibrated", False):
            self.hand_calibration = {
                "baseline_hand_size": self.config.get("hand_size", 0.25),
                "calibrated": True
            }
        
        self.auto_calibrate = self.config.get("auto_calibrate", False)
        self.sensitivity = self.config.get("sensitivity", 1.0)
        
        self._face_samples = deque(maxlen=30)
        self._hand_samples = deque(maxlen=30)
        self._is_calibrating_face = False
        self._is_calibrating_hand = False
    
    def _load_config(self):
        """Загрузить конфигурацию из файла."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_config(self):
        """Сохранить конфигурацию в файл."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        self.config.update({
            "face_calibrated": self.face_calibration["calibrated"],
            "face_ear": self.face_calibration["baseline_ear"],
            "face_mar": self.face_calibration["baseline_mar"],
            "face_pitch": self.face_calibration["baseline_pitch"],
            "hand_calibrated": self.hand_calibration["calibrated"],
            "hand_size": self.hand_calibration["baseline_hand_size"],
            "auto_calibrate": self.auto_calibrate,
            "sensitivity": self.sensitivity,
        })
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def start_face_calibration(self):
        """Начать калибровку лица."""
        self._face_samples.clear()
        self._is_calibrating_face = True
    
    def add_face_sample(self, ear, mar, pitch):
        """Добавить образец для калибровки лица."""
        if self._is_calibrating_face:
            self._face_samples.append({
                "ear": ear,
                "mar": mar,
                "pitch": pitch,
                "time": time.time()
            })
    
    def finish_face_calibration(self):
        """Завершить калибровку лица и сохранить результаты."""
        if len(self._face_samples) < 10:
            self._is_calibrating_face = False
            return False
        
        recent_samples = list(self._face_samples)[-20:]
        
        self.face_calibration = {
            "baseline_ear": sum(s["ear"] for s in recent_samples) / len(recent_samples),
            "baseline_mar": sum(s["mar"] for s in recent_samples) / len(recent_samples),
            "baseline_pitch": sum(s["pitch"] for s in recent_samples) / len(recent_samples),
            "calibrated": True
        }
        
        self._is_calibrating_face = False
        self._save_config()
        return True
    
    def start_hand_calibration(self):
        """Начать калибровку руки."""
        self._hand_samples.clear()
        self._is_calibrating_hand = True
    
    def add_hand_sample(self, hand_size):
        """Добавить образец для калибровки руки."""
        if self._is_calibrating_hand:
            self._hand_samples.append({
                "size": hand_size,
                "time": time.time()
            })
    
    def finish_hand_calibration(self):
        """Завершить калибровку руки."""
        if len(self._hand_samples) < 10:
            self._is_calibrating_hand = False
            return False
        
        recent_samples = list(self._hand_samples)[-20:]
        
        self.hand_calibration = {
            "baseline_hand_size": sum(s["size"] for s in recent_samples) / len(recent_samples),
            "calibrated": True
        }
        
        self._is_calibrating_hand = False
        self._save_config()
        return True
    
    def auto_calibrate_if_needed(self, ear, mar, pitch, hand_size):
        """Автоматическая калибровка если включена и не откалибровано."""
        if not self.auto_calibrate:
            return
        
        current_time = time.time()
        
        # Auto calibrate face if not calibrated
        if not self.face_calibration["calibrated"]:
            if len(self._face_samples) < 30:
                self._face_samples.append({
                    "ear": ear, "mar": mar, "pitch": pitch, "time": current_time
                })
            elif len(self._face_samples) == 30:
                self.finish_face_calibration()
        
        # Auto calibrate hand if not calibrated
        if not self.hand_calibration["calibrated"]:
            if len(self._hand_samples) < 30:
                self._hand_samples.append({
                    "size": hand_size, "time": current_time
                })
            elif len(self._hand_samples) == 30:
                self.finish_hand_calibration()
    
    def get_calibrated_ear_threshold(self):
        """Получить порог EAR с учётом калибровки."""
        if self.face_calibration["calibrated"]:
            return self.face_calibration["baseline_ear"] * 0.75
        return 0.22
    
    def get_calibrated_mar_threshold(self):
        """Получить порог MAR с учётом калибровки."""
        if self.face_calibration["calibrated"]:
            return max(0.5, self.face_calibration["baseline_mar"] * 1.3)
        return 0.6
    
    def get_hand_size_factor(self):
        """Получить коэффициент размера руки для корректировки чувствительности."""
        if self.hand_calibration["calibrated"]:
            return 0.25 / self.hand_calibration["baseline_hand_size"]
        return 1.0
    
    def set_sensitivity(self, value):
        """Установить чувствительность."""
        self.sensitivity = max(0.3, min(3.0, value))
        self._save_config()
    
    def set_auto_calibrate(self, value):
        """Установить режим автоматической калибровки."""
        self.auto_calibrate = value
        self._save_config()
    
    def reset_face_calibration(self):
        """Сбросить калибровку лица."""
        self.face_calibration = {
            "baseline_ear": 0.30,
            "baseline_mar": 0.15,
            "baseline_pitch": 0.0,
            "calibrated": False
        }
        self._face_samples.clear()
        self._save_config()
    
    def reset_hand_calibration(self):
        """Сбросить калибровку руки."""
        self.hand_calibration = {
            "baseline_hand_size": 0.25,
            "calibrated": False
        }
        self._hand_samples.clear()
        self._save_config()
    
    def get_status(self):
        """Получить статус калировки."""
        return {
            "face_calibrated": self.face_calibration["calibrated"],
            "hand_calibrated": self.hand_calibration["calibrated"],
            "auto_calibrate": self.auto_calibrate,
            "sensitivity": self.sensitivity,
            "is_calibrating_face": self._is_calibrating_face,
            "is_calibrating_hand": self._is_calibrating_hand,
        }

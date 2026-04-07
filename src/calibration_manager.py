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
        
        # Posture calibration data
        self.posture_calibration = {
            "baseline_pitch": 0.0,
            "calibrated": False,
        }
        if self.config.get("posture_calibrated", False):
            self.posture_calibration = {
                "baseline_pitch": self.config.get("posture_pitch", 0.0),
                "calibrated": True,
            }

        # Gesture active zone calibration
        self.gesture_zone = {
            "x_min": 0.15,
            "y_min": 0.15,
            "x_max": 0.85,
            "y_max": 0.85,
            "calibrated": False,
        }
        if self.config.get("gesture_zone_calibrated", False):
            self.gesture_zone = {
                "x_min": self.config.get("gz_x_min", 0.15),
                "y_min": self.config.get("gz_y_min", 0.15),
                "x_max": self.config.get("gz_x_max", 0.85),
                "y_max": self.config.get("gz_y_max", 0.85),
                "calibrated": True,
            }

        self._face_samples = deque(maxlen=30)
        self._hand_samples = deque(maxlen=30)
        self._is_calibrating_face = False
        self._is_calibrating_hand = False
        self._face_cal_start_time = 0.0
        self._hand_cal_start_time = 0.0

        # Posture calibration
        self._posture_samples: deque = deque(maxlen=30)
        self._is_calibrating_posture = False
        self._posture_cal_start_time = 0.0

        # Gesture zone calibration
        # step: 'topleft' | 'bottomright' | None
        self._is_calibrating_zone = False
        self._zone_step: str = 'topleft'   # current sub-step
        self._zone_topleft_samples: list = []
        self._zone_bottomright_samples: list = []
        self._zone_cal_start_time = 0.0
    
    def _load_config(self):
        """Загрузить конфигурацию из файла."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def load_config(self):
        """Перезагрузить конфигурацию из файла (обновить состояние)."""
        self.config = self._load_config()
        
        # Обновить статус калибровки лица
        if self.config.get("face_calibrated", False):
            self.face_calibration = {
                "baseline_ear": self.config.get("face_ear", 0.30),
                "baseline_mar": self.config.get("face_mar", 0.15),
                "baseline_pitch": self.config.get("face_pitch", 0.0),
                "calibrated": True
            }
        
        # Обновить статус калибровки руки
        if self.config.get("hand_calibrated", False):
            self.hand_calibration = {
                "baseline_hand_size": self.config.get("hand_size", 0.25),
                "calibrated": True
            }

        # Обновить статус калибровки осанки
        if self.config.get("posture_calibrated", False):
            self.posture_calibration = {
                "baseline_pitch": self.config.get("posture_pitch", 0.0),
                "calibrated": True,
            }

        # Обновить статус калибровки зоны жестов
        if self.config.get("gesture_zone_calibrated", False):
            self.gesture_zone = {
                "x_min": self.config.get("gz_x_min", 0.15),
                "y_min": self.config.get("gz_y_min", 0.15),
                "x_max": self.config.get("gz_x_max", 0.85),
                "y_max": self.config.get("gz_y_max", 0.85),
                "calibrated": True,
            }

        self.sensitivity = self.config.get("sensitivity", 1.0)
    
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
            "posture_calibrated": self.posture_calibration["calibrated"],
            "posture_pitch": self.posture_calibration["baseline_pitch"],
            "gesture_zone_calibrated": self.gesture_zone["calibrated"],
            "gz_x_min": self.gesture_zone["x_min"],
            "gz_y_min": self.gesture_zone["y_min"],
            "gz_x_max": self.gesture_zone["x_max"],
            "gz_y_max": self.gesture_zone["y_max"],
            "auto_calibrate": self.auto_calibrate,
            "sensitivity": self.sensitivity,
        })
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def start_face_calibration(self):
        """Начать калибровку лица."""
        self._face_samples.clear()
        self._is_calibrating_face = True
        self._face_cal_start_time = time.time()

    def add_face_sample(self, ear, mar, pitch):
        """Добавить образец для калибровки лица."""
        if self._is_calibrating_face:
            # Минимальное время калибрации 2 сек — иначе сессия завершается
            # мгновенно, когда сэмплы уже собраны ранее
            if time.time() - self._face_cal_start_time < 2.0:
                return
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
        self._hand_cal_start_time = time.time()

    def add_hand_sample(self, hand_size):
        """Добавить образец для калибровки руки."""
        if self._is_calibrating_hand:
            # Минимальное время калибрации 2 сек
            if time.time() - self._hand_cal_start_time < 2.0:
                return
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
    
    # ── Posture calibration ────────────────────────────────────────────────────

    def start_posture_calibration(self):
        """Начать калибровку осанки (пользователь сидит прямо)."""
        self._posture_samples.clear()
        self._is_calibrating_posture = True
        self._posture_cal_start_time = time.time()

    def add_posture_sample(self, pitch: float):
        """Добавить измерение угла наклона головы во время калибровки осанки."""
        if self._is_calibrating_posture:
            if time.time() - self._posture_cal_start_time < 2.0:
                return
            self._posture_samples.append(pitch)

    def finish_posture_calibration(self) -> bool:
        """Завершить калибровку осанки и сохранить базовый угол."""
        if len(self._posture_samples) < 10:
            self._is_calibrating_posture = False
            return False
        recent = list(self._posture_samples)[-20:]
        baseline = sum(recent) / len(recent)
        self.posture_calibration = {
            "baseline_pitch": baseline,
            "calibrated": True,
        }
        self._is_calibrating_posture = False
        self._save_config()
        return True

    def reset_posture_calibration(self):
        """Сбросить калибровку осанки."""
        self.posture_calibration = {"baseline_pitch": 0.0, "calibrated": False}
        self._posture_samples.clear()
        self._save_config()

    # ── Gesture zone calibration ───────────────────────────────────────────────

    def start_gesture_zone_calibration(self):
        """Начать калибровку активной зоны жестов (шаг 1 — верхний левый угол)."""
        self._zone_topleft_samples.clear()
        self._zone_bottomright_samples.clear()
        self._zone_step = 'topleft'
        self._is_calibrating_zone = True
        self._zone_cal_start_time = time.time()

    def advance_gesture_zone_step(self):
        """Переключиться на шаг 2 — нижний правый угол."""
        self._zone_step = 'bottomright'
        self._zone_cal_start_time = time.time()

    def add_zone_sample(self, norm_x: float, norm_y: float):
        """Добавить нормализованную позицию руки (0..1) в текущий шаг."""
        if not self._is_calibrating_zone:
            return
        if time.time() - self._zone_cal_start_time < 2.5:
            return
        if self._zone_step == 'topleft':
            self._zone_topleft_samples.append((norm_x, norm_y))
        elif self._zone_step == 'bottomright':
            self._zone_bottomright_samples.append((norm_x, norm_y))

    def finish_gesture_zone_calibration(self) -> bool:
        """Завершить калибровку зоны и сохранить x_min, y_min, x_max, y_max."""
        if (len(self._zone_topleft_samples) < 5
                or len(self._zone_bottomright_samples) < 5):
            self._is_calibrating_zone = False
            return False
        tl = self._zone_topleft_samples
        br = self._zone_bottomright_samples
        x_min = sum(x for x, _ in tl) / len(tl)
        y_min = sum(y for _, y in tl) / len(tl)
        x_max = sum(x for x, _ in br) / len(br)
        y_max = sum(y for _, y in br) / len(br)
        # Гарантируем правильное направление осей (min < max)
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        # Минимальный размер зоны 0.25 по каждой оси — защита от вырожденной зоны
        if x_max - x_min < 0.25:
            cx = (x_min + x_max) / 2
            x_min, x_max = max(0.0, cx - 0.125), min(1.0, cx + 0.125)
        if y_max - y_min < 0.25:
            cy = (y_min + y_max) / 2
            y_min, y_max = max(0.0, cy - 0.125), min(1.0, cy + 0.125)
        self.gesture_zone = {
            "x_min": round(x_min, 4),
            "y_min": round(y_min, 4),
            "x_max": round(x_max, 4),
            "y_max": round(y_max, 4),
            "calibrated": True,
        }
        self._is_calibrating_zone = False
        self._save_config()
        return True

    def reset_gesture_zone_calibration(self):
        """Сбросить калибровку активной зоны."""
        self.gesture_zone = {
            "x_min": 0.15, "y_min": 0.15,
            "x_max": 0.85, "y_max": 0.85,
            "calibrated": False,
        }
        self._zone_topleft_samples.clear()
        self._zone_bottomright_samples.clear()
        self._save_config()

    # ── Auto calibration ───────────────────────────────────────────────────────

    def auto_calibrate_if_needed(self, ear, mar, pitch, hand_size=None):
        """Автоматическая калибровка если включена и не откалибровано.

        Возвращает:
            tuple (face_done, hand_done) — True если соответствующая
            калибровка только что завершилась в этом вызове.
        """
        if not self.auto_calibrate:
            return False, False

        # Не мешаем ручной калибровке — она управляется отдельно
        if (self._is_calibrating_face or self._is_calibrating_hand
                or self._is_calibrating_posture or self._is_calibrating_zone):
            return False, False

        current_time = time.time()
        face_done = False
        hand_done = False

        # Auto calibrate face if not calibrated
        if not self.face_calibration["calibrated"]:
            if len(self._face_samples) < 30:
                self._face_samples.append({
                    "ear": ear, "mar": mar, "pitch": pitch, "time": current_time
                })
            if len(self._face_samples) >= 30:
                face_done = self.finish_face_calibration()

        # Auto calibrate hand only when we have real hand data
        if not self.hand_calibration["calibrated"] and hand_size:
            if len(self._hand_samples) < 30:
                self._hand_samples.append({
                    "size": hand_size, "time": current_time
                })
            if len(self._hand_samples) >= 30:
                hand_done = self.finish_hand_calibration()

        return face_done, hand_done
    
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
            "posture_calibrated": self.posture_calibration["calibrated"],
            "gesture_zone_calibrated": self.gesture_zone["calibrated"],
            "auto_calibrate": self.auto_calibrate,
            "sensitivity": self.sensitivity,
            "is_calibrating_face": self._is_calibrating_face,
            "is_calibrating_hand": self._is_calibrating_hand,
            "is_calibrating_posture": self._is_calibrating_posture,
            "is_calibrating_zone": self._is_calibrating_zone,
            "zone_step": self._zone_step,
        }

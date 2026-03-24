import os
import sys
import threading
from enum import Enum


class NotificationSound(Enum):
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CALIBRATION_START = "calibration_start"
    CALIBRATION_DONE = "calibration_done"
    ATTENTION = "attention"


class SoundManager:
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
        self.enabled = True
        self.volume = 0.5
        self._init_pygame()
    
    def _init_pygame(self):
        try:
            import pygame
            pygame.mixer.init()
            self._pygame = pygame
            self._available = True
        except ImportError:
            self._pygame = None
            self._available = False
    
    def _get_sound_path(self, sound_type: NotificationSound) -> str:
        sounds_dir = os.path.join(os.path.dirname(__file__), "..", "sounds")
        return os.path.join(sounds_dir, f"{sound_type.value}.wav")
    
    def _beep(self, frequency=800, duration=150, volume=None):
        if not self.enabled:
            return
        vol = volume if volume else self.volume
        
        try:
            if self._available and self._pygame:
                import numpy as np
                sample_rate = 44100
                samples = int(sample_rate * duration / 1000)
                t = np.linspace(0, duration / 1000, samples, False)
                wave = np.sin(2 * np.pi * frequency * t)
                wave = wave * vol
                wave = (wave * 32767).astype(np.int16)
                stereo = np.column_stack((wave, wave))
                sound = self._pygame.mixer.Sound(buffer=stereo.tobytes())
                sound.play()
            else:
                import winsound
                winsound.Beep(frequency, int(duration * vol))
        except Exception:
            pass
    
    def play(self, sound_type: NotificationSound = None, frequency: int = None, duration: int = 150):
        if not self.enabled:
            return
        
        if sound_type:
            frequencies = {
                NotificationSound.SUCCESS: (600, 100),
                NotificationSound.WARNING: (800, 200),
                NotificationSound.ERROR: (400, 300),
                NotificationSound.CALIBRATION_START: (500, 150),
                NotificationSound.CALIBRATION_DONE: (800, 100),
                NotificationSound.ATTENTION: (1000, 100),
            }
            freq, dur = frequencies.get(sound_type, (800, 150))
            threading.Thread(target=self._beep, args=(freq, dur), daemon=True).start()
        elif frequency:
            threading.Thread(target=self._beep, args=(frequency, duration), daemon=True).start()
    
    def success(self):
        self.play(NotificationSound.SUCCESS)
    
    def warning(self):
        self.play(NotificationSound.WARNING)
    
    def error(self):
        self.play(NotificationSound.ERROR)
    
    def calibration_start(self):
        self.play(NotificationSound.CALIBRATION_START)
    
    def calibration_done(self):
        self.play(NotificationSound.CALIBRATION_DONE)
    
    def attention(self):
        self.play(NotificationSound.ATTENTION)
    
    def set_enabled(self, enabled: bool):
        self.enabled = enabled
    
    def set_volume(self, volume: float):
        self.volume = max(0.0, min(1.0, volume))


sound_manager = SoundManager()

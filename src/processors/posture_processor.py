import time
from src.posture_analyzer import PostureAnalyzer
from src.config_manager import config_manager
from src.logger import logger


class PostureProcessor:
    def __init__(self, calibration_manager=None):
        self._config = config_manager.posture
        self._window_size = self._config.get('window_size_seconds', 5)
        self._calibration = calibration_manager

        self.analyzer = PostureAnalyzer(window_size_seconds=self._window_size)
        self._posture_start_time = None
        self._time_trigger = self._config.get('posture_time_trigger', 1.0)
        self._last_event_time = 0
        self._cooldown = 1.0
    
    def process(self, landmarks, frame_width, frame_height, pitch: float, current_time: float) -> dict:
        result = {
            'is_bad': False,
            'posture_score': 0,
            'posture_level': 'good',
            'posture_status': 'Good',
            'head_tilt': 0,
            'head_forward': 0,
            'event': None,
            'pitch_alert': False
        }
        
        try:
            if landmarks:
                posture_data = self.analyzer.update(
                    landmarks,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    current_time=current_time
                )
                
                result['posture_score'] = posture_data.get("posture_score", 0)
                result['posture_level'] = posture_data.get("posture_level", "good")
                result['head_tilt'] = posture_data.get("head_tilt", 0)
                result['head_forward'] = posture_data.get("head_forward", 0)

                if posture_data.get("is_bad", False):
                    result['is_bad'] = True
                    result['posture_status'] = "Bad Posture"
                    
                    tilt = result['head_tilt']
                    forward = result['head_forward']
                    
                    if tilt > 15:
                        result['event'] = f"Наклон головы ({int(tilt)}°)"
                    elif forward > self._config.get('head_forward_threshold', 0.08):
                        result['event'] = "Голова вперёд"
                    else:
                        result['event'] = "Плохая осанка"
                    
                    if current_time - self._last_event_time >= self._cooldown:
                        self._last_event_time = current_time
            
            pitch_min = config_manager.face.get('pitch_threshold_min', -5.0)
            pitch_max = config_manager.face.get('pitch_threshold_max', 35.0)
            if pitch < pitch_min or pitch > pitch_max:
                if self._posture_start_time is None:
                    self._posture_start_time = current_time
                elif current_time - self._posture_start_time > self._time_trigger:
                    result['is_bad'] = True
                    result['pitch_alert'] = True
                    result['posture_status'] = "Bad Posture"
                    if result['event'] is None:
                        result['event'] = f"Наклон головы ({int(pitch)}°)"
            else:
                self._posture_start_time = None
            
            logger.debug(f"Posture: level={result['posture_level']}, bad={result['is_bad']}")
            
        except Exception as e:
            logger.error(f"Error in PostureProcessor: {e}")
        
        return result
    
    def reset(self):
        self.analyzer = PostureAnalyzer(window_size_seconds=self._window_size)
        self._posture_start_time = None
        self._last_event_time = 0
    
    def set_calibration_manager(self, calibration_manager):
        """Обновить ссылку на CalibrationManager (можно вызвать после инициализации)."""
        self._calibration = calibration_manager

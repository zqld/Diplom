from typing import Optional
from src.fatigue_analyzer import FatigueAnalyzer
from src.config_manager import config_manager
from src.logger import logger


class FatigueProcessor:
    def __init__(self, calibration_manager=None):
        self._config = config_manager.fatigue
        self._window_size = self._config.get('window_size_seconds', 30)
        
        self.analyzer = FatigueAnalyzer(
            window_size_seconds=self._window_size,
            calibration_manager=calibration_manager,
        )
        self._last_event_time = 0
        self._cooldown = 2.0
        self._yawn_limit = None
    
    def process(self, ear: float, mar: float, pitch: float, emotion: str,
                current_time: float) -> dict:
        result = {
            'fatigue_score': 0,
            'fatigue_level': 'normal',
            'fatigue_status': 'Normal',
            'event': None,
            'cooldown_active': False
        }
        
        try:
            fatigue_data = self.analyzer.update(ear, mar, pitch, emotion, current_time)
            
            result['fatigue_score'] = fatigue_data.get("fatigue_score", 0)
            result['fatigue_level'] = fatigue_data.get("fatigue_level", "normal")
            result['fatigue_status'] = fatigue_data.get("fatigue_level", "normal").capitalize()
            
            # Yawn cooldown зависит от лимита зевков пользователя
            if self._yawn_limit is not None:
                cooldown = max(1.0, float(self._yawn_limit) * 2.0)
            else:
                cooldown = self._cooldown
            
            if mar > self._config.get('mar_threshold_yawn', 0.6):
                if current_time - self._last_event_time >= cooldown:
                    result['event'] = "Зевок"
                    result['fatigue_status'] = "Yawning"
                    self._last_event_time = current_time
            else:
                fatigue_event = self.analyzer.get_fatigue_event(current_time)
                if fatigue_event is not None:
                    result['event'] = fatigue_event
                    result['fatigue_status'] = fatigue_event
                    if fatigue_event == "Сильная усталость":
                        result['fatigue_status'] = "Fatigued"
                    elif fatigue_event == "Умеренная усталость":
                        result['fatigue_status'] = "Tired"
                    elif fatigue_event == "Лёгкая усталость (снижение)":
                        result['fatigue_status'] = "Mild"
            
            logger.debug(f"Fatigue: level={result['fatigue_level']}, score={result['fatigue_score']}")
            
        except Exception as e:
            logger.error(f"Error in FatigueProcessor: {e}")
        
        return result
    
    def reset(self):
        self.analyzer = FatigueAnalyzer(
            window_size_seconds=self._window_size,
            calibration_manager=self.analyzer._calibration_manager
            if hasattr(self.analyzer, '_calibration_manager') else None,
        )
        self._last_event_time = 0

    def set_calibration_manager(self, calibration_manager):
        """Обновить CalibrationManager и пересоздать анализатор с новыми порогами."""
        self.analyzer = FatigueAnalyzer(
            window_size_seconds=self._window_size,
            calibration_manager=calibration_manager,
        )

    def set_yawn_limit(self, yawn_limit: int):
        """Обновить yawn_limit (влияет на cooldown между зевками)."""
        # yawn_limit будет применён в process() при следующем вызове
        self._yawn_limit = yawn_limit

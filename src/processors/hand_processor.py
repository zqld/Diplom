from src.hand_tracker import HandTracker
from src.gesture_controller import GestureController
from src.config_manager import config_manager
from src.logger import logger


class HandProcessor:
    def __init__(self, calibration_manager=None):
        self._config = config_manager.gesture
        self._max_hands = self._config.get('max_hands', 1)
        self._min_confidence = self._config.get('min_detection_confidence', 0.7)
        
        self.tracker = HandTracker(
            max_hands=self._max_hands,
            min_detection_confidence=self._min_confidence
        )
        self.gesture_controller = GestureController(calibration_manager=calibration_manager)
        
        self._enabled = self._config.get('enabled', False)
    
    def process(self, frame, frame_width, frame_height) -> dict:
        result = {
            'detected': False,
            'landmarks': None,
            'gesture': 'none',
            'current_gesture': None
        }
        
        try:
            hand_frame, hand_results = self.tracker.process_frame(frame, draw=True)
            hand_detected = self.tracker.is_hand_present(hand_results)
            
            result['detected'] = hand_detected
            
            if hand_detected and self._enabled:
                hand_landmarks = self.tracker.get_landmarks(hand_results)
                if hand_landmarks:
                    fingers_up = self.tracker.get_fingers_up(hand_landmarks[0])
                    gesture = self.gesture_controller.process_hand(
                        hand_landmarks[0],
                        frame_width,
                        frame_height,
                        fingers_up
                    )
                    result['gesture'] = gesture
                    result['current_gesture'] = gesture
            
            logger.debug(f"Hand: detected={hand_detected}, gesture={result['gesture']}")
            
        except Exception as e:
            logger.error(f"Error in HandProcessor: {e}")
        
        return result
    
    def toggle_gesture_control(self) -> bool:
        self._enabled = self.gesture_controller.toggle()
        return self._enabled
    
    def disable_gesture_control(self):
        self.gesture_controller.disable()
        self._enabled = False
    
    def set_enabled(self, enabled: bool):
        if enabled and not self._enabled:
            self._enabled = self.gesture_controller.toggle()
        elif not enabled and self._enabled:
            self.gesture_controller.disable()
            self._enabled = False
    
    def is_enabled(self) -> bool:
        return self._enabled
    
    def reload_config(self):
        self._config = config_manager.gesture
        self._max_hands = self._config.get('max_hands', 1)
        self._min_confidence = self._config.get('min_detection_confidence', 0.7)
        
        self.tracker = HandTracker(
            max_hands=self._max_hands,
            min_detection_confidence=self._min_confidence
        )
        
        sensitivity = self._config.get('sensitivity', 1.0)
        self.gesture_controller.set_sensitivity(sensitivity)

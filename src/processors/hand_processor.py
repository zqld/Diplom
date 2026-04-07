from src.hand_tracker import HandTracker
from src.gesture_controller import GestureController
from src.config_manager import config_manager
from src.logger import logger
import math


class HandProcessor:
    def __init__(self, calibration_manager=None):
        self._config = config_manager.gesture
        self._max_hands = self._config.get('max_hands', 1)
        self._min_detection_conf = self._config.get('min_detection_confidence', 0.75)
        self._min_tracking_conf  = self._config.get('min_tracking_confidence',  0.65)

        # model_complexity=0 — лёгкая модель MediaPipe Hands (~5 ms/frame).
        # Это критично: при complexity=1 рука обрабатывается ~15 ms, что
        # при вызове каждые 2 кадра (~30 ms окно) даёт 50 % CPU нагрузку.
        self.tracker = HandTracker(
            max_hands=self._max_hands,
            min_detection_confidence=self._min_detection_conf,
            min_tracking_confidence=self._min_tracking_conf,
            model_complexity=0,
        )
        self.gesture_controller = GestureController(calibration_manager=calibration_manager)

        self._last_hand_position = None
        self._hand_size = None

        # По умолчанию ВЫКЛЮЧЕНО
        self._enabled = self._config.get('enabled', False)
    
    def _calculate_hand_size(self, hand_landmarks):
        """Рассчитать размер руки."""
        wrist = hand_landmarks.landmark[0]
        middle_tip = hand_landmarks.landmark[12]
        return math.sqrt((wrist.x - middle_tip.x)**2 + (wrist.y - middle_tip.y)**2)
    
    def process(self, frame, frame_width, frame_height) -> dict:
        result = {
            'detected': False,
            'landmarks': None,
            'gesture': 'none',
            'current_gesture': None,
            'hand_size': None,
            'palm_x': None,   # нормализованная позиция (0..1) для калибровки зоны
            'palm_y': None,
        }
        
        try:
            hand_frame, hand_results = self.tracker.process_frame(frame, draw=True)
            hand_detected = self.tracker.is_hand_present(hand_results)
            
            result['detected'] = hand_detected
            
            # Всегда вычисляем hand_size и позицию ладони — даже когда жесты выключены
            # (нужно для калибровки зоны жестов и калибровки размера руки)
            hand_landmarks = None
            if hand_detected:
                hand_landmarks = self.tracker.get_landmarks(hand_results)
                if hand_landmarks:
                    hand_size = self._calculate_hand_size(hand_landmarks[0])
                    result['hand_size'] = hand_size
                    self._hand_size = hand_size
                    # Нормализованная позиция landmark[9] (ладонь) — для зонной калибровки
                    result['palm_x'] = hand_landmarks[0].landmark[9].x
                    result['palm_y'] = hand_landmarks[0].landmark[9].y

            if hand_detected and self._enabled and hand_landmarks:
                fingers_up = self.tracker.get_fingers_up(hand_landmarks[0])

                palm_norm_x = result['palm_x']
                palm_norm_y = result['palm_y']

                # Вычисляем позицию руки для сглаживания
                hand_x = palm_norm_x * frame_width
                hand_y = palm_norm_y * frame_height
                
                gesture = self.gesture_controller.process_hand(
                    hand_landmarks[0],
                    frame_width,
                    frame_height,
                    fingers_up,
                    hand_x=hand_x,
                    hand_y=hand_y,
                    hand_size=hand_size
                )
                result['gesture'] = gesture
                result['current_gesture'] = gesture
                
                self._last_hand_position = (hand_x, hand_y)
            
            # logger.debug вынесен из горячего пути — форматирование строки каждый кадр дорого
            
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

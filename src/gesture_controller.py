import time
import math
import pyautogui
from collections import deque


class GestureController:
    """
    Контроллер жестов для управления мышью.
    Распознаёт жесты и выполняет соответствующие действия.
    """
    
    def __init__(self, screen_width=None, screen_height=None, calibration_manager=None):
        """
        Инициализация контроллера жестов.
        
        Args:
            screen_width, screen_height: Размеры экрана
            calibration_manager: Менеджер калибровки
        """
        # Настройки экрана
        self.screen_width = screen_width or pyautogui.size()[0]
        self.screen_height = screen_height or pyautogui.size()[1]
        
        # Calibration manager
        self.calibration = calibration_manager
        
        # Параметры управления
        self.smoothing = 10  # Коэффициент сглаживания (больше = плавнее)
        self.click_threshold = 0.055  # Порог для пинча
        self.drag_threshold = 0.06
        
        # Состояние
        self.enabled = False
        self.previous_x = self.screen_width // 2
        self.previous_y = self.screen_height // 2
        
        # Отслеживание жестов
        self.is_pinching = False
        self.is_dragging = False
        self.last_click_time = 0
        self.click_cooldown = 0.3  # Секунды между кликами
        
        # История для сглаживания
        self.position_history = deque(maxlen=5)
        
        # Отображение координат (зеркалирование для веб-камеры)
        self.mirror_x = False
        self.mirror_y = False
        
        # Состояние руки
        self.hand_detected = False
        self.current_gesture = "none"
        
        # Sensitivity - загружаем из calibration если есть
        self._sensitivity = 1.0
        if calibration_manager and calibration_manager.sensitivity:
            self._sensitivity = calibration_manager.sensitivity
        
    def enable(self):
        """Включить управление мышью."""
        self.enabled = True
        self.is_pinching = False
        self.is_dragging = False
        print("[GestureControl] Управление мышью ВКЛ")
        
    def disable(self):
        """Выключить управление мышью."""
        self.enabled = False
        self.is_dragging = False
        print("[GestureControl] Управление мышью ВЫКЛ")
        
    def toggle(self):
        """Переключить состояние."""
        if self.enabled:
            self.disable()
        else:
            self.enable()
        return self.enabled
    
    def process_hand(self, hand_landmarks, frame_width, frame_height, fingers_up):
        """
        Обработать данные руки и выполнить действия.
        
        Args:
            hand_landmarks: landmarks руки
            frame_width, frame_height: размеры кадра
            fingers_up: список пальцев [thumb, index, middle, ring, pinky]
            
        Returns:
            Текущий жест (строка)
        """
        if not self.enabled or not hand_landmarks:
            self.current_gesture = "none"
            return "disabled"
        
        self.hand_detected = True
        
        # Получаем координаты
        index_tip = hand_landmarks.landmark[8]
        thumb_tip = hand_landmarks.landmark[4]
        middle_tip = hand_landmarks.landmark[12]
        
        # Преобразуем в координаты экрана
        cursor_x = int(index_tip.x * frame_width)
        cursor_y = int(index_tip.y * frame_height)
        
        # Применяем зеркалирование
        if self.mirror_x:
            cursor_x = frame_width - cursor_x
        
        # Вычисляем размер руки для калибровки
        hand_size = self._calculate_hand_size(hand_landmarks)
        
        # Получаем чувствительность
        sensitivity = self._sensitivity
        if self.calibration:
            sensitivity = self.calibration.sensitivity
        
        # Простое преобразование координат: позиция на камере -> позиция на экране
        # Без дополнительных множителей, только чувствительность
        scale_factor = 0.8 * sensitivity
        
        screen_x = int(cursor_x * self.screen_width * scale_factor / frame_width)
        screen_y = int(cursor_y * self.screen_height * scale_factor / frame_height)
        
        # Ограничиваем в пределах экрана
        screen_x = max(0, min(self.screen_width - 1, screen_x))
        screen_y = max(0, min(self.screen_height - 1, screen_y))
        
        # Сглаживание - большее значение = более плавно
        smooth_factor = self.smoothing / 10.0
        smooth_x = int(self.previous_x + (screen_x - self.previous_x) * smooth_factor)
        smooth_y = int(self.previous_y + (screen_y - self.previous_y) * smooth_factor)
        
        # Вычисляем расстояния для жестов
        pinch_distance = self._calculate_distance(
            hand_landmarks.landmark[4],  # thumb
            hand_landmarks.landmark[8]    # index
        )
        
        drag_distance = self._calculate_distance(
            hand_landmarks.landmark[8],   # index
            hand_landmarks.landmark[12]   # middle
        )
        
        current_time = time.time()
        
        # Определяем жест
        gesture = self._determine_gesture(
            fingers_up, 
            pinch_distance, 
            drag_distance,
            hand_landmarks
        )
        
        self.current_gesture = gesture
        
        # Выполняем действие на основе жеста
        if gesture == "cursor":
            # Движение курсора
            pyautogui.moveTo(smooth_x, smooth_y, _pause=False)
            self.previous_x = smooth_x
            self.previous_y = smooth_y
            
        elif gesture == "left_click":
            # Левый клик (пинч)
            if not self.is_pinching and current_time - self.last_click_time > self.click_cooldown:
                pyautogui.click(button='left', _pause=False)
                self.last_click_time = current_time
                self.is_pinching = True
                
        elif gesture == "right_click":
            # Правый клик (указательный + средний вместе)
            if not self.is_pinching and current_time - self.last_click_time > self.click_cooldown:
                pyautogui.click(button='right', _pause=False)
                self.last_click_time = current_time
                self.is_pinching = True
                
        elif gesture == "drag":
            # Перетаскивание (кулак)
            if not self.is_dragging:
                pyautogui.mouseDown(_pause=False)
                self.is_dragging = True
            pyautogui.moveTo(smooth_x, smooth_y, _pause=False)
            self.previous_x = smooth_x
            self.previous_y = smooth_y
            
        elif gesture == "drag_release":
            # Отпускание перетаскивания
            if self.is_dragging:
                pyautogui.mouseUp(_pause=False)
                self.is_dragging = False
                    
        elif gesture == "pinch_release":
            # Освобождение от пинча
            self.is_pinching = False
            
        return gesture
    
    def _determine_gesture(self, fingers_up, pinch_distance, drag_distance, hand_landmarks):
        """
        Определить текущий жест.
        
        Fingers: [thumb, index, middle, ring, pinky]
        """
        thumb, index, middle, ring, pinky = fingers_up
        
        # Пинч для клика (большой + указательный близко)
        if pinch_distance < self.click_threshold:
            if self.is_dragging:
                return "drag_release"
            return "left_click"
        
        # Освобождение пинча
        if self.is_pinching and pinch_distance > self.click_threshold + 0.02:
            return "pinch_release"
        
        # Правый клик (указательный + средний близко)
        if drag_distance < self.drag_threshold and index and middle and not (thumb or ring or pinky):
            if not self.is_pinching:
                return "right_click"
        
        # Перетаскивание (кулак - все пальцы согнуты)
        if not index and not middle and not ring and not pinky and not thumb:
            return "drag"
        
        # Обычное движение (только указательный поднят)
        if index and not middle and not ring and not pinky and not self.is_dragging:
            return "cursor"
        
        return "none"
    
    def _calculate_distance(self, point1, point2):
        """Рассчитать нормированное расстояние между точками."""
        return math.sqrt(
            (point1.x - point2.x)**2 + 
            (point1.y - point2.y)**2
        )
    
    def _calculate_hand_size(self, hand_landmarks):
        """Рассчитать размер руки на основе расстояния между запястьем и средним пальцем."""
        wrist = hand_landmarks.landmark[0]
        middle_pip = hand_landmarks.landmark[10]
        
        return math.sqrt((wrist.x - middle_pip.x)**2 + (wrist.y - middle_pip.y)**2)
    
    def set_sensitivity(self, value):
        """Установить чувствительность (0.3 - 3.0)."""
        self._sensitivity = max(0.3, min(3.0, value))
        if self.calibration:
            self.calibration.set_sensitivity(self._sensitivity)
    
    def get_sensitivity(self):
        """Получить текущую чувствительность."""
        return self._sensitivity
    
    def get_status(self):
        """Получить текущий статус."""
        return {
            'enabled': self.enabled,
            'gesture': self.current_gesture,
            'hand_detected': self.hand_detected,
            'is_dragging': self.is_dragging,
            'sensitivity': self._sensitivity,
        }
    
    def reset(self):
        """Сбросить состояние."""
        self.is_pinching = False
        self.is_dragging = False
        self.hand_detected = False
        self.current_gesture = "none"
        self.position_history.clear()

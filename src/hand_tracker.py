import cv2
import numpy as np

mp = None
try:
    import mediapipe as mp
    mediapipe_available = True
except (ImportError, Exception) as e:
    print(f"MediaPipe not available: {e}")
    mediapipe_available = False


class HandTracker:
    def __init__(self, static_image_mode=False, max_hands=1,
                 min_detection_confidence=0.75, min_tracking_confidence=0.65,
                 model_complexity=0):
        """
        Args:
            model_complexity: 0 — лёгкая модель (быстро, ~5 ms/frame),
                              1 — тяжёлая (точнее, ~15 ms/frame).
                              Для real-time рекомендуется 0.
            min_detection_confidence: Порог первичного обнаружения. 0.75 даёт
                меньше ложных срабатываний, чем 0.5, но надёжнее, чем 0.9.
            min_tracking_confidence: Порог трекинга между кадрами. 0.65
                достаточно для стабильного удержания без лишних re-detect.
        """
        self.max_hands = max_hands
        self.initialized = False
        self._last_results = None   # кэш последних результатов для draw_cached()

        if not mediapipe_available:
            print("HandTracker: MediaPipe not available, gestures disabled")
            return

        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_hands,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_draw_styles = mp.solutions.drawing_styles
            self.initialized = True
        except Exception as e:
            print(f"HandTracker init error: {e}")

    def draw_cached(self, frame):
        """Нарисовать landmarks из последнего успешного кадра.

        Вызывать каждый кадр, чтобы не было мигания при кратких потерях руки.
        """
        if not self.initialized or self._last_results is None:
            return
        if self._last_results.multi_hand_landmarks:
            for hand_landmarks in self._last_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw_styles.get_default_hand_landmarks_style(),
                    self.mp_draw_styles.get_default_hand_connections_style()
                )

    def process_frame(self, frame, draw=True):
        """
        Обработать кадр и найти руки.

        Args:
            frame: Кадр в формате BGR
            draw: Рисовать ли landmarks

        Returns:
            processed_frame: Обработанный кадр
            results: Результаты детекции MediaPipe
        """
        if not self.initialized:
            return frame, None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        # Кэшируем только когда рука реально обнаружена
        if results and results.multi_hand_landmarks:
            self._last_results = results
        
        if draw and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw_styles.get_default_hand_landmarks_style(),
                    self.mp_draw_styles.get_default_hand_connections_style()
                )
        
        return frame, results
    
    def get_landmarks(self, results):
        """
        Извлечь landmarks рук из результатов.
        
        Returns:
            List of hand landmarks, or empty list if none found
        """
        if not results or not results.multi_hand_landmarks:
            return []
        
        return results.multi_hand_landmarks
    
    def get_hand_side(self, results, hand_idx=0):
        """
        Определить сторону руки (левая/правая).
        
        Args:
            results: Результаты MediaPipe
            hand_idx: Индекс руки (0 = первая обнаруженная)
            
        Returns:
            'left', 'right', или None
        """
        if not results or not results.multi_handedness:
            return None
        
        if hand_idx < len(results.multi_handedness):
            return results.multi_handedness[hand_idx].classification[0].label
        
        return None
    
    def is_hand_present(self, results):
        """
        Проверить, есть ли рука в кадре.
        """
        return results and results.multi_hand_landmarks is not None
    
    def get_index_finger_tip(self, landmarks, frame_width, frame_height):
        """
        Получить координаты кончика указательного пальца.
        
        Args:
            landmarks: Landmarks руки
            frame_width, frame_height: Размеры кадра
            
        Returns:
            (x, y) в пикселях
        """
        if not landmarks:
            return None
            
        index_tip = landmarks.landmark[8]
        x = int(index_tip.x * frame_width)
        y = int(index_tip.y * frame_height)
        return (x, y)
    
    def get_thumb_tip(self, landmarks, frame_width, frame_height):
        """
        Получить координаты кончика большого пальца.
        """
        if not landmarks:
            return None
            
        thumb_tip = landmarks.landmark[4]
        x = int(thumb_tip.x * frame_width)
        y = int(thumb_tip.y * frame_height)
        return (x, y)
    
    def get_landmark_coords(self, landmarks, landmark_idx, frame_width, frame_height):
        """
        Получить координаты любой точки руки.
        
        Args:
            landmarks: Landmarks руки
            landmark_idx: Индекс точки (0-20)
            frame_width, frame_height: Размеры кадра
            
        Returns:
            (x, y) в пикселях или None
        """
        if not landmarks or landmark_idx >= len(landmarks.landmark):
            return None
            
        point = landmarks.landmark[landmark_idx]
        x = int(point.x * frame_width)
        y = int(point.y * frame_height)
        return (x, y)
    
    def get_fingers_up(self, landmarks):
        """
        Определить, какие пальцы подняты.
        
        Returns:
            List of 5 booleans: [thumb, index, middle, ring, pinky]
            True = палец поднят
        """
        if not landmarks:
            return [False] * 5
        
        fingers = []
        lm = landmarks.landmark
        
        # Большой палец (по X координате)
        if lm[4].x > lm[3].x:
            fingers.append(True)
        else:
            fingers.append(False)
        
        # Остальные пальцы (по Y координате - ниже = выше на экране)
        for tip_idx, pip_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            if lm[tip_idx].y < lm[pip_idx].y:
                fingers.append(True)
            else:
                fingers.append(False)
        
        return fingers
    
    def get_distance(self, landmarks, idx1, idx2):
        """
        Рассчитать расстояние между двумя точками.
        
        Args:
            landmarks: Landmarks руки
            idx1, idx2: Индексы точек
            
        Returns:
            Normalizedное расстояние (0-1)
        """
        if not landmarks:
            return 1.0
        
        p1 = landmarks.landmark[idx1]
        p2 = landmarks.landmark[idx2]
        
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def get_palm_center(self, landmarks, frame_width, frame_height):
        """
        Получить центр ладони.
        """
        if not landmarks:
            return None
        
        # Центр ладони - среднее от точек 0, 5, 9, 13, 17
        palm_points = [0, 5, 9, 13, 17]
        center_x = sum(landmarks.landmark[i].x for i in palm_points) / len(palm_points)
        center_y = sum(landmarks.landmark[i].y for i in palm_points) / len(palm_points)
        
        return (int(center_x * frame_width), int(center_y * frame_height))


# Константы для удобства
FINGER_INDICES = {
    'thumb': 4,
    'index': 8,
    'middle': 12,
    'ring': 16,
    'pinky': 20,
    'thumb_mcp': 2,
    'index_pip': 6,
    'middle_pip': 10,
    'ring_pip': 14,
    'pinky_pip': 18,
}

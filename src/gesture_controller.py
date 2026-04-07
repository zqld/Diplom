"""
GestureController — управление мышью жестами руки.

Улучшения v2:
- Правильное маппирование: нормализованные координаты MediaPipe → экран
  через «активную зону» камеры (центральные 60–80 % кадра → весь экран).
- Единственный EMA-фильтр (нет двойного сглаживания).
- Dead-zone: игнорируем дрожание < 4 пикселей.
- Sensitivity управляет шириной активной зоны: выше → зона уже → движение резче.
- Клик-жест: кулак (все пальцы сжаты) → ЛКМ; V-жест (указ.+средний) → ПКМ.
- Скролл: три пальца вверх/вниз.
"""

import time
import math
import pyautogui
from collections import deque

pyautogui.FAILSAFE = False

try:
    import screeninfo
    _m = screeninfo.get_monitors()
    PRIMARY_SCREEN_WIDTH  = _m[0].width  if _m else pyautogui.size().width
    PRIMARY_SCREEN_HEIGHT = _m[0].height if _m else pyautogui.size().height
except Exception:
    PRIMARY_SCREEN_WIDTH  = pyautogui.size().width
    PRIMARY_SCREEN_HEIGHT = pyautogui.size().height


class GestureController:
    """Управление мышью жестами руки через нормализованные MediaPipe landmarks."""

    # ── Жесты ─────────────────────────────────────────────────────────────────
    GESTURE_CURSOR      = "cursor"       # указательный вверх → движение мыши
    GESTURE_LEFT_CLICK  = "left_click"   # кулак
    GESTURE_RIGHT_CLICK = "right_click"  # указательный + мизинец
    GESTURE_SCROLL_UP   = "scroll_up"    # 3 пальца: указат+средний+безымянный (без мизинца)
    GESTURE_SCROLL_DOWN = "scroll_down"  # 4 пальца: указат+средний+безымянный+мизинец
    GESTURE_NONE        = "none"

    def __init__(self, screen_width=None, screen_height=None, calibration_manager=None):
        self.screen_width  = screen_width  or PRIMARY_SCREEN_WIDTH
        self.screen_height = screen_height or PRIMARY_SCREEN_HEIGHT
        self.calibration   = calibration_manager

        # ── Состояние ────────────────────────────────────────────────────────
        self.enabled         = False
        self.current_gesture = self.GESTURE_NONE
        self.hand_detected   = False

        # Позиция курсора (сглаженная)
        self.prev_x = self.screen_width  // 2
        self.prev_y = self.screen_height // 2

        # ── Параметры сглаживания ─────────────────────────────────────────────
        # Адаптивный EMA: при малых движениях (шум/дрожание) alpha низкий →
        # плавно; при быстрых намеренных движениях alpha растёт → отзывчиво.
        self._alpha_slow = 0.12   # alpha при скорости < _speed_slow px/тик
        self._alpha_fast = 0.45   # alpha при скорости > _speed_fast px/тик
        self._speed_slow = 20.0   # px/тик — нижняя граница «быстрого» движения
        self._speed_fast = 120.0  # px/тик — верхняя граница

        # Dead-zone: движения меньше этого порога игнорируются (шум пикселей).
        self._dead_zone  = 6      # px

        # Защита от выбросов: если целевая позиция прыгает дальше порога от
        # текущей сглаженной — скорее всего ошибка детекции.
        # _outlier_consecutive считает подряд идущие выбросы:
        # 1-2 кадра → одиночная ошибка, игнорируем (цель → prev).
        # 3+ кадров → скорее всего смена зоны калибровки; телепортируем prev к цели.
        self._outlier_threshold   = 280  # px
        self._outlier_consecutive = 0    # счётчик подряд идущих выбросов

        # ── Активная зона камеры ─────────────────────────────────────────────
        # base_margin: отступ от края кадра (нормализованный 0..0.5).
        # Зона [margin, 1-margin] × [margin, 1-margin] → весь экран.
        # sensitivity=1 → margin≈0.15 (центр. 70 % кадра);
        # sensitivity=2 → margin≈0.08 (центр. 84 %); sensitivity=0.5 → margin≈0.25.
        self._base_margin = 0.15

        # ── Чувствительность ─────────────────────────────────────────────────
        self._sensitivity = 1.0
        if calibration_manager and calibration_manager.sensitivity:
            self._sensitivity = calibration_manager.sensitivity

        # ── Стабилизация жестов ───────────────────────────────────────────────
        # Клик выполняется только после N одинаковых кадров подряд.
        self._gesture_buf    = deque(maxlen=6)
        self._stable_frames  = 4   # нужно 4 кадра одинакового жеста для кликов
        self._last_confirmed = self.GESTURE_NONE

        # ── Кулдауны ─────────────────────────────────────────────────────────
        self._last_click_time  = 0.0
        self._click_cooldown   = 0.45   # сек между кликами
        self._last_scroll_time = 0.0
        self._scroll_cooldown  = 0.12   # сек между тиками скролла

    # ── Публичный API ─────────────────────────────────────────────────────────
    def enable(self):
        self.enabled = True
        self.prev_x  = self.screen_width  // 2
        self.prev_y  = self.screen_height // 2
        self._gesture_buf.clear()
        self._last_confirmed = self.GESTURE_NONE
        self._outlier_consecutive = 0

    def disable(self):
        self.enabled = False
        self.hand_detected = False

    def toggle(self) -> bool:
        if self.enabled:
            self.disable()
        else:
            self.enable()
        return self.enabled

    def set_sensitivity(self, value: float):
        self._sensitivity = max(0.3, min(3.0, float(value)))
        if self.calibration:
            self.calibration.set_sensitivity(self._sensitivity)

    def get_sensitivity(self) -> float:
        return self._sensitivity

    def get_status(self) -> dict:
        return {
            'enabled':      self.enabled,
            'gesture':      self.current_gesture,
            'hand_detected': self.hand_detected,
            'sensitivity':  self._sensitivity,
        }

    def reset(self):
        self._gesture_buf.clear()
        self._last_confirmed = self.GESTURE_NONE
        self.current_gesture = self.GESTURE_NONE
        self.hand_detected   = False
        self._outlier_consecutive = 0

    # ── Основной метод обработки ──────────────────────────────────────────────
    def process_hand(self, hand_landmarks, frame_width, frame_height,
                     fingers_up, hand_x=None, hand_y=None, hand_size=None):
        """
        Обработать кадр с рукой.

        Args:
            hand_landmarks : MediaPipe NormalizedLandmarkList (21 точка)
            frame_width/height: размеры кадра (используются только для совместимости)
            fingers_up     : [thumb, index, middle, ring, pinky] — bool-список
            hand_x/y/size  : необязательные доп. данные (игнорируются в v2)

        Returns:
            str — текущий подтверждённый жест
        """
        if not self.enabled or not hand_landmarks:
            self.current_gesture = self.GESTURE_NONE
            return "disabled"

        self.hand_detected = True
        now = time.time()

        # ── 1. Координаты из нормализованных landmarks ─────────────────────
        # Используем landmark[5] (index MCP) — стабильнее, чем кончик пальца.
        lm = hand_landmarks.landmark[5]
        lm_x = lm.x   # 0..1
        lm_y = lm.y   # 0..1

        # ── 2. Маппирование активной зоны → экран ──────────────────────────
        # Если зона откалибрована — используем личные границы пользователя.
        # Иначе: вычисляем симметричный margin из sensitivity.
        if (self.calibration and
                getattr(self.calibration, 'gesture_zone', {}).get('calibrated')):
            gz = self.calibration.gesture_zone
            zone_x0 = gz['x_min']
            zone_y0 = gz['y_min']
            zone_w  = max(0.01, gz['x_max'] - gz['x_min'])
            zone_h  = max(0.01, gz['y_max'] - gz['y_min'])
        else:
            margin  = max(0.04, self._base_margin / self._sensitivity)
            zone_x0 = margin
            zone_y0 = margin
            zone_w  = max(0.01, 1.0 - 2 * margin)
            zone_h  = max(0.01, 1.0 - 2 * margin)

        norm_x = max(0.0, min(1.0, (lm_x - zone_x0) / zone_w))
        norm_y = max(0.0, min(1.0, (lm_y - zone_y0) / zone_h))

        target_x = norm_x * self.screen_width
        target_y = norm_y * self.screen_height

        # ── 3. Защита от выбросов (outlier rejection) ──────────────────────
        # Если целевая позиция прыгает дальше порога — два случая:
        #   • 1–2 кадра подряд → одиночная ошибка детекции, держим prev.
        #   • 3+ кадров подряд → смена зоны калибровки / реальное перемещение;
        #     телепортируем prev к цели, чтобы не заморозить курсор навсегда.
        raw_dx = target_x - self.prev_x
        raw_dy = target_y - self.prev_y
        if math.sqrt(raw_dx * raw_dx + raw_dy * raw_dy) > self._outlier_threshold:
            self._outlier_consecutive += 1
            if self._outlier_consecutive >= 3:
                # Принимаем прыжок — обновляем prev без движения мыши
                self.prev_x = target_x
                self.prev_y = target_y
            else:
                target_x = self.prev_x
                target_y = self.prev_y
        else:
            self._outlier_consecutive = 0

        # ── 4. Адаптивный EMA-фильтр ───────────────────────────────────────
        # Скорость «сырого» смещения определяет alpha:
        #   - малая скорость (шум/дрожание) → низкий alpha → плавно
        #   - высокая скорость (намеренный жест) → высокий alpha → отзывчиво
        raw_speed = math.sqrt(raw_dx * raw_dx + raw_dy * raw_dy)
        t = max(0.0, min(1.0,
                         (raw_speed - self._speed_slow) /
                         max(1.0, self._speed_fast - self._speed_slow)))
        # Sensitivity сдвигает оба предела вверх при высокой чувствительности
        s_boost = (self._sensitivity - 1.0) * 0.04
        alpha = (self._alpha_slow + s_boost) + t * (self._alpha_fast - self._alpha_slow)
        alpha = max(0.05, min(0.60, alpha))

        new_x = self.prev_x + (target_x - self.prev_x) * alpha
        new_y = self.prev_y + (target_y - self.prev_y) * alpha

        # ── 5. Dead-zone ───────────────────────────────────────────────────
        dx = abs(new_x - self.prev_x)
        dy = abs(new_y - self.prev_y)
        if dx < self._dead_zone and dy < self._dead_zone:
            new_x = self.prev_x
            new_y = self.prev_y

        # Клипируем к экрану
        new_x = max(0, min(self.screen_width  - 1, new_x))
        new_y = max(0, min(self.screen_height - 1, new_y))
        final_x = int(new_x)
        final_y = int(new_y)

        # ── 6. Определение жеста ───────────────────────────────────────────
        raw_gesture = self._classify_gesture(fingers_up, hand_landmarks)
        self._gesture_buf.append(raw_gesture)
        confirmed = self._confirm_gesture()
        self.current_gesture = confirmed

        # ── 7. Выполнение действия ─────────────────────────────────────────
        if confirmed == self.GESTURE_CURSOR:
            pyautogui.moveTo(final_x, final_y, _pause=False)
            self.prev_x = new_x
            self.prev_y = new_y

        elif confirmed == self.GESTURE_LEFT_CLICK:
            if now - self._last_click_time > self._click_cooldown:
                pyautogui.click(button='left', _pause=False)
                self._last_click_time = now

        elif confirmed == self.GESTURE_RIGHT_CLICK:
            if now - self._last_click_time > self._click_cooldown:
                pyautogui.click(button='right', _pause=False)
                self._last_click_time = now

        elif confirmed == self.GESTURE_SCROLL_UP:
            if now - self._last_scroll_time > self._scroll_cooldown:
                pyautogui.scroll(3, _pause=False)
                self._last_scroll_time = now
            # Движение мыши тоже обновляем
            self.prev_x = new_x
            self.prev_y = new_y

        elif confirmed == self.GESTURE_SCROLL_DOWN:
            if now - self._last_scroll_time > self._scroll_cooldown:
                pyautogui.scroll(-3, _pause=False)
                self._last_scroll_time = now
            self.prev_x = new_x
            self.prev_y = new_y

        else:
            # none — просто обновляем позицию для плавного старта
            self.prev_x = new_x
            self.prev_y = new_y

        return confirmed

    # ── Внутренние методы ─────────────────────────────────────────────────────
    def _classify_gesture(self, fingers_up, hand_landmarks) -> str:
        """
        Классифицировать жест на основе подъёма пальцев.

        fingers_up = [thumb, index, middle, ring, pinky]
        """
        thumb, index, middle, ring, pinky = fingers_up

        # Кулак (все согнуты) → ЛКМ
        if not index and not middle and not ring and not pinky:
            return self.GESTURE_LEFT_CLICK

        # Указательный + мизинец (без среднего и безымянного) → ПКМ
        if index and pinky and not middle and not ring:
            return self.GESTURE_RIGHT_CLICK

        # 4 пальца (указат+средний+безымянный+мизинец, без большого) → прокрутка вниз
        # Проверяем ДО трёх пальцев, т.к. это надмножество
        if index and middle and ring and pinky and not thumb:
            return self.GESTURE_SCROLL_DOWN

        # 3 пальца (указат+средний+безымянный, без мизинца) → прокрутка вверх
        if index and middle and ring and not pinky:
            return self.GESTURE_SCROLL_UP

        # Только указательный → движение курсора
        if index and not middle and not ring and not pinky:
            return self.GESTURE_CURSOR

        return self.GESTURE_NONE

    def _confirm_gesture(self) -> str:
        """
        Вернуть подтверждённый жест.
        - Клики требуют `_stable_frames` одинаковых кадров.
        - Движение и скролл — мгновенны (чтобы не было задержки).
        """
        if not self._gesture_buf:
            return self.GESTURE_NONE

        latest = self._gesture_buf[-1]

        # Движение и скролл — подтверждаем сразу
        if latest in (self.GESTURE_CURSOR, self.GESTURE_SCROLL_UP,
                      self.GESTURE_SCROLL_DOWN, self.GESTURE_NONE):
            self._last_confirmed = latest
            return latest

        # Клики: нужно N одинаковых кадров подряд
        recent = list(self._gesture_buf)[-self._stable_frames:]
        if len(recent) == self._stable_frames and len(set(recent)) == 1:
            confirmed = recent[0]
        else:
            confirmed = self._last_confirmed  # держим предыдущий подтверждённый

        self._last_confirmed = confirmed
        return confirmed

    # ── Совместимость со старым кодом ────────────────────────────────────────
    def _calculate_distance(self, p1, p2) -> float:
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def _calculate_hand_size(self, hand_landmarks) -> float:
        return self._calculate_distance(
            hand_landmarks.landmark[0],
            hand_landmarks.landmark[10]
        )

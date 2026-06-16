import pytest
import math


class FakeCalibration:
    """Имитация CalibrationManager для тестов жестов."""
    def __init__(self, sensitivity=1.0, zone_calibrated=False,
                 zone=(0.15, 0.85, 0.15, 0.85)):
        self.sensitivity = sensitivity
        self.gesture_zone = {
            "calibrated": zone_calibrated,
            "x_min": zone[0],
            "x_max": zone[1],
            "y_min": zone[2],
            "y_max": zone[3],
        }

    def set_sensitivity(self, value):
        self.sensitivity = value


@pytest.fixture
def gc_no_calib():
    """GestureController без калибровки зоны, sensitivity=1.0."""
    from src.gesture_controller import GestureController
    cm = FakeCalibration(sensitivity=1.0, zone_calibrated=False)
    return GestureController(screen_width=1920, screen_height=1080,
                             calibration_manager=cm)


@pytest.fixture
def gc_calib_sens1():
    """GestureController с калиброванной зоной [0.20, 0.75], sensitivity=1.0."""
    from src.gesture_controller import GestureController
    cm = FakeCalibration(sensitivity=1.0, zone_calibrated=True,
                         zone=(0.20, 0.75, 0.10, 0.65))
    return GestureController(screen_width=1920, screen_height=1080,
                             calibration_manager=cm)


@pytest.fixture
def gc_calib_sens3():
    """GestureController с калиброванной зоной [0.20, 0.75], sensitivity=3.0."""
    from src.gesture_controller import GestureController
    cm = FakeCalibration(sensitivity=3.0, zone_calibrated=True,
                         zone=(0.20, 0.75, 0.10, 0.65))
    return GestureController(screen_width=1920, screen_height=1080,
                             calibration_manager=cm)


class FakeLandmark:
    """Имитация MediaPipe landmark."""
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class FakeHand:
    """Имитация MediaPipe hand_landmarks с доступом через .landmark[5]."""
    def __init__(self, x, y):
        self._lm = {5: FakeLandmark(x, y)}

    @property
    def landmark(self):
        return self._lm


class TestGestureSensitivity:
    def _process(self, gc, hand_x, hand_y, times=10):
        """Вспомогательный метод: вызвать process_hand times раз для стабилизации
        (EMA + outlier rejection)."""
        gc.enable()
        for _ in range(times):
            gc.process_hand(
                FakeHand(hand_x, hand_y),
                frame_width=640, frame_height=480,
                fingers_up=[False, True, False, False, False],
            )

    def test_non_calibrated_margin_at_sens1(self, gc_no_calib):
        """Без калибровки, sensitivity=1: margin=0.15, zone[0.15, 0.85]."""
        self._process(gc_no_calib, 0.50, 0.50)
        assert abs(gc_no_calib.prev_x - 960) < 50
        assert abs(gc_no_calib.prev_y - 540) < 50

    def test_calibrated_zone_at_sens1(self, gc_calib_sens1):
        """Калиброванная зона [0.20,0.75], sens=1: границы не меняются."""
        self._process(gc_calib_sens1, 0.50, 0.50)
        assert abs(gc_calib_sens1.prev_x - 1047) < 50

    def test_calibrated_zone_gain_at_sens3(self, gc_calib_sens3):
        """Калиброванная зона, sens=3: gain растягивает от центра.
        landmark 0.50 → norm_x=0.545 → target_base=1047,
        gain=3: 960 + (1047-960)*3 ≈ 1222."""
        self._process(gc_calib_sens3, 0.50, 0.50)
        assert abs(gc_calib_sens3.prev_x - 1222) < 50

    def test_calibrated_zone_same_hand_less_cursor_at_sens3(self):
        """Одна и та же позиция руки (0.35, левее центра) даёт МЕНЬШЕ курсора
        при sens=3 vs sens=1 из-за gain-растяжения от центра.
        sens=1 → 960 + (524-960)*1 = 524,
        sens=3 → 960 + (524-960)*3 = -349 → clamp to 0."""
        from src.gesture_controller import GestureController

        cm1 = FakeCalibration(sensitivity=1.0, zone_calibrated=True,
                              zone=(0.20, 0.75, 0.10, 0.65))
        gc1 = GestureController(screen_width=1920, screen_height=1080,
                                calibration_manager=cm1)

        cm3 = FakeCalibration(sensitivity=3.0, zone_calibrated=True,
                              zone=(0.20, 0.75, 0.10, 0.65))
        gc3 = GestureController(screen_width=1920, screen_height=1080,
                                calibration_manager=cm3)

        self._process(gc1, 0.35, 0.50)
        self._process(gc3, 0.35, 0.50)

        # gc1 ≈ 524, gc3 ≈ 0 (clamped)
        assert gc3.prev_x < gc1.prev_x
        assert gc3.prev_x < 100

    def test_set_sensitivity_updates_gain(self):
        """Динамическое изменение sensitivity через set_sensitivity."""
        from src.gesture_controller import GestureController
        cm = FakeCalibration(sensitivity=2.0, zone_calibrated=True,
                             zone=(0.20, 0.75, 0.10, 0.65))
        gc = GestureController(screen_width=1920, screen_height=1080,
                               calibration_manager=cm)
        assert gc._sensitivity == 2.0
        gc.set_sensitivity(1.0)
        assert gc._sensitivity == 1.0

    def test_calibrated_zone_gain_below_one(self):
        """При sensitivity < 1 gain тормозит курсор (держит ближе к центру).
        landmark 0.65 (правее центра):
        sens=1 → target_base=1571,
        sens=0.5 → 960 + (1571-960)*0.5 = 1265 (ближе к центру)."""
        from src.gesture_controller import GestureController

        cm1 = FakeCalibration(sensitivity=1.0, zone_calibrated=True,
                              zone=(0.20, 0.75, 0.10, 0.65))
        gc1 = GestureController(screen_width=1920, screen_height=1080,
                                calibration_manager=cm1)

        cm05 = FakeCalibration(sensitivity=0.5, zone_calibrated=True,
                               zone=(0.20, 0.75, 0.10, 0.65))
        gc05 = GestureController(screen_width=1920, screen_height=1080,
                                 calibration_manager=cm05)

        self._process(gc1, 0.65, 0.50)
        self._process(gc05, 0.65, 0.50)

        assert gc1.prev_x > 1500  # sens=1 → near 1571
        assert abs(gc05.prev_x - 1265) < 50  # sens=0.5 → clamped nearer center

    def test_non_calibrated_zone_changes_with_sensitivity(self):
        """Без калибровки: margin = base_margin/sensitivity.
        При sens=1 landmark 0.20 у края зоны (margin=0.15), при sens=3 (margin=0.05)
        → target_x при sens=1 < target_x при sens=3."""
        from src.gesture_controller import GestureController

        cm1 = FakeCalibration(sensitivity=1.0, zone_calibrated=False)
        gc1 = GestureController(screen_width=1920, screen_height=1080,
                                calibration_manager=cm1)

        cm3 = FakeCalibration(sensitivity=3.0, zone_calibrated=False)
        gc3 = GestureController(screen_width=1920, screen_height=1080,
                                calibration_manager=cm3)

        self._process(gc1, 0.20, 0.50)
        self._process(gc3, 0.20, 0.50)

        # sens=1: margin=0.15 → norm_x≈0 → target≈0
        # sens=3: margin=0.05 → norm_x=0.167 → target≈320
        assert gc3.prev_x > gc1.prev_x + 100

    def test_cursor_frozen_during_left_click(self):
        """При сыром LEFT_CLICK prev_x/prev_y не меняются, даже если рука
        смещается — клик должен быть точным."""
        from unittest.mock import patch
        from src.gesture_controller import GestureController

        cm = FakeCalibration(sensitivity=1.0, zone_calibrated=False)
        gc = GestureController(screen_width=1920, screen_height=1080,
                               calibration_manager=cm)
        gc.enable()

        # Шаг 1: двигаем курсор к центру (CURSOR)
        for _ in range(10):
            gc.process_hand(
                FakeHand(0.50, 0.50),
                frame_width=640, frame_height=480,
                fingers_up=[False, True, False, False, False],
            )
        cursor_x, cursor_y = gc.prev_x, gc.prev_y
        assert abs(cursor_x - 960) < 50

        # Шаг 2: рука смещается, raw_gesture = LEFT_CLICK (кулак)
        with patch('pyautogui.click'):
            for _ in range(6):
                gc.process_hand(
                    FakeHand(0.55, 0.45),  # смещённая позиция
                    frame_width=640, frame_height=480,
                    fingers_up=[False, False, False, False, False],
                )

        # prev_x/prev_y заморожены — не изменились
        assert gc.prev_x == cursor_x
        assert gc.prev_y == cursor_y

    def test_cursor_frozen_during_right_click(self):
        """При сыром RIGHT_CLICK prev_x/prev_y заморожены."""
        from unittest.mock import patch
        from src.gesture_controller import GestureController

        cm = FakeCalibration(sensitivity=1.0, zone_calibrated=False)
        gc = GestureController(screen_width=1920, screen_height=1080,
                               calibration_manager=cm)
        gc.enable()

        for _ in range(10):
            gc.process_hand(
                FakeHand(0.50, 0.50),
                frame_width=640, frame_height=480,
                fingers_up=[False, True, False, False, False],
            )
        cursor_x, cursor_y = gc.prev_x, gc.prev_y

        # RIGHT_CLICK: index + pinky up
        with patch('pyautogui.click'):
            for _ in range(6):
                gc.process_hand(
                    FakeHand(0.55, 0.45),
                    frame_width=640, frame_height=480,
                    fingers_up=[False, True, False, False, True],
                )

        assert gc.prev_x == cursor_x
        assert gc.prev_y == cursor_y

    def test_cursor_unfrozen_after_click_release(self):
        """После отпускания клика (raw_gesture → CURSOR) курсор снова
        следует за рукой."""
        from unittest.mock import patch
        from src.gesture_controller import GestureController

        cm = FakeCalibration(sensitivity=1.0, zone_calibrated=False)
        gc = GestureController(screen_width=1920, screen_height=1080,
                               calibration_manager=cm)
        gc.enable()

        # CURSOR в центре
        for _ in range(10):
            gc.process_hand(
                FakeHand(0.50, 0.50),
                frame_width=640, frame_height=480,
                fingers_up=[False, True, False, False, False],
            )
        cursor_x, cursor_y = gc.prev_x, gc.prev_y

        # LEFT_CLICK
        with patch('pyautogui.click'):
            for _ in range(5):
                gc.process_hand(
                    FakeHand(0.55, 0.45),
                    frame_width=640, frame_height=480,
                    fingers_up=[False, False, False, False, False],
                )

        assert gc.prev_x == cursor_x  # заморожено

        # Рука раскрывается — CURSOR снова
        for _ in range(10):
            gc.process_hand(
                FakeHand(0.50, 0.50),
                frame_width=640, frame_height=480,
                fingers_up=[False, True, False, False, False],
            )

        # prev_x/prev_y теперь обновляются (следуют за рукой)
        assert abs(gc.prev_x - 960) < 50

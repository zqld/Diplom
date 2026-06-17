import os
import json
import tempfile
import pytest
import numpy as np
from collections import deque

# ── Fix 4: settings save/load ──────────────────────────────────────

class FakeNotificationManager:
    """Копия логики save/load из NotificationManager для теста без Qt."""
    SETTINGS_PATH = os.path.join(tempfile.gettempdir(), "_test_settings.json")

    def __init__(self):
        self.settings = {
            "work_limit_minutes": 45,
            "posture_window_minutes": 3,
            "posture_bad_percent": 30,
            "yawn_limit": 4,
            "yawn_window_minutes": 10,
        }
        self.load_settings()

    def load_settings(self):
        try:
            if os.path.exists(self.SETTINGS_PATH):
                with open(self.SETTINGS_PATH, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                for k in self.settings:
                    if k in saved:
                        self.settings[k] = saved[k]
        except Exception:
            pass

    def save_settings(self):
        try:
            os.makedirs(os.path.dirname(self.SETTINGS_PATH), exist_ok=True)
            with open(self.SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def update_settings(self, new_settings):
        self.settings.update(new_settings)
        self.save_settings()

    def cleanup(self):
        if os.path.exists(self.SETTINGS_PATH):
            os.unlink(self.SETTINGS_PATH)


class TestSettingsPersistence:
    def test_save_and_load_roundtrip(self):
        mgr = FakeNotificationManager()
        mgr.update_settings({"yawn_limit": 8, "posture_bad_percent": 60})
        mgr2 = FakeNotificationManager()
        assert mgr2.settings["yawn_limit"] == 8
        assert mgr2.settings["posture_bad_percent"] == 60
        mgr2.cleanup()

    def test_load_unknown_key_ignored(self):
        mgr = FakeNotificationManager()
        with open(mgr.SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump({"nonexistent": 999, "yawn_limit": 3}, f)
        mgr.load_settings()
        assert mgr.settings["yawn_limit"] == 3
        assert "nonexistent" not in mgr.settings
        mgr.cleanup()

    def test_defaults_used_when_no_file(self):
        mgr = FakeNotificationManager()
        if os.path.exists(mgr.SETTINGS_PATH):
            os.unlink(mgr.SETTINGS_PATH)
        mgr.load_settings()
        assert mgr.settings["yawn_limit"] == 4
        assert mgr.settings["work_limit_minutes"] == 45


# ── Fix 1: FatigueAnalyzer with calibration ────────────────────────

class FakeCalibration:
    """Имитация CalibrationManager.face_calibration."""
    def __init__(self, ear=0.30, mar=0.15, calibrated=True):
        self.face_calibration = {
            "calibrated": calibrated,
            "baseline_ear": ear,
            "baseline_mar": mar,
        }


@pytest.fixture
def fa_default():
    """FatigueAnalyzer без калибровки."""
    from src.fatigue_analyzer import FatigueAnalyzer
    return FatigueAnalyzer(window_size_seconds=30)


@pytest.fixture
def fa_calibrated():
    """FatigueAnalyzer с калибровкой (baseline_ear=0.38, baseline_mar=0.18)."""
    from src.fatigue_analyzer import FatigueAnalyzer
    cm = FakeCalibration(ear=0.38, mar=0.18)
    return FatigueAnalyzer(window_size_seconds=30, calibration_manager=cm)


class TestFatigueAnalyzerCalibration:
    def test_default_thresholds(self, fa_default):
        assert fa_default._ear_open == pytest.approx(0.25, rel=0.01)
        assert fa_default._ear_closed == pytest.approx(0.18, rel=0.01)
        assert fa_default._ear_critical == pytest.approx(0.22, rel=0.01)
        assert fa_default._mar_max_physio == pytest.approx(0.70, rel=0.01)

    def test_calibrated_thresholds_shifted_up(self, fa_calibrated):
        # baseline_ear=0.38 → ratio=0.38/0.30 ≈ 1.267
        # ear_open = 0.28 * 1.267 ≈ 0.355
        assert fa_calibrated._ear_open > 0.30
        assert fa_calibrated._ear_open < 0.45
        # ear_closed = 0.18 * 1.267 ≈ 0.228
        assert fa_calibrated._ear_closed > 0.20
        # mar_max = 0.70 * (0.18/0.15) = 0.84
        assert fa_calibrated._mar_max_physio == pytest.approx(0.84, rel=0.01)

    def test_calibrated_blink_detection(self, fa_calibrated):
        """С повышенными порогами blink детектируется при более высоком EAR."""
        # У некалиброванного: blink при ear < 0.22 после ear > 0.28
        # У калиброванного (ear_open≈0.355, ear_critical≈0.279):
        # blink при ear < ear_critical после ear > ear_open
        fa_calibrated._detect_blink(0.38, 100.0)  # open
        fa_calibrated._detect_blink(0.26, 100.1)  # ниже critical → blink
        assert fa_calibrated.blink_count == 1

    def test_default_mar_score(self, fa_default):
        fa_default.mar_history.extend([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        score = fa_default._get_mar_score()
        # max=0.6, minmax over [0, 0.70] → 0.6/0.7 ≈ 0.857
        assert score == pytest.approx(0.857, rel=0.01)

    def test_calibrated_mar_score(self, fa_calibrated):
        """С mar_max=0.84, score для max=0.6 будет ниже."""
        fa_calibrated.mar_history.extend([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        score = fa_calibrated._get_mar_score()
        # max=0.6, minmax over [0, 0.84] → 0.6/0.84 ≈ 0.714
        assert score == pytest.approx(0.714, rel=0.01)


# ── Fix 2: PostureAnalyzer custom bad_threshold ────────────────────

class TestPostureAnalyzerThreshold:
    def test_default_bad_threshold(self):
        from src.posture_analyzer import PostureAnalyzer
        pa = PostureAnalyzer()
        assert pa._bad_threshold == 60

    def test_custom_bad_threshold(self):
        from src.posture_analyzer import PostureAnalyzer
        pa = PostureAnalyzer(bad_threshold=30)
        assert pa._bad_threshold == 30

    def test_lower_threshold_triggers_bad_earlier(self):
        from src.posture_analyzer import PostureAnalyzer
        pa_sensitive = PostureAnalyzer(bad_threshold=30)
        pa_default = PostureAnalyzer(bad_threshold=60)
        # head_tilt=20° → tilt_bucket=20, head_forward=0.12 → forward_bucket=35
        # score = 20 + 35 = 55
        _, bad_sensitive = pa_sensitive._analyze_posture_state(20, 0.12, 0.05)
        _, bad_default = pa_default._analyze_posture_state(20, 0.12, 0.05)
        assert bad_sensitive is True   # 55 >= 30
        assert bad_default is False    # 55 < 60

    def test_higher_threshold_tolerates_more(self):
        from src.posture_analyzer import PostureAnalyzer
        pa = PostureAnalyzer(bad_threshold=90)
        # head_tilt=25° → 35, head_forward=0.18 → 50, face_pos=0.10→0
        # score = 35 + 50 = 85
        _, bad = pa._analyze_posture_state(25, 0.18, 0.05)
        assert bad is False  # 85 < 90


# ── Fix 2: PostureProcessor._compute_bad_threshold ─────────────────

class TestPostureProcessorThreshold:
    def test_default_returns_60(self):
        from src.processors.posture_processor import PostureProcessor
        assert PostureProcessor._compute_bad_threshold(None) == 60

    @pytest.mark.parametrize("pct,expected", [
        (100, 30),   # max sensitivity → threshold=30
        (75,  40),   # 75/50=1.5, 60/1.5=40
        (50,  60),   # default → 60
        (25,  120),  # 25/50=0.5, 60/0.5=120
        (5,   600),  # min sensitivity → 600
    ])
    def test_compute_threshold(self, pct, expected):
        from src.processors.posture_processor import PostureProcessor
        assert PostureProcessor._compute_bad_threshold(pct) == expected

    def test_min_floor_15(self):
        from src.processors.posture_processor import PostureProcessor
        # 100% → factor=2 → 60/2=30, max(15,30)=30
        assert PostureProcessor._compute_bad_threshold(100) == 30
        # extreme case
        assert PostureProcessor._compute_bad_threshold(200) >= 15


# ── Fix 3: FatigueProcessor yawn_limit cooldown ────────────────────

class TestFatigueProcessorYawnLimit:
    def test_process_with_yawn_limit(self):
        from src.processors.fatigue_processor import FatigueProcessor
        fp = FatigueProcessor()
        fp.set_yawn_limit(2)

        # Первый зевок (mar=0.7 > 0.6) — должен сработать
        result = fp.process(0.30, 0.70, 5.0, "Neutral", 100.0)
        assert result['event'] == "Зевок"

        # Второй сразу — cooldown=4.0 (2*2), ещё не прошёл
        result = fp.process(0.30, 0.70, 5.0, "Neutral", 101.0)
        assert result['event'] is None  # cooldown активен

        # Через 5 сек — должно сработать снова
        result = fp.process(0.30, 0.70, 5.0, "Neutral", 105.0)
        assert result['event'] == "Зевок"  # cooldown прошёл

    def test_default_yawn_cooldown(self):
        """Без set_yawn_limit — стандартный cooldown 2 сек."""
        from src.processors.fatigue_processor import FatigueProcessor
        fp = FatigueProcessor()

        result = fp.process(0.30, 0.70, 5.0, "Neutral", 100.0)
        assert result['event'] == "Зевок"

        # Через 1 сек — cooldown=2 ещё не прошёл
        result = fp.process(0.30, 0.70, 5.0, "Neutral", 101.0)
        assert result['event'] is None

        # Через 3 сек — прошёл
        result = fp.process(0.30, 0.70, 5.0, "Neutral", 103.0)
        assert result['event'] == "Зевок"


# ── Fix 3: FatigueProcessor calibration_manager ────────────────────

class TestFatigueProcessorCalibration:
    def test_calibration_passed_to_analyzer(self):
        from src.processors.fatigue_processor import FatigueProcessor
        cm = FakeCalibration(ear=0.38, mar=0.18)
        fp = FatigueProcessor(calibration_manager=cm)
        # Проверяем, что пороги в анализаторе скорректированы
        assert fp.analyzer._ear_open > 0.30
        assert fp.analyzer._mar_max_physio == pytest.approx(0.84, rel=0.01)

    def test_set_calibration_updates_analyzer(self):
        from src.processors.fatigue_processor import FatigueProcessor
        fp = FatigueProcessor()
        # По умолчанию — базовые пороги
        assert fp.analyzer._ear_open == pytest.approx(0.25, rel=0.01)
        # После установки калибровки — пороги меняются
        cm = FakeCalibration(ear=0.38, mar=0.18)
        fp.set_calibration_manager(cm)
        assert fp.analyzer._ear_open > 0.30

"""
Fatigue analyzer with mathematically normalised component scores.

КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ (v2):
  • EAR score — НЕЛИНЕЙНЫЙ (экспоненциальное падение при EAR < порога).
    При EAR=0.18 (закрытые глаза) score → 1.0 → attention_score → минимум.
  • Убрано избыточное сглаживание rolling average на 10 кадров.
    Текущий EAR имеет вес 0.85, история — 0.15.
  • Мгновенный штраф к attention_score если EAR < 0.22.
  • Все компоненты нормализованы строго к [0, 1].

Normalisation methods:
  • EAR  — Экспоненциальная функция с порогом (было: Min-Max)
  • MAR  — Min-Max  over physiological range [0.0, 0.70]
  • Blink rate — Sigmoid  (natural saturation at low / high extremes)
  • Emotion — Categorical lookup  (already bounded 0–1)
  • Trend  — Min-Max  over expected EAR change [0, 0.10]
"""

import numpy as np
from collections import deque
import time

# ── Normalisation constants ──────────────────────────────────────
EAR_MIN_PHYSIO = 0.12   # глаза плотно закрыты
EAR_MAX_PHYSIO = 0.40   # глаза широко открыты
EAR_CRITICAL_THRESHOLD = 0.22  # ниже — мгновенный штраф
EAR_CLOSED_THRESHOLD = 0.18    # ниже — глаза закрыты (сон/микросон)
MAR_MIN_PHYSIO = 0.0
MAR_MAX_PHYSIO = 0.70   # extreme yawn
BLINK_SIGMOID_CENTER = 25.0   # blinks / min where score ≈ 0.5
BLINK_SIGMOID_SLOPE = 0.15
TREND_MIN = 0.0
TREND_MAX = 0.10

# ── Exponential EAR scoring ────────────────────────────────────
# При EAR >= 0.28 → score ≈ 0 (бодр)
# При EAR = 0.22 → score ≈ 0.5
# При EAR <= 0.18 → score → 1.0 (спит)
_EAR_EXPONENTIAL_RATE = 0.35  # крутизна экспоненты


def _min_max(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi] and normalise to [0, 1]."""
    if hi <= lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def _sigmoid(value: float, center: float, slope: float) -> float:
    """Logistic sigmoid → [0, 1].  Higher value → higher score."""
    return float(1.0 / (1.0 + np.exp(-slope * (value - center))))


def _ear_to_score_exponential(ear: float) -> float:
    """
    EAR → [0, 1] через экспоненциальную функцию.

    EAR >= 0.28 → score = 0.0  (глаза открыты, усталости нет)
    EAR = 0.22  → score ≈ 0.5  (начало усталости)
    EAR <= 0.18 → score → 1.0  (глаза закрыты, сильная усталость)

    Формула: score = 1 - exp(-rate * (EAR_threshold - EAR))
    """
    if ear >= 0.28:
        return 0.0
    if ear <= EAR_CLOSED_THRESHOLD:
        return 1.0
    # Экспоненциальное нарастание от 0.18 до 0.28
    score = 1.0 - np.exp(-_EAR_EXPONENTIAL_RATE * (0.28 - ear))
    # Нормализуем к [0, 1] в этом диапазоне
    max_score_at_028 = 1.0 - np.exp(-_EAR_EXPONENTIAL_RATE * (0.28 - 0.18))
    return float(np.clip(score / max_score_at_028, 0.0, 1.0))


class FatigueAnalyzer:
    """
    Анализатор усталости на основе комбинированного скора и анализа трендов.

    Все компоненты нормализованы к [0, 1] перед взвешиванием.
    Итоговый скор — в диапазоне [0, 100].

    ИСПРАВЛЕНИЯ v2:
      • EAR score — нелинейный, экспоненциальное падение
      • Меньше сглаживание (было rolling 10 кадров, стало 0.85*current + 0.15*avg)
      • Мгновенный штраф при EAR < 0.22
    """

    def __init__(self, window_size_seconds: int = 30):
        self.window_size = window_size_seconds

        # ── Rolling histories (уменьшено для более быстрой реакции) ─
        self.ear_history = deque(maxlen=60)
        self.mar_history = deque(maxlen=60)
        self.pitch_history = deque(maxlen=60)
        self.emotion_history = deque(maxlen=50)
        self.timestamps = deque(maxlen=100)

        # ── Blink tracking ──────────────────────────────────
        self.blink_count = 0
        self.blink_timestamps = deque(maxlen=30)
        self.last_ear = 0.35
        self.was_eyes_closed = False

        # ── Event cooldown ──────────────────────────────────
        self.fatigue_events = []
        self.last_fatigue_event_time = 0

        # ── Weights (sum = 1.0) ─────────────────────────────
        self.weights = {
            "ear": 0.55,    # EAR — главный признак
            "mar": 0.15,
            "blink": 0.10,
            "emotion": 0.10,
            "trend": 0.10,
        }

    # ── Public API ─────────────────────────────────────────────

    def update(self, ear: float, mar: float, pitch: float,
               emotion: str, current_time: float) -> dict:
        """
        Обновить данные и вернуть текущий уровень усталости.

        Returns dict with:
            fatigue_score   (0–100), fatigue_level, trend,
            ear_trend, blink_rate, avg_ear, avg_mar
        """
        self.ear_history.append(ear)
        self.mar_history.append(mar)
        self.pitch_history.append(pitch)
        self.emotion_history.append(emotion)
        self.timestamps.append(current_time)

        self._detect_blink(ear, current_time)

        # Нормализованные суб-скоры (каждый ∈ [0, 1])
        ear_score = self._get_ear_score()
        mar_score = self._get_mar_score()
        blink_score = self._get_blink_score()
        emotion_score = self._get_emotion_score()
        trend_score = self._get_trend_score()

        # Взвешенная сумма → [0, 1], затем масштаб в [0, 100]
        w = self.weights
        raw_score = (
            w["ear"] * ear_score +
            w["mar"] * mar_score +
            w["blink"] * blink_score +
            w["emotion"] * emotion_score +
            w["trend"] * trend_score
        )
        fatigue_score = min(100.0, max(0.0, raw_score * 100.0))

        # ── МГНОВЕННЫЙ ШТРАФ: если EAR ниже критического ──
        # При EAR < 0.18 — принудительно повышаем fatigue_score до максимума
        if ear <= EAR_CLOSED_THRESHOLD:
            # Глаза закрыты — внимание не может быть высоким
            fatigue_score = max(fatigue_score, 85.0)
        elif ear < EAR_CRITICAL_THRESHOLD:
            # EAR между 0.18 и 0.22 — экспоненциальный штраф
            penalty = 1.0 - _ear_to_score_exponential(ear)
            fatigue_score = max(fatigue_score, 60.0 + 25.0 * penalty)

        fatigue_level, trend = self._analyze_fatigue_state(fatigue_score)

        return {
            "fatigue_score": fatigue_score,
            "fatigue_level": fatigue_level,
            "trend": trend,
            "ear_trend": self._get_ear_trend(),
            "blink_rate": self._get_blink_rate(),
            "avg_ear": self._get_avg_ear(),
            "avg_mar": self._get_avg_mar(),
            # Debug: individual normalised components
            "_debug": {
                "ear_norm": round(ear_score, 3),
                "mar_norm": round(mar_score, 3),
                "blink_norm": round(blink_score, 3),
                "emotion_norm": round(emotion_score, 3),
                "trend_norm": round(trend_score, 3),
            },
        }

    # ── Blink detection (unchanged logic) ──────────────────────

    def _detect_blink(self, ear: float, current_time: float):
        """Определение моргания по пересечению EAR порогов."""
        if self.last_ear > 0.28 and ear < 0.22:
            if not self.was_eyes_closed:
                self.blink_count += 1
                self.blink_timestamps.append(current_time)
                self.was_eyes_closed = True
        elif ear > 0.25:
            self.was_eyes_closed = False
        self.last_ear = ear

    # ── Normalised sub-scores ──────────────────────────────────

    def _get_ear_score(self) -> float:
        """
        EAR → [0, 1]  (0 = широко открыты, 1 = закрыты).

        НЕЛИНЕЙНАЯ экспоненциальная функция:
          EAR >= 0.28 → score = 0.0
          EAR = 0.22  → score ≈ 0.5
          EAR <= 0.18 → score = 1.0

        Учитывает текущий EAR (вес 0.85) + минимум за 3 кадра (0.15).
        Убрано избыточное сглаживание на 10 кадров.
        """
        if len(self.ear_history) < 1:
            return 0.0

        current_ear = float(self.ear_history[-1])
        recent3 = list(self.ear_history)[-3:]
        min_ear = float(min(recent3))

        # Основной score — текущий EAR (нелинейный)
        current_score = _ear_to_score_exponential(current_ear)

        # Лёгкое сглаживание по минимуму за 3 кадра
        min_score = _ear_to_score_exponential(min_ear)

        return float(np.clip(
            0.85 * current_score + 0.15 * min_score,
            0.0, 1.0
        ))

    def _get_mar_score(self) -> float:
        """
        MAR → [0, 1]  (0 = рот закрыт, 1 = зевота).

        Учитывает пиковое значение за последние 10 отсчётов.
        """
        if len(self.mar_history) < 5:
            return 0.0

        recent_mar = list(self.mar_history)[-10:]
        max_mar = max(recent_mar)

        return _min_max(max_mar, MAR_MIN_PHYSIO, MAR_MAX_PHYSIO)

    def _get_blink_score(self) -> float:
        """
        Blink rate → [0, 1] через сигмоиду.

        Натуральное насыщение:  < 10/мин → ~0,  > 45/мин → ~1.
        """
        blink_rate = self._get_blink_rate()
        return _sigmoid(blink_rate, BLINK_SIGMOID_CENTER, BLINK_SIGMOID_SLOPE)

    def _get_emotion_score(self) -> float:
        """
        Emotion → [0, 1]  по категориальной карте.

        'Сонливость' / 'Усталость' → 1.0, 'Радость' → 0.0.
        """
        if len(self.emotion_history) < 3:
            return 0.1  # neutral default

        emotions = list(self.emotion_history)[-10:]

        fatigue_keywords = [
            "Усталость", "Tired", "Грусть", "Sad",
            "Сонливость", "Drowsy", "Скука", "Bored",
        ]
        positive_keywords = [
            "Счастье", "Happy", "Радость", "Joy",
            "Внимательность", "Alert", "Нейтрально", "Neutral",
            "Спокойствие", "Calm",
        ]

        fatigue_count = sum(
            1 for e in emotions
            if any(kw.lower() in e.lower() for kw in fatigue_keywords)
        )
        positive_count = sum(
            1 for e in emotions
            if any(kw.lower() in e.lower() for kw in positive_keywords)
        )

        # Доля «усталых» эмоций в окне
        ratio = fatigue_count / max(len(emotions), 1)

        # Бонус если много позитивных (снижает скор)
        penalty = positive_count / max(len(emotions), 1) * 0.3

        return float(np.clip(ratio - penalty, 0.0, 1.0))

    def _get_trend_score(self) -> float:
        """
        EAR trend → [0, 1]  (0 = стабильный/растущий, 1 = резкое падение).

        Сравнивает среднее EAR в первой и второй половинах окна (15 отсчётов).
        """
        if len(self.ear_history) < 15:
            return 0.0

        recent = list(self.ear_history)[-15:]
        first_half = np.mean(recent[:7])
        second_half = np.mean(recent[7:])

        # Отрицательное изменение = EAR падает = усталость растёт
        change = float(max(0, first_half - second_half))

        return _min_max(change, TREND_MIN, TREND_MAX)

    # ── Helpers ────────────────────────────────────────────────

    def _get_ear_trend(self) -> str:
        """Направление тренда EAR."""
        if len(self.ear_history) < 15:
            return "stable"

        recent = list(self.ear_history)[-15:]
        first_half = np.mean(recent[:7])
        second_half = np.mean(recent[7:])
        change = second_half - first_half

        if change < -0.03:
            return "decreasing"
        elif change > 0.03:
            return "increasing"
        return "stable"

    def _get_blink_rate(self) -> int:
        """Частота морганий в минуту (скользящее окно 60 с)."""
        if len(self.blink_timestamps) < 2:
            return 0

        recent_blinks = [t for t in self.blink_timestamps
                         if time.time() - t < 60]
        if len(recent_blinks) < 2:
            return 0

        time_span = recent_blinks[-1] - recent_blinks[0]
        if time_span < 1:
            time_span = 1
        return int(len(recent_blinks) / time_span * 60)

    def _get_avg_ear(self) -> float:
        if not self.ear_history:
            return 0.35
        return float(np.mean(list(self.ear_history)[-20:]))

    def _get_avg_mar(self) -> float:
        if not self.mar_history:
            return 0.0
        return float(np.mean(list(self.mar_history)[-20:]))

    # ── State analysis ─────────────────────────────────────────

    def _analyze_fatigue_state(self, fatigue_score: float):
        """
        Определить уровень усталости и тренд.

        Levels: normal (<30), mild (30–49), moderate (50–69), severe (≥70).
        """
        trend = self._get_ear_trend()

        if fatigue_score >= 70:
            level = "severe"
        elif fatigue_score >= 50:
            level = "moderate"
        elif fatigue_score >= 30:
            level = "mild"
        else:
            level = "normal"

        # Ухудшаем уровень при негативном тренде
        if trend == "decreasing" and fatigue_score > 30:
            level = self._worsen_level(level)

        return level, trend

    @staticmethod
    def _worsen_level(level: str) -> str:
        levels = ["normal", "mild", "moderate", "severe"]
        try:
            idx = levels.index(level)
            return levels[min(idx + 1, 3)]
        except ValueError:
            return level

    # ── Events ─────────────────────────────────────────────────

    def get_fatigue_event(self, current_time: float):
        """Вернуть текстовое событие усталости с учётом cooldown."""
        cooldown = 5.0
        if current_time - self.last_fatigue_event_time < cooldown:
            return None

        level, trend = self._analyze_fatigue_state(
            self._calculate_raw_score_for_event()
        )

        if level == "severe":
            self.last_fatigue_event_time = current_time
            return "Сильная усталость"
        elif level == "moderate":
            self.last_fatigue_event_time = current_time
            return "Умеренная усталость"
        elif level == "mild" and trend == "decreasing":
            self.last_fatigue_event_time = current_time
            return "Лёгкая усталость (снижение)"

        return None

    def _calculate_raw_score_for_event(self) -> float:
        """Пересчитать скор для event-логики (0–100)."""
        w = self.weights
        return min(100.0, max(0.0, (
            w["ear"] * self._get_ear_score() +
            w["mar"] * self._get_mar_score() +
            w["blink"] * self._get_blink_score() +
            w["emotion"] * self._get_emotion_score() +
            w["trend"] * self._get_trend_score()
        ) * 100.0))

    def get_status_text(self) -> str:
        """Текстовое описание текущего состояния."""
        if len(self.ear_history) < 5:
            return "Анализ..."

        score = self._calculate_raw_score_for_event()
        level, trend = self._analyze_fatigue_state(score)

        status_map = {
            "normal": "Бодрое",
            "mild": "Лёгкая усталость",
            "moderate": "Усталость",
            "severe": "Сильная усталость",
        }
        status = status_map.get(level, "Бодрое")

        if trend == "decreasing":
            status += " ↓"
        elif trend == "increasing":
            status += " ↑"

        return status

    def reset(self):
        """Полный сброс."""
        self.ear_history.clear()
        self.mar_history.clear()
        self.pitch_history.clear()
        self.emotion_history.clear()
        self.timestamps.clear()
        self.blink_count = 0
        self.blink_timestamps.clear()
        self.fatigue_events.clear()
        self.last_fatigue_event_time = 0

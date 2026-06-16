import time
from collections import deque


class AttentionTracker:
    """Скользящее среднее внимания за временное окно.

    Хранит не чаще 1 sample/sec, удаляет записи старше window_seconds.
    Пока данных меньше окна — возвращает среднее по накопленному.
    """

    def __init__(self, window_seconds: int = 600, sample_interval: float = 1.0):
        self._window = window_seconds
        self._interval = sample_interval
        self._samples: deque = deque()
        self._last_raw: float = 100.0
        self._last_sample_time: float = 0.0

    def update(self, value: float, timestamp: float | None = None) -> float:
        """Добавить sample (не чаще sample_interval) и вернуть сглаженное."""
        if timestamp is None:
            timestamp = time.time()
        self._last_raw = value

        # Добавляем sample не чаще interval
        if timestamp - self._last_sample_time >= self._interval:
            self._samples.append((timestamp, value))
            self._last_sample_time = timestamp

        return self._get_smoothed(timestamp)

    def get_smoothed(self, timestamp: float | None = None) -> float:
        """Вернуть среднее по окну (или последнее raw, если окно пусто)."""
        if timestamp is None:
            timestamp = time.time()
        return self._get_smoothed(timestamp)

    def _get_smoothed(self, timestamp: float) -> float:
        # Удаляем устаревшие записи
        cutoff = timestamp - self._window
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

        if not self._samples:
            return self._last_raw

        total = 0.0
        for _, v in self._samples:
            total += v
        return total / len(self._samples)

    def reset(self) -> None:
        """Очистить историю (например, при смене пользователя)."""
        self._samples.clear()
        self._last_raw = 100.0
        self._last_sample_time = 0.0

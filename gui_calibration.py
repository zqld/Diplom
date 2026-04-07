"""
CalibrationDialog — минималистичный пошаговый диалог калибровки.

Шаги:
  1. Лицо    — смотреть прямо, не двигаться ~3 сек
  2. Осанка  — сесть прямо, зафиксировать позу ~3 сек
  3. Рука    — показать раскрытую ладонь ~3 сек
  4. Зона    — переместить руку в верхний левый угол, затем в нижний правый

Прогресс опрашивается через QTimer (100 мс), без лишних сигналов.
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFrame, QProgressBar,
                             QScrollArea, QWidget)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

try:
    from src.sound_manager import sound_manager as _sound_manager
except Exception:
    _sound_manager = None

_C = {
    'bg':        '#1A1A1F',
    'card':      '#252530',
    'input':     '#2D2D3A',
    'border':    '#3A3A45',
    'primary':   '#FFFFFF',
    'muted':     '#A0A0B0',
    'dim':       '#6A6A7A',
    'accent':    '#6B8AFE',
    'good':      '#4ADE80',
    'warning':   '#FBBF24',
}


class _StepCard(QFrame):
    """Карточка одного шага калибровки."""

    def __init__(self, icon: str, title: str, hint: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {_C['input']};
                border-radius: 12px;
                border: 1px solid {_C['border']};
            }}
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(10)

        # ── заголовок ──
        top = QHBoxLayout()
        top.setSpacing(10)

        self._icon_lbl = QLabel(icon)
        self._icon_lbl.setFont(QFont("Segoe UI", 20))
        self._icon_lbl.setFixedWidth(34)
        self._icon_lbl.setStyleSheet("background: transparent; border: none;")
        top.addWidget(self._icon_lbl)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)

        self._title_lbl = QLabel(title)
        self._title_lbl.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
        self._title_lbl.setStyleSheet(
            f"color: {_C['primary']}; background: transparent; border: none;")
        title_col.addWidget(self._title_lbl)

        self._hint_lbl = QLabel(hint)
        self._hint_lbl.setFont(QFont("Segoe UI", 11))
        self._hint_lbl.setWordWrap(True)
        self._hint_lbl.setStyleSheet(
            f"color: {_C['muted']}; background: transparent; border: none;")
        title_col.addWidget(self._hint_lbl)

        top.addLayout(title_col)
        top.addStretch()

        self._status_lbl = QLabel("ожидание")
        self._status_lbl.setFont(QFont("Segoe UI", 11))
        self._status_lbl.setStyleSheet(
            f"color: {_C['dim']}; background: transparent; border: none;")
        top.addWidget(self._status_lbl)

        lay.addLayout(top)

        # ── прогресс-бар ──
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(8)
        self._bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {_C['border']};
                border-radius: 4px;
                border: none;
            }}
            QProgressBar::chunk {{
                background-color: {_C['accent']};
                border-radius: 4px;
            }}
        """)
        lay.addWidget(self._bar)

    # ── публичный API ──────────────────────────────────────────────────────────

    def set_hint(self, text: str):
        self._hint_lbl.setText(text)

    def set_progress(self, pct: int):
        self._bar.setValue(pct)

    def set_state(self, state: str):
        """state: 'idle' | 'active' | 'done' | 'skip'"""
        if state == 'idle':
            self._status_lbl.setText("ожидание")
            self._status_lbl.setStyleSheet(
                f"color: {_C['dim']}; background: transparent; border: none;")
            self._bar.setValue(0)
            self._bar.setStyleSheet(self._bar.styleSheet()
                                    .replace(_C['good'], _C['accent'])
                                    .replace(_C['warning'], _C['accent']))
        elif state == 'active':
            self._status_lbl.setText("идёт сбор...")
            self._status_lbl.setStyleSheet(
                f"color: {_C['warning']}; background: transparent; border: none;")
            self._bar.setStyleSheet(self._bar.styleSheet()
                                    .replace(_C['good'], _C['accent'])
                                    .replace(_C['warning'], _C['accent']))
        elif state == 'done':
            self._status_lbl.setText("готово ✓")
            self._status_lbl.setStyleSheet(
                f"color: {_C['good']}; background: transparent; border: none;")
            self._bar.setValue(100)
            self._bar.setStyleSheet(f"""
                QProgressBar {{
                    background-color: {_C['border']};
                    border-radius: 4px;
                    border: none;
                }}
                QProgressBar::chunk {{
                    background-color: {_C['good']};
                    border-radius: 4px;
                }}
            """)
        elif state == 'transition':
            self._status_lbl.setText("переместите руку!")
            self._status_lbl.setStyleSheet(
                f"color: {_C['accent']}; font-weight: 600; background: transparent; border: none;")
            self._bar.setValue(0)
            self._bar.setStyleSheet(f"""
                QProgressBar {{
                    background-color: {_C['border']};
                    border-radius: 4px;
                    border: none;
                }}
                QProgressBar::chunk {{
                    background-color: {_C['accent']};
                    border-radius: 4px;
                }}
            """)
        elif state == 'skip':
            self._status_lbl.setText("пропущено")
            self._status_lbl.setStyleSheet(
                f"color: {_C['dim']}; background: transparent; border: none;")


class CalibrationDialog(QDialog):
    """
    Минималистичный пошаговый диалог калибровки.

    Шаги: лицо → осанка → рука → зона жестов (каждый можно пропустить).

    Использование:
        dlg = CalibrationDialog(calibration_manager, parent=self)
        dlg.exec()
    """

    # Целевые значения сэмплов (порог завершения шага)
    _FACE_TARGET    = 20
    _POSTURE_TARGET = 20
    _HAND_TARGET    = 20
    _ZONE_TARGET    = 15   # для каждого угла

    def __init__(self, calibration_manager, parent=None):
        super().__init__(parent)
        self._cm = calibration_manager

        self.setWindowTitle("Калибровка")
        self.setFixedSize(480, 580)
        self.setModal(True)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {_C['bg']};
                font-family: 'Segoe UI', sans-serif;
            }}
        """)

        # ── корневой layout ───────────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(14)

        hdr = QLabel("🎯  Калибровка")
        hdr.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        hdr.setStyleSheet(f"color: {_C['primary']};")
        root.addWidget(hdr)

        sub = QLabel("Следуйте инструкциям на карточках. Каждый шаг можно пропустить.")
        sub.setFont(QFont("Segoe UI", 11))
        sub.setWordWrap(True)
        sub.setStyleSheet(f"color: {_C['muted']};")
        root.addWidget(sub)

        # ── карточки шагов ────────────────────────────────────────────────────
        self._face_card = _StepCard(
            "😐", "Калибровка лица",
            "Смотрите прямо в камеру, не двигайтесь (~3 сек)"
        )
        root.addWidget(self._face_card)

        self._posture_card = _StepCard(
            "🧍", "Калибровка осанки",
            "Сядьте прямо в удобную позу и удерживайте её (~3 сек)"
        )
        root.addWidget(self._posture_card)

        self._hand_card = _StepCard(
            "✋", "Калибровка руки",
            "Покажите раскрытую ладонь на расстоянии 40–50 см"
        )
        root.addWidget(self._hand_card)

        self._zone_card = _StepCard(
            "📐", "Зона управления мышью",
            "Наведите руку в верхний левый угол экрана и держите (~2 сек)"
        )
        root.addWidget(self._zone_card)

        root.addStretch()

        # ── кнопки ────────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch()

        self._btn_skip = QPushButton("Пропустить шаг")
        self._btn_skip.setFixedHeight(40)
        self._btn_skip.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_skip.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {_C['dim']};
                border: 1px solid {_C['border']};
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                color: {_C['muted']};
                border-color: {_C['muted']};
            }}
        """)
        self._btn_skip.clicked.connect(self._skip_step)
        self._btn_skip.setVisible(False)
        btn_row.addWidget(self._btn_skip)

        self._btn_start = QPushButton("Начать")
        self._btn_start.setFixedSize(120, 40)
        self._btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_start.setStyleSheet(f"""
            QPushButton {{
                background-color: {_C['accent']};
                color: #FFFFFF;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{ background-color: #8AA3FF; }}
            QPushButton:disabled {{
                background-color: {_C['border']};
                color: {_C['dim']};
            }}
        """)
        self._btn_start.clicked.connect(self._start)
        btn_row.addWidget(self._btn_start)

        root.addLayout(btn_row)

        # ── внутреннее состояние ──────────────────────────────────────────────
        # Фазы: 'idle' → 'face' → 'posture' → 'hand' → 'zone_tl' → 'zone_br' → 'done'
        self._phase = 'idle'

        # Флаги «мы сами запустили калибровку» — защита от ложного завершения
        self._face_was_started    = False
        self._posture_was_started = False
        self._hand_was_started    = False
        self._zone_tl_started     = False
        self._zone_br_waiting     = False  # True пока показываем "переместите руку"

        # Таймер опроса (каждые 100 мс)
        self._poll = QTimer(self)
        self._poll.setInterval(100)
        self._poll.timeout.connect(self._tick)

    # ── слоты ─────────────────────────────────────────────────────────────────

    def _start(self):
        if not self._cm:
            return
        self._phase = 'face'
        self._face_was_started = True
        self._cm.start_face_calibration()
        self._face_card.set_state('active')
        self._btn_start.setEnabled(False)
        self._btn_start.setText("Идёт...")
        self._btn_skip.setVisible(True)
        self._poll.start()

    def _skip_step(self):
        """Пропустить текущий активный шаг и перейти к следующему."""
        phase = self._phase
        if phase == 'face':
            if self._cm and self._cm._is_calibrating_face:
                self._cm._is_calibrating_face = False
            self._face_card.set_state('skip')
            self._enter_posture()
        elif phase == 'posture':
            if self._cm and self._cm._is_calibrating_posture:
                self._cm._is_calibrating_posture = False
            self._posture_card.set_state('skip')
            self._enter_hand()
        elif phase == 'hand':
            if self._cm and self._cm._is_calibrating_hand:
                self._cm._is_calibrating_hand = False
            self._hand_card.set_state('skip')
            self._enter_zone_tl()
        elif phase in ('zone_tl', 'zone_br'):
            if self._cm and self._cm._is_calibrating_zone:
                self._cm._is_calibrating_zone = False
            self._zone_card.set_state('skip')
            self._finish_all()

    # ── переходы между шагами ─────────────────────────────────────────────────

    def _enter_posture(self):
        self._phase = 'posture'
        self._posture_was_started = True
        self._cm.start_posture_calibration()
        self._posture_card.set_state('active')

    def _enter_hand(self):
        self._phase = 'hand'
        self._hand_was_started = True
        self._cm.start_hand_calibration()
        self._hand_card.set_state('active')

    def _enter_zone_tl(self):
        self._phase = 'zone_tl'
        self._zone_tl_started = True
        self._cm.start_gesture_zone_calibration()
        self._zone_card.set_state('active')
        self._zone_card.set_hint(
            "Наведите руку в верхний ЛЕВЫЙ угол экрана и держите (~2 сек)"
        )

    def _enter_zone_br(self):
        self._phase = 'zone_br'
        self._zone_br_waiting = True
        # Звуковой сигнал смены угла
        if _sound_manager:
            _sound_manager.attention()
        # Визуальный индикатор: прогресс сбрасывается, яркий статус
        self._zone_card.set_state('transition')
        self._zone_card.set_hint(
            "✅ Левый угол зафиксирован!\n"
            "Теперь переместите руку в нижний ПРАВЫЙ угол и держите (~3 сек)"
        )
        # Сообщаем calibration_manager о переходе с задержкой 1.5 с —
        # даём пользователю время переместить руку
        QTimer.singleShot(1500, self._confirm_zone_br)

    # ── таймерный опрос ───────────────────────────────────────────────────────

    def _confirm_zone_br(self):
        """Вызывается через 1.5 с после перехода — реально запускает сбор BR-угла."""
        if self._phase != 'zone_br':
            return
        self._zone_br_waiting = False
        self._cm.advance_gesture_zone_step()   # сбрасывает таймер + переключает step
        self._zone_card.set_state('active')

    def _tick(self):
        if not self._cm:
            return

        if self._phase == 'face':
            progress = len(self._cm._face_samples)
            self._face_card.set_progress(
                min(100, int(progress / self._FACE_TARGET * 100))
            )
            # Завершение: флаг _is_calibrating_face сбросился ПОСЛЕ нашего старта
            if self._face_was_started and not self._cm._is_calibrating_face:
                self._face_card.set_progress(100)
                self._face_card.set_state('done')
                self._enter_posture()

        elif self._phase == 'posture':
            progress = len(self._cm._posture_samples)
            self._posture_card.set_progress(
                min(100, int(progress / self._POSTURE_TARGET * 100))
            )
            if self._posture_was_started and not self._cm._is_calibrating_posture:
                self._posture_card.set_progress(100)
                self._posture_card.set_state('done')
                self._enter_hand()

        elif self._phase == 'hand':
            progress = len(self._cm._hand_samples)
            self._hand_card.set_progress(
                min(100, int(progress / self._HAND_TARGET * 100))
            )
            if self._hand_was_started and not self._cm._is_calibrating_hand:
                self._hand_card.set_progress(100)
                self._hand_card.set_state('done')
                self._enter_zone_tl()

        elif self._phase == 'zone_tl':
            progress = len(self._cm._zone_topleft_samples)
            self._zone_card.set_progress(
                min(50, int(progress / self._ZONE_TARGET * 50))   # 0–50 %
            )
            if self._zone_tl_started and progress >= self._ZONE_TARGET:
                self._enter_zone_br()

        elif self._phase == 'zone_br':
            if self._zone_br_waiting:
                # Ждём пока пользователь переместит руку (1.5 с)
                return
            progress = len(self._cm._zone_bottomright_samples)
            self._zone_card.set_progress(
                min(100, 50 + int(progress / self._ZONE_TARGET * 50))  # 50–100 %
            )
            if not self._cm._is_calibrating_zone:
                # finish_gesture_zone_calibration() уже вызван из VideoThread
                self._zone_card.set_progress(100)
                self._zone_card.set_state('done')
                self._phase = 'done'
                self._finish_all()

    def _finish_all(self):
        self._poll.stop()
        self._btn_skip.setVisible(False)
        self._btn_start.setText("Готово ✓")
        self._btn_start.setEnabled(True)
        self._btn_start.setStyleSheet(f"""
            QPushButton {{
                background-color: {_C['good']};
                color: #111;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{ background-color: #6EF7A0; }}
        """)
        self._btn_start.clicked.disconnect()
        self._btn_start.clicked.connect(self.accept)
        # Автозакрытие через 1.5 сек
        QTimer.singleShot(1500, self.accept)

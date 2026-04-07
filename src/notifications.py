import sys
from PyQt6.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                             QPushButton, QApplication, QGraphicsOpacityEffect, QFrame)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QCursor
from sqlalchemy import create_engine, text
import datetime

# --- КОНФИГУРАЦИЯ ПО УМОЛЧАНИЮ ---
DEFAULT_SETTINGS = {
    "work_limit_minutes": 45,
    "posture_window_minutes": 3,
    "posture_bad_percent": 30,
    "yawn_limit": 4,
    "yawn_window_minutes": 10
}

# Цветовая палитра в едином стиле приложения
_COLORS = {
    'bg':           '#252530',
    'border':       '#3A3A45',
    'text_primary': '#FFFFFF',
    'text_muted':   '#A0A0B0',
    'accent':       '#6B8AFE',   # перерыв / общее
    'danger':       '#F87171',   # осанка
    'warning':      '#FBBF24',   # усталость
    'good':         '#4ADE80',   # pomodoro / успех
}


# --- 1. TOAST УВЕДОМЛЕНИЕ — единый стиль для всего приложения ---
class ToastNotification(QWidget):
    """
    Всплывающее уведомление в правом нижнем углу экрана.
    accent_color — цвет левой полоски и заголовка (по умолчанию accent).
    """
    def __init__(self, title: str, message: str,
                 accent_color: str = None, parent=None):
        super().__init__(parent)
        accent = accent_color or _COLORS['accent']

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedSize(360, 110)

        # ── Outer container (provides the rounded border + left accent stripe) ──
        container = QFrame(self)
        container.setGeometry(0, 0, 360, 110)
        container.setStyleSheet(f"""
            QFrame {{
                background-color: {_COLORS['bg']};
                border: 1px solid {_COLORS['border']};
                border-left: 4px solid {accent};
                border-radius: 12px;
            }}
        """)

        # ── Inner layout ──
        root = QVBoxLayout(container)
        root.setContentsMargins(18, 12, 14, 12)
        root.setSpacing(5)

        # Title row: icon-title + close button
        title_row = QHBoxLayout()
        title_row.setSpacing(0)

        lbl_title = QLabel(title)
        lbl_title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        lbl_title.setStyleSheet(f"color: {accent}; background: transparent; border: none;")
        title_row.addWidget(lbl_title)
        title_row.addStretch()

        btn_close = QPushButton("×")
        btn_close.setFixedSize(22, 22)
        btn_close.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        btn_close.setStyleSheet(f"""
            QPushButton {{
                color: {_COLORS['text_muted']};
                background: transparent;
                border: none;
                font-size: 18px;
                font-weight: bold;
                padding: 0;
            }}
            QPushButton:hover {{ color: {_COLORS['text_primary']}; }}
        """)
        btn_close.clicked.connect(self._close_now)
        title_row.addWidget(btn_close)
        root.addLayout(title_row)

        # Message
        lbl_msg = QLabel(message)
        lbl_msg.setFont(QFont("Segoe UI", 11))
        lbl_msg.setWordWrap(True)
        lbl_msg.setStyleSheet(f"color: {_COLORS['text_muted']}; background: transparent; border: none;")
        root.addWidget(lbl_msg)

        # ── Opacity animation ──
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(0)

        self._anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self._anim.setDuration(400)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        # Auto-close timer (6 seconds visible)
        self._close_timer = QTimer(self)
        self._close_timer.setSingleShot(True)
        self._close_timer.setInterval(6000)
        self._close_timer.timeout.connect(self.fade_out)

        self._closing = False

    def show_toast(self):
        screen = QApplication.primaryScreen().availableGeometry()
        x = screen.right()  - self.width()  - 20
        y = screen.bottom() - self.height() - 20
        self.move(x, y)
        self.show()

        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.start()
        self._close_timer.start()

    def fade_out(self):
        if self._closing:
            return
        self._closing = True
        self._close_timer.stop()
        self._anim.setStartValue(self.opacity_effect.opacity())
        self._anim.setEndValue(0.0)
        self._anim.finished.connect(self.close)
        self._anim.start()

    def _close_now(self):
        self._close_timer.stop()
        self.fade_out()

# --- 2. МЕНЕДЖЕР ЛОГИКИ ---
class NotificationManager:
    def __init__(self, db_path="session_data.db"):
        self.db_uri = f"sqlite:///data/{db_path}"
        self.engine = create_engine(self.db_uri)
        
        # Настройки
        self.settings = DEFAULT_SETTINGS.copy()
        
        # Время начала работы (для таймера перерыва)
        self.session_start = datetime.datetime.now()
        
        # Кулдауны (чтобы не спамить уведомлениями каждые 5 секунд)
        self.last_alert_time = {
            "posture": datetime.datetime.min,
            "fatigue": datetime.datetime.min,
            "break": datetime.datetime.min
        }

    def update_settings(self, new_settings):
        """Обновление настроек из GUI"""
        self.settings.update(new_settings)

    def check_conditions(self):
        """
        Главный метод анализа. Вызывается раз в минуту (или чаще).
        Возвращает (Title, Message) или None.
        """
        now = datetime.datetime.now()
        conn = self.engine.connect()
        alert = None

        try:
            # 1. ПРОВЕРКА ВРЕМЕНИ РАБОТЫ (Time Limit)
            # Если прошло больше времени, чем указано в настройках
            work_duration = (now - self.session_start).total_seconds() / 60
            if work_duration > self.settings["work_limit_minutes"]:
                # Если с последнего уведомления прошло > 10 минут
                if (now - self.last_alert_time["break"]).total_seconds() > 600:
                    self.last_alert_time["break"] = now
                    # Сбрасываем таймер сессии, будто человек отдохнул (условно)
                    # Либо просто напоминаем
                    return "Пора отдохнуть", f"Вы работаете уже {int(work_duration)} минут без перерыва. Сделайте разминку."

            # 2. АНАЛИЗ ОСАНКИ (За последние N минут)
            window_min = self.settings["posture_window_minutes"]
            time_threshold = now - datetime.timedelta(minutes=window_min)
            
            sql_posture = text("""
                SELECT posture_status FROM face_logs 
                WHERE timestamp > :thresh
            """)
            result = conn.execute(sql_posture, {"thresh": time_threshold}).fetchall()
            
            if result:
                total_records = len(result)
                # В БД хранится "Bad Posture [model_name]" — сравниваем через startswith
                bad_count = sum(1 for r in result if str(r[0]).startswith('Bad Posture'))
                bad_percent = (bad_count / total_records) * 100
                
                # Если процент плохой осанки выше порога
                if bad_percent > self.settings["posture_bad_percent"]:
                    # Кулдаун 5 минут
                    if (now - self.last_alert_time["posture"]).total_seconds() > 300:
                        self.last_alert_time["posture"] = now
                        return "Следите за осанкой", f"За последние {window_min} мин вы сутулились {int(bad_percent)}% времени."

            # 3. АНАЛИЗ УСТАЛОСТИ (Зевки)
            window_yawn = self.settings["yawn_window_minutes"]
            time_threshold_yawn = now - datetime.timedelta(minutes=window_yawn)
            
            sql_yawn = text("""
                SELECT COUNT(*) FROM face_logs
                WHERE timestamp > :thresh AND fatigue_status LIKE 'Yawning%'
            """)
            yawn_count = conn.execute(sql_yawn, {"thresh": time_threshold_yawn}).scalar()
            
            if yawn_count >= self.settings["yawn_limit"]:
                # Кулдаун 10 минут
                if (now - self.last_alert_time["fatigue"]).total_seconds() > 600:
                    self.last_alert_time["fatigue"] = now
                    return "Обнаружена усталость", f"Вы часто зеваете ({yawn_count} раз за {window_yawn} мин). Рекомендуется проветрить помещение."

        except Exception as e:
            print(f"Ошибка анализатора: {e}")
        finally:
            conn.close()
        
        return None
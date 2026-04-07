from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
                             QFrame, QDateTimeEdit, QWidget, QSizePolicy)
from PyQt6.QtCore import Qt, QDateTime
from PyQt6.QtGui import QFont
from src.analytics import AnalyticsEngine, MplCanvas
from src.data_exporter import DataExporter
import pandas as pd
import matplotlib.dates as mdates
import datetime


DARK_COLORS = {
    'bg_main': '#1A1A1F',
    'bg_card': '#252530',
    'bg_input': '#2D2D3A',
    'text_primary': '#FFFFFF',
    'text_secondary': '#A0A0B0',
    'text_muted': '#6A6A7A',
    'accent': '#6B8AFE',
    'accent_hover': '#8AA3FF',
    'border': '#3A3A45',
    'border_light': '#4A4A55',
    'good': '#4ADE80',
    'warning': '#FBBF24',
    'danger': '#F87171',
    'good_bg': 'rgba(74, 222, 128, 0.15)',
    'warning_bg': 'rgba(251, 191, 36, 0.15)',
    'danger_bg': 'rgba(248, 113, 113, 0.15)',
}

CHART_COLORS = {
    'background': '#1E1E25',
    'text': '#FFFFFF',
    'text_secondary': '#A0A0B0',
    'grid': '#3A3A45',
    'accent': '#6B8AFE',
    'good': '#4ADE80',
    'warning': '#FBBF24',
    'danger': '#F87171',
    'posture_line': '#9CA3AF',
    'ear_line': '#8A8AFE',
    'threshold': '#FBBF24',
}


class ModernStatsCard(QFrame):
    def __init__(self, icon, title, value="--", unit="", color=None, parent=None):
        super().__init__(parent)
        self._color = color or DARK_COLORS['text_primary']
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {DARK_COLORS['bg_card']};
                border-radius: 14px;
                border: none;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(6)

        # Icon + title row
        top_row = QHBoxLayout()
        icon_label = QLabel(icon)
        icon_label.setFont(QFont("Segoe UI", 18))
        icon_label.setStyleSheet("background: transparent;")
        top_row.addWidget(icon_label)
        top_row.addStretch()
        layout.addLayout(top_row)

        # Value
        self.value_label = QLabel(f"{value}")
        self.value_label.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        self.value_label.setStyleSheet(f"color: {self._color}; background: transparent;")
        layout.addWidget(self.value_label)

        # Title + unit
        sub_row = QHBoxLayout()
        self.title_label = QLabel(f"{title}{(' ' + unit) if unit else ''}")
        self.title_label.setFont(QFont("Segoe UI", 11))
        self.title_label.setStyleSheet(f"color: {DARK_COLORS['text_muted']}; background: transparent;")
        sub_row.addWidget(self.title_label)
        sub_row.addStretch()
        layout.addLayout(sub_row)

    def update_value(self, value, color=None):
        self.value_label.setText(str(value))
        if color:
            self.value_label.setStyleSheet(f"color: {color}; background: transparent;")


class QuickPeriodButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFixedHeight(34)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._active = False
        self._apply_style(False)

    def _apply_style(self, active):
        if active:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {DARK_COLORS['accent']};
                    color: #FFFFFF;
                    border: none;
                    border-radius: 8px;
                    padding: 6px 14px;
                    font-size: 12px;
                    font-weight: 600;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {DARK_COLORS['bg_input']};
                    color: {DARK_COLORS['text_secondary']};
                    border: 1px solid {DARK_COLORS['border']};
                    border-radius: 8px;
                    padding: 6px 14px;
                    font-size: 12px;
                    font-weight: 500;
                }}
                QPushButton:hover {{
                    border-color: {DARK_COLORS['accent']};
                    color: {DARK_COLORS['accent']};
                }}
            """)

    def set_active(self, active):
        self._active = active
        self._apply_style(active)


class StatsWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Аналитика сессии")
        self.resize(1320, 860)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {DARK_COLORS['bg_main']};
            }}
            QLabel {{
                font-family: 'Segoe UI', sans-serif;
                background: transparent;
            }}
            QDateTimeEdit {{
                background-color: {DARK_COLORS['bg_input']};
                border: 1px solid {DARK_COLORS['border']};
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 13px;
                color: {DARK_COLORS['text_primary']};
                font-family: 'Segoe UI';
            }}
            QDateTimeEdit:focus {{
                border-color: {DARK_COLORS['accent']};
            }}
            QDateTimeEdit::drop-down {{ border: none; width: 20px; }}
            QDateTimeEdit::down-arrow {{
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid {DARK_COLORS['text_muted']};
            }}
            QScrollBar:vertical {{
                background: {DARK_COLORS['bg_card']};
                width: 6px; border-radius: 3px;
            }}
            QScrollBar::handle:vertical {{
                background: {DARK_COLORS['border_light']};
                border-radius: 3px; min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
        """)

        self.engine = AnalyticsEngine()
        self._quick_period = "1h"  # default

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(28, 28, 28, 28)
        main_layout.setSpacing(24)

        # ── LEFT PANEL ────────────────────────────────────────────────
        left_panel = QWidget()
        left_panel.setStyleSheet("background: transparent;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(20)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header_row = QHBoxLayout()
        title = QLabel("📊  Аналитика")
        title.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {DARK_COLORS['text_primary']};")
        header_row.addWidget(title)
        header_row.addStretch()
        left_layout.addLayout(header_row)

        # 4 metric cards
        cards_row = QHBoxLayout()
        cards_row.setSpacing(16)
        self.card_records  = ModernStatsCard("📋", "Записей за период", "0")
        self.card_yawns    = ModernStatsCard("😴", "Зевков", "0", color=DARK_COLORS['warning'])
        self.card_posture  = ModernStatsCard("🧍", "Плохая осанка", "0%", color=DARK_COLORS['good'])
        self.card_attention = ModernStatsCard("👁", "Ср. внимание", "0%", color=DARK_COLORS['accent'])
        for c in [self.card_records, self.card_yawns, self.card_posture, self.card_attention]:
            cards_row.addWidget(c)
        left_layout.addLayout(cards_row)

        # Chart container
        chart_container = QFrame()
        chart_container.setStyleSheet(f"""
            QFrame {{
                background-color: {DARK_COLORS['bg_card']};
                border-radius: 16px;
                border: none;
            }}
        """)
        chart_layout = QVBoxLayout(chart_container)
        chart_layout.setContentsMargins(24, 20, 24, 20)
        chart_layout.setSpacing(12)

        chart_header_row = QHBoxLayout()
        chart_title = QLabel("Динамика показателей")
        chart_title.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        chart_title.setStyleSheet(f"color: {DARK_COLORS['text_primary']};")
        chart_header_row.addWidget(chart_title)
        chart_header_row.addStretch()

        self.status_dot = QLabel("● Нет данных")
        self.status_dot.setFont(QFont("Segoe UI", 11))
        self.status_dot.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
        chart_header_row.addWidget(self.status_dot)
        chart_layout.addLayout(chart_header_row)

        self.canvas = MplCanvas(self, width=7, height=6, dpi=100)
        self.canvas.setStyleSheet("background: transparent; border: none;")
        chart_layout.addWidget(self.canvas)

        left_layout.addWidget(chart_container, stretch=1)

        # ── RIGHT PANEL ───────────────────────────────────────────────
        right_panel = QFrame()
        right_panel.setFixedWidth(290)
        right_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {DARK_COLORS['bg_card']};
                border-radius: 16px;
                border: none;
            }}
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(22, 22, 22, 22)
        right_layout.setSpacing(16)

        # Quick period section
        period_title = QLabel("ПЕРИОД")
        period_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        period_title.setStyleSheet(f"color: {DARK_COLORS['text_muted']}; letter-spacing: 1.2px;")
        right_layout.addWidget(period_title)

        quick_row1 = QHBoxLayout()
        quick_row1.setSpacing(8)
        self.btn_1h   = QuickPeriodButton("Час")
        self.btn_3h   = QuickPeriodButton("3 часа")
        self.btn_today = QuickPeriodButton("Сегодня")
        for btn in [self.btn_1h, self.btn_3h, self.btn_today]:
            quick_row1.addWidget(btn)
        right_layout.addLayout(quick_row1)

        self.btn_1h.clicked.connect(lambda: self._set_quick_period("1h"))
        self.btn_3h.clicked.connect(lambda: self._set_quick_period("3h"))
        self.btn_today.clicked.connect(lambda: self._set_quick_period("today"))

        # Divider
        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet(f"background-color: {DARK_COLORS['border']}; border: none;")
        right_layout.addWidget(div)

        # Custom range label
        custom_label = QLabel("Свой период")
        custom_label.setFont(QFont("Segoe UI", 11))
        custom_label.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
        right_layout.addWidget(custom_label)

        # Start datetime
        start_frame = QFrame()
        start_frame.setStyleSheet(f"""
            QFrame {{ background-color: {DARK_COLORS['bg_input']}; border-radius: 10px; border: none; }}
        """)
        sl = QVBoxLayout(start_frame)
        sl.setContentsMargins(14, 10, 14, 10)
        sl.setSpacing(4)
        QLabel_s = QLabel("От")
        QLabel_s.setFont(QFont("Segoe UI", 10))
        QLabel_s.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
        sl.addWidget(QLabel_s)
        self.dt_start = QDateTimeEdit(QDateTime.currentDateTime().addSecs(-3600))
        self.dt_start.setDisplayFormat("dd.MM.yyyy  HH:mm")
        self.dt_start.setCalendarPopup(True)
        self.dt_start.setFixedHeight(38)
        self.dt_start.dateTimeChanged.connect(self._on_custom_range_changed)
        sl.addWidget(self.dt_start)
        right_layout.addWidget(start_frame)

        # End datetime
        end_frame = QFrame()
        end_frame.setStyleSheet(f"""
            QFrame {{ background-color: {DARK_COLORS['bg_input']}; border-radius: 10px; border: none; }}
        """)
        el = QVBoxLayout(end_frame)
        el.setContentsMargins(14, 10, 14, 10)
        el.setSpacing(4)
        QLabel_e = QLabel("До")
        QLabel_e.setFont(QFont("Segoe UI", 10))
        QLabel_e.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
        el.addWidget(QLabel_e)
        self.dt_end = QDateTimeEdit(QDateTime.currentDateTime())
        self.dt_end.setDisplayFormat("dd.MM.yyyy  HH:mm")
        self.dt_end.setCalendarPopup(True)
        self.dt_end.setFixedHeight(38)
        self.dt_end.dateTimeChanged.connect(self._on_custom_range_changed)
        el.addWidget(self.dt_end)
        right_layout.addWidget(end_frame)

        # Apply button
        self.btn_apply = QPushButton("Обновить данные")
        self.btn_apply.setFixedHeight(46)
        self.btn_apply.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_apply.setStyleSheet(f"""
            QPushButton {{
                background-color: {DARK_COLORS['accent']};
                color: #FFFFFF; border: none; border-radius: 10px;
                font-size: 14px; font-weight: 600;
            }}
            QPushButton:hover {{ background-color: {DARK_COLORS['accent_hover']}; }}
        """)
        self.btn_apply.clicked.connect(self.plot_data)
        right_layout.addWidget(self.btn_apply)

        # Export button
        self.btn_export = QPushButton("📥  Экспорт CSV")
        self.btn_export.setFixedHeight(40)
        self.btn_export.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_export.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {DARK_COLORS['text_secondary']};
                border: 1px solid {DARK_COLORS['border']};
                border-radius: 10px; font-size: 13px;
            }}
            QPushButton:hover {{
                border-color: {DARK_COLORS['accent']};
                color: {DARK_COLORS['accent']};
            }}
        """)
        self.btn_export.clicked.connect(self.export_data)
        right_layout.addWidget(self.btn_export)

        # Summary card
        summary_card = QFrame()
        summary_card.setStyleSheet(f"""
            QFrame {{ background-color: {DARK_COLORS['bg_input']}; border-radius: 12px; border: none; }}
        """)
        summary_vl = QVBoxLayout(summary_card)
        summary_vl.setContentsMargins(16, 14, 16, 14)
        summary_vl.setSpacing(8)

        summary_title_lbl = QLabel("Итоги периода")
        summary_title_lbl.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        summary_title_lbl.setStyleSheet(f"color: {DARK_COLORS['text_secondary']};")
        summary_vl.addWidget(summary_title_lbl)

        self.summary_text = QLabel("Выберите период и нажмите «Обновить данные».")
        self.summary_text.setWordWrap(True)
        self.summary_text.setFont(QFont("Segoe UI", 11))
        self.summary_text.setStyleSheet(f"color: {DARK_COLORS['text_muted']}; line-height: 1.5;")
        summary_vl.addWidget(self.summary_text)

        right_layout.addWidget(summary_card)
        right_layout.addStretch()

        # Close button
        btn_close = QPushButton("Закрыть")
        btn_close.setFixedHeight(46)
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {DARK_COLORS['text_secondary']};
                border: 1px solid {DARK_COLORS['border']};
                border-radius: 10px; font-size: 14px; font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {DARK_COLORS['bg_input']};
                border-color: {DARK_COLORS['border_light']};
            }}
        """)
        btn_close.clicked.connect(self.close)
        right_layout.addWidget(btn_close)

        main_layout.addWidget(left_panel, stretch=1)
        main_layout.addWidget(right_panel)

        # Initial load
        self._set_quick_period("1h")

    # ── Quick-period helpers ──────────────────────────────────────────
    def _set_quick_period(self, period: str):
        self._quick_period = period
        now = QDateTime.currentDateTime()
        self.btn_1h.set_active(period == "1h")
        self.btn_3h.set_active(period == "3h")
        self.btn_today.set_active(period == "today")

        self.dt_start.blockSignals(True)
        self.dt_end.blockSignals(True)
        if period == "1h":
            self.dt_start.setDateTime(now.addSecs(-3600))
            self.dt_end.setDateTime(now)
        elif period == "3h":
            self.dt_start.setDateTime(now.addSecs(-10800))
            self.dt_end.setDateTime(now)
        elif period == "today":
            start_of_day = QDateTime(now.date(), now.time())
            start_of_day.setTime(start_of_day.time().fromMSecsSinceStartOfDay(0))
            from PyQt6.QtCore import QTime
            self.dt_start.setDateTime(QDateTime(now.date(), QTime(0, 0, 0)))
            self.dt_end.setDateTime(now)
        self.dt_start.blockSignals(False)
        self.dt_end.blockSignals(False)
        self.plot_data()

    def _on_custom_range_changed(self):
        # Deactivate all quick buttons when user manually edits datetime
        self._quick_period = "custom"
        self.btn_1h.set_active(False)
        self.btn_3h.set_active(False)
        self.btn_today.set_active(False)

    # ── Main chart rendering ──────────────────────────────────────────
    def plot_data(self):
        start_dt = self.dt_start.dateTime().toPyDateTime()
        end_dt   = self.dt_end.dateTime().toPyDateTime()

        df = self.engine.load_data(start_dt, end_dt)

        if df is None or df.empty:
            self._show_empty_state("Нет данных за выбранный период")
            return

        ax1, ax2 = self.canvas.axes
        ax1.clear()
        ax2.clear()

        time_fmt = mdates.DateFormatter('%H:%M')

        # ── Chart 1: Posture (pitch angle) ──
        df['pitch_smooth'] = df['pitch'].rolling(window=5, min_periods=1).mean() if len(df) > 5 else df['pitch']
        ax1.plot(df['timestamp'], df['pitch_smooth'],
                 color=CHART_COLORS['posture_line'], linewidth=2.0, label='Наклон головы', alpha=0.9)
        ax1.axhline(y=15, color=CHART_COLORS['threshold'], linestyle='--',
                    alpha=0.6, linewidth=1.5, label='Порог (15°)')
        ax1.fill_between(df['timestamp'], df['pitch_smooth'], 15,
                         where=(df['pitch_smooth'] > 15),
                         facecolor='#F87171', alpha=0.25, interpolate=True)
        ax1.fill_between(df['timestamp'], df['pitch_smooth'], 15,
                         where=(df['pitch_smooth'] <= 15),
                         facecolor='#4ADE80', alpha=0.12, interpolate=True)

        # Overlay bad posture events as vertical shading
        bad_posture = df[df['posture_status'] == 'Bad Posture']
        if not bad_posture.empty:
            for _, row in bad_posture.iterrows():
                ax1.axvspan(row['timestamp'], row['timestamp'] + datetime.timedelta(seconds=2),
                            color='#F87171', alpha=0.12)

        self._style_ax(ax1, 'Осанка — наклон головы', 'Градусы', time_fmt, legend=True)

        # ── Chart 2: Fatigue (EAR + events) ──
        df['ear_smooth'] = df['ear'].rolling(window=10, min_periods=1).mean() if len(df) > 10 else df['ear']
        ax2.plot(df['timestamp'], df['ear_smooth'],
                 color=CHART_COLORS['ear_line'], linewidth=2.0, label='EAR (открытость глаз)', alpha=0.9)
        ax2.axhline(y=0.22, color=CHART_COLORS['threshold'], linestyle='--',
                    alpha=0.6, linewidth=1.5, label='Порог засыпания')

        yawns = df[df['fatigue_status'] == 'Yawning']
        if not yawns.empty:
            ax2.scatter(yawns['timestamp'], yawns['ear'],
                        color='#F87171', s=70, marker='o', label=f'Зевок ({len(yawns)})',
                        zorder=5, alpha=0.9, edgecolors='#1E1E25', linewidths=1)

        closed_eyes = df[df['fatigue_status'] == 'Eyes Closed']
        if not closed_eyes.empty:
            ax2.scatter(closed_eyes['timestamp'], closed_eyes['ear'],
                        color='#FBBF24', s=60, marker='x',
                        label=f'Закр. глаза ({len(closed_eyes)})', zorder=5, alpha=0.9)

        self._style_ax(ax2, 'Усталость — EAR', 'EAR', time_fmt, legend=True, xlabel='Время')

        self.canvas.fig.tight_layout(pad=2.5)
        self.canvas.draw()

        # ── Update cards & summary ──
        total_records     = len(df)
        bad_posture_count = len(df[df['posture_status'] == 'Bad Posture'])
        posture_percent   = (bad_posture_count / total_records * 100) if total_records > 0 else 0
        yawn_count        = len(yawns)

        avg_ear       = df['ear'].mean() if len(df) > 0 else 0.3
        avg_attention = max(0, min(100, int((avg_ear - 0.15) / 0.20 * 100)))

        self._update_cards(total_records, yawn_count, posture_percent, avg_attention)
        self._update_summary(total_records, yawn_count, posture_percent, avg_attention, start_dt, end_dt, df)
        self.status_dot.setText(f"● {total_records} записей")
        self.status_dot.setStyleSheet(f"color: {DARK_COLORS['good']};")

    def _style_ax(self, ax, title, ylabel, time_fmt, legend=False, xlabel=None):
        bg = CHART_COLORS['background']
        ax.set_facecolor(bg)
        ax.set_title(title, color=CHART_COLORS['text'], fontsize=12,
                     fontweight='600', pad=12, loc='left')
        ax.set_ylabel(ylabel, color=CHART_COLORS['text_secondary'], fontsize=10, labelpad=8)
        if xlabel:
            ax.set_xlabel(xlabel, color=CHART_COLORS['text_secondary'], fontsize=10, labelpad=8)
        ax.xaxis.set_major_formatter(time_fmt)
        ax.tick_params(colors=CHART_COLORS['text_secondary'], labelsize=9)
        ax.grid(True, linestyle='-', alpha=0.25, color=CHART_COLORS['grid'], zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color(CHART_COLORS['grid'])
        if legend:
            ax.legend(loc='upper right', frameon=True, facecolor=bg,
                      edgecolor=CHART_COLORS['grid'],
                      labelcolor=CHART_COLORS['text_secondary'],
                      fontsize=9, framealpha=1)

    def _show_empty_state(self, message):
        ax1, ax2 = self.canvas.axes
        for ax in [ax1, ax2]:
            ax.clear()
            ax.set_facecolor(CHART_COLORS['background'])
            ax.text(0.5, 0.5, message, transform=ax.transAxes,
                    ha='center', va='center', fontsize=14,
                    color=DARK_COLORS['text_muted'], fontfamily='Segoe UI')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        self.canvas.draw()
        self.summary_text.setText("Нет данных за выбранный период.")
        self.status_dot.setText("● Нет данных")
        self.status_dot.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
        self._update_cards(0, 0, 0, 0)

    def _update_cards(self, records, yawns, posture_percent, attention):
        self.card_records.update_value(records)

        yawn_color = DARK_COLORS['danger'] if yawns >= 4 else (DARK_COLORS['warning'] if yawns >= 2 else DARK_COLORS['good'])
        self.card_yawns.update_value(yawns, yawn_color)

        posture_color = DARK_COLORS['danger'] if posture_percent > 30 else (DARK_COLORS['warning'] if posture_percent > 10 else DARK_COLORS['good'])
        self.card_posture.update_value(f"{posture_percent:.1f}%", posture_color)

        att_color = DARK_COLORS['good'] if attention >= 70 else (DARK_COLORS['warning'] if attention >= 40 else DARK_COLORS['danger'])
        self.card_attention.update_value(f"{attention}%", att_color)

    def _update_summary(self, records, yawns, posture_pct, attention, start_dt, end_dt, df):
        duration_min = int((end_dt - start_dt).total_seconds() / 60)
        drowsy_count = len(df[df['fatigue_status'] == 'Drowsy']) if df is not None else 0
        eye_closed   = len(df[df['fatigue_status'] == 'Eyes Closed']) if df is not None else 0

        posture_verdict = "✅ Хорошая" if posture_pct <= 10 else ("⚠️ Требует внимания" if posture_pct <= 30 else "❌ Плохая")
        att_verdict     = "✅ Высокое" if attention >= 70 else ("⚠️ Среднее" if attention >= 40 else "❌ Низкое")

        date_str = start_dt.strftime('%d.%m.%Y')
        time_str = f"{start_dt.strftime('%H:%M')} — {end_dt.strftime('%H:%M')}"

        lines = [
            f"<b style='color:{DARK_COLORS['text_secondary']};'>{date_str}  {time_str}</b><br>",
            f"<span style='color:{DARK_COLORS['text_muted']};'>Длительность:</span> <span style='color:{DARK_COLORS['text_secondary']};'>{duration_min} мин</span><br>",
            f"<span style='color:{DARK_COLORS['text_muted']};'>Осанка:</span> <span>{posture_verdict}</span><br>",
            f"<span style='color:{DARK_COLORS['text_muted']};'>Внимание:</span> <span>{att_verdict}</span><br>",
        ]
        if drowsy_count:
            lines.append(f"<span style='color:{DARK_COLORS['text_muted']};'>Сонливость:</span> <span style='color:{DARK_COLORS['warning']};'>{drowsy_count} сек</span><br>")
        if eye_closed:
            lines.append(f"<span style='color:{DARK_COLORS['text_muted']};'>Закр. глаза:</span> <span style='color:{DARK_COLORS['danger']};'>{eye_closed} сек</span>")

        self.summary_text.setText(f"<div style='line-height:1.9; font-size:11px;'>{''.join(lines)}</div>")

    def export_data(self):
        from PyQt6.QtWidgets import QMessageBox
        start_dt = self.dt_start.dateTime().toPyDateTime()
        end_dt   = self.dt_end.dateTime().toPyDateTime()
        exporter = DataExporter()
        filepath, message = exporter.export_to_csv(start_dt, end_dt)
        msg = QMessageBox(self)
        msg.setWindowTitle("Экспорт данных")
        msg.setStyleSheet(f"""
            QMessageBox {{ background-color: {DARK_COLORS['bg_card']}; color: {DARK_COLORS['text_primary']}; }}
            QPushButton {{
                background-color: {DARK_COLORS['accent']}; color: #FFFFFF;
                border: none; border-radius: 8px; padding: 8px 20px;
            }}
        """)
        if filepath:
            msg.setText(f"✓ {message}\n\nФайл: {filepath}")
            msg.setIcon(QMessageBox.Icon.Information)
        else:
            msg.setText(f"✗ {message}")
            msg.setIcon(QMessageBox.Icon.Warning)
        msg.exec()

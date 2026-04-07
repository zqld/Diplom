from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QPushButton, QFrame,
                             QHBoxLayout, QGridLayout, QTabWidget, QWidget,
                             QScrollArea, QProgressBar, QSizePolicy)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from src.progress_tracker import ProgressTracker
import datetime


DARK_COLORS = {
    'bg_main':       '#1A1A1F',
    'bg_card':       '#252530',
    'bg_input':      '#2D2D3A',
    'text_primary':  '#FFFFFF',
    'text_secondary':'#A0A0B0',
    'text_muted':    '#6A6A7A',
    'accent':        '#6B8AFE',
    'accent_hover':  '#8AA3FF',
    'good':          '#4ADE80',
    'warning':       '#FBBF24',
    'danger':        '#F87171',
    'border':        '#3A3A45',
    'border_light':  '#4A4A55',
}

# Russian weekday abbreviations (Monday = 0)
DAYS_RU = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']


def _ru_date(date_str: str) -> str:
    """Convert '2026-04-06' → '06.04 (Пн)'."""
    dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    return dt.strftime('%d.%m') + f' ({DAYS_RU[dt.weekday()]})'


def _attention_color(val: int) -> str:
    if val >= 70:
        return DARK_COLORS['good']
    elif val >= 40:
        return DARK_COLORS['warning']
    return DARK_COLORS['danger']


def _trend_color(trend: str) -> str:
    return {
        'improving': DARK_COLORS['good'],
        'declining': DARK_COLORS['danger'],
    }.get(trend, DARK_COLORS['text_secondary'])


def _trend_label(trend: str) -> str:
    return {'improving': 'Улучшается ↑', 'declining': 'Снижается ↓', 'stable': 'Стабильно →'}.get(trend, trend)


# ── Matplotlib canvas ──────────────────────────────────────────────────────────
class ProgressCanvas(FigureCanvas):
    BG = '#1E1E25'

    def __init__(self, parent=None, width=7, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor=self.BG)
        self.ax  = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.BG)
        self.fig.tight_layout(pad=2.5)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def draw_chart(self, history: list):
        self.ax.clear()
        self.ax.set_facecolor(self.BG)
        self.fig.set_facecolor(self.BG)

        active = [h for h in history if h['total_records'] > 0]
        if not active:
            self.ax.text(0.5, 0.5, 'Нет данных за 7 дней',
                         transform=self.ax.transAxes, ha='center', va='center',
                         fontsize=14, color=DARK_COLORS['text_muted'], fontfamily='Segoe UI')
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            for s in self.ax.spines.values():
                s.set_visible(False)
            self.draw()
            return

        dates     = [datetime.datetime.strptime(h['date'], '%Y-%m-%d') for h in history]
        attention = [h['avg_attention'] for h in history]
        fatigue   = [h['fatigue_events'] for h in history]

        # Attention line
        self.ax.plot(dates, attention, color='#6B8AFE', linewidth=2.5,
                     marker='o', markersize=8, label='Внимание %', zorder=3)

        # Fill under line
        self.ax.fill_between(dates, attention, alpha=0.12, color='#6B8AFE')

        # Threshold lines
        self.ax.axhline(y=70, color='#4ADE80', linestyle='--', alpha=0.5, linewidth=1.5, label='Хорошо (70%)')
        self.ax.axhline(y=40, color='#FBBF24', linestyle='--', alpha=0.5, linewidth=1.5, label='Тревога (40%)')

        # Data labels on points
        for d, a in zip(dates, attention):
            if a > 0:
                self.ax.annotate(f'{a}%', (d, a), textcoords='offset points',
                                 xytext=(0, 10), ha='center', fontsize=9,
                                 color=_attention_color(a), fontfamily='Segoe UI')

        # Axes styling
        self.ax.set_ylim(-5, 110)
        self.ax.set_yticks([0, 25, 50, 75, 100])
        self.ax.set_ylabel('Внимание %', color=DARK_COLORS['text_secondary'], fontsize=10)
        self.ax.tick_params(colors=DARK_COLORS['text_muted'], labelsize=9)

        import matplotlib.dates as mdates
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
        self.ax.xaxis.set_major_locator(mdates.DayLocator())

        self.ax.grid(True, linestyle='-', alpha=0.18, color='#3A3A45')
        for s in self.ax.spines.values():
            s.set_visible(False)
        self.ax.spines['bottom'].set_visible(True)
        self.ax.spines['bottom'].set_color('#3A3A45')

        self.ax.legend(loc='lower right', facecolor='#252530',
                       edgecolor='#3A3A45', labelcolor=DARK_COLORS['text_secondary'],
                       fontsize=9, framealpha=1)

        self.fig.tight_layout(pad=2.5)
        self.draw()


# ── Reusable summary stat card ─────────────────────────────────────────────────
class SummaryCard(QFrame):
    def __init__(self, icon, title, value, color=None, parent=None):
        super().__init__(parent)
        color = color or DARK_COLORS['text_primary']
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {DARK_COLORS['bg_input']};
                border-radius: 14px; border: none;
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(6)

        icon_lbl = QLabel(icon)
        icon_lbl.setFont(QFont("Segoe UI", 20))
        icon_lbl.setStyleSheet("background: transparent;")
        lay.addWidget(icon_lbl)

        val_lbl = QLabel(str(value))
        val_lbl.setFont(QFont("Segoe UI", 26, QFont.Weight.Bold))
        val_lbl.setStyleSheet(f"color: {color}; background: transparent;")
        lay.addWidget(val_lbl)
        self.val_lbl = val_lbl

        title_lbl = QLabel(title)
        title_lbl.setFont(QFont("Segoe UI", 11))
        title_lbl.setStyleSheet(f"color: {DARK_COLORS['text_muted']}; background: transparent;")
        lay.addWidget(title_lbl)


# ── Day card in weekly tab ─────────────────────────────────────────────────────
class DayCard(QFrame):
    def __init__(self, day_data: dict, parent=None):
        super().__init__(parent)
        date_str = _ru_date(day_data['date'])
        has_data = day_data['total_records'] > 0
        att      = day_data['avg_attention']

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {DARK_COLORS['bg_card']};
                border-radius: 12px; border: none;
            }}
        """)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(0)

        # Date label
        date_lbl = QLabel(date_str)
        date_lbl.setFont(QFont("Segoe UI", 13, QFont.Weight.Medium))
        date_lbl.setStyleSheet(f"color: {DARK_COLORS['text_primary']}; min-width: 110px; background: transparent;")
        lay.addWidget(date_lbl)

        if has_data:
            # Attention badge + bar
            att      = int(att)
            att_col  = _attention_color(att)
            att_lbl  = QLabel(f"{att}%")
            att_lbl.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
            att_lbl.setStyleSheet(f"color: {att_col}; min-width: 52px; background: transparent;")
            lay.addWidget(att_lbl)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(att))
            bar.setTextVisible(False)
            bar.setFixedWidth(110)
            bar.setFixedHeight(8)
            bar.setStyleSheet(f"""
                QProgressBar {{
                    background-color: {DARK_COLORS['bg_input']};
                    border-radius: 4px; border: none;
                }}
                QProgressBar::chunk {{
                    background-color: {att_col};
                    border-radius: 4px;
                }}
            """)
            lay.addWidget(bar)
            lay.addSpacing(24)

            # Fatigue events
            fat_lbl = QLabel(f"😴  {day_data['fatigue_events']}")
            fat_lbl.setFont(QFont("Segoe UI", 12))
            fat_lbl.setStyleSheet(f"color: {DARK_COLORS['warning']}; min-width: 60px; background: transparent;")
            lay.addWidget(fat_lbl)

            # Posture events
            pos_lbl = QLabel(f"🧍  {day_data['posture_events']}")
            pos_lbl.setFont(QFont("Segoe UI", 12))
            pos_lbl.setStyleSheet(f"color: {DARK_COLORS['danger']}; min-width: 60px; background: transparent;")
            lay.addWidget(pos_lbl)

            # Records count
            rec_lbl = QLabel(f"📊  {day_data['total_records']}")
            rec_lbl.setFont(QFont("Segoe UI", 11))
            rec_lbl.setStyleSheet(f"color: {DARK_COLORS['text_muted']}; background: transparent;")
            lay.addWidget(rec_lbl)
        else:
            no_lbl = QLabel("Нет данных за этот день")
            no_lbl.setFont(QFont("Segoe UI", 12))
            no_lbl.setStyleSheet(f"color: {DARK_COLORS['text_muted']}; font-style: italic; background: transparent;")
            lay.addWidget(no_lbl)

        lay.addStretch()


# ── Main Progress Window ───────────────────────────────────────────────────────
class ProgressWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("История прогресса")
        self.resize(960, 680)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {DARK_COLORS['bg_main']};
            }}
            QLabel {{
                font-family: 'Segoe UI', sans-serif;
                color: {DARK_COLORS['text_primary']};
                background: transparent;
            }}
            QScrollArea {{ background: transparent; border: none; }}
            QScrollBar:vertical {{
                background: {DARK_COLORS['bg_card']}; width: 6px; border-radius: 3px;
            }}
            QScrollBar::handle:vertical {{
                background: {DARK_COLORS['border_light']}; border-radius: 3px; min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
        """)

        self.tracker = ProgressTracker()
        self._history = self.tracker.get_progress_history(7)
        self._summary = self.tracker.get_weekly_summary()

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(20)

        # ── Header ──────────────────────────────────────────────────
        hdr_row = QHBoxLayout()
        hdr_lbl = QLabel("📈  Прогресс за неделю")
        hdr_lbl.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        hdr_row.addWidget(hdr_lbl)
        hdr_row.addStretch()

        active_days = self._summary.get('active_days', 0)
        days_lbl = QLabel(f"Активных дней: {active_days} / 7")
        days_lbl.setFont(QFont("Segoe UI", 13))
        days_lbl.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
        hdr_row.addWidget(days_lbl)
        root.addLayout(hdr_row)

        # ── Tabs ────────────────────────────────────────────────────
        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                background-color: {DARK_COLORS['bg_card']};
                border-radius: 14px; border: none; padding: 18px;
            }}
            QTabBar::tab {{
                background-color: {DARK_COLORS['bg_input']};
                color: {DARK_COLORS['text_secondary']};
                padding: 11px 26px; border-radius: 8px;
                margin-right: 8px; font-size: 13px; font-weight: 500;
            }}
            QTabBar::tab:selected {{
                background-color: {DARK_COLORS['accent']}; color: #FFFFFF;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {DARK_COLORS['border']}; color: {DARK_COLORS['text_primary']};
            }}
        """)

        tabs.addTab(self._build_summary_tab(), "📊  Сводка")
        tabs.addTab(self._build_chart_tab(),   "📈  График")
        tabs.addTab(self._build_weekly_tab(),  "📅  По дням")
        root.addWidget(tabs, stretch=1)

        # ── Close button ────────────────────────────────────────────
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
                border-color: {DARK_COLORS['accent_hover']};
                color: {DARK_COLORS['accent_hover']};
            }}
        """)
        btn_close.clicked.connect(self.close)
        root.addWidget(btn_close)

    # ── Summary tab ───────────────────────────────────────────────────────────
    def _build_summary_tab(self) -> QWidget:
        w   = QWidget()
        w.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(w)
        lay.setSpacing(16)
        lay.setContentsMargins(0, 0, 0, 0)

        s = self._summary

        # 4 cards in a grid
        grid = QGridLayout()
        grid.setSpacing(16)

        cards = [
            ("👁",  "Среднее внимание",  f"{s['avg_attention']}%",    _attention_color(s['avg_attention'])),
            ("😴",  "Событий усталости", str(s['total_fatigue']),      DARK_COLORS['warning']),
            ("🧍",  "Событий осанки",    str(s['total_posture']),       DARK_COLORS['danger']),
            ("📈",  "Тренд недели",      _trend_label(s['trend']),     _trend_color(s['trend'])),
        ]
        for i, (icon, title, value, color) in enumerate(cards):
            card = SummaryCard(icon, title, value, color)
            grid.addWidget(card, i // 2, i % 2)
        lay.addLayout(grid)

        # Improvement banner
        improvement = s.get('improvement', 0)
        if s.get('active_days', 0) >= 2:
            banner = QFrame()
            imp_color = DARK_COLORS['good'] if improvement >= 0 else DARK_COLORS['danger']
            arrow     = "↑" if improvement >= 0 else "↓"
            banner.setStyleSheet(f"""
                QFrame {{
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:0,
                        stop:0 {DARK_COLORS['bg_input']}, stop:1 {DARK_COLORS['bg_card']}
                    );
                    border-radius: 12px; border: none;
                }}
            """)
            bl = QHBoxLayout(banner)
            bl.setContentsMargins(20, 14, 20, 14)

            txt_lbl = QLabel(f"Изменение за неделю:  {arrow}  {abs(round(improvement, 1))}%")
            txt_lbl.setFont(QFont("Segoe UI", 14, QFont.Weight.Medium))
            txt_lbl.setStyleSheet(f"color: {imp_color};")
            bl.addWidget(txt_lbl)
            bl.addStretch()
            lay.addWidget(banner)

        lay.addStretch()
        return w

    # ── Chart tab ─────────────────────────────────────────────────────────────
    def _build_chart_tab(self) -> QWidget:
        w   = QWidget()
        w.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(w)
        lay.setSpacing(12)
        lay.setContentsMargins(0, 0, 0, 0)

        hdr = QLabel("Динамика внимания за 7 дней")
        hdr.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        hdr.setStyleSheet(f"color: {DARK_COLORS['text_primary']};")
        lay.addWidget(hdr)

        # Legend hint
        legend_row = QHBoxLayout()
        for dot_color, txt in [
            (DARK_COLORS['good'],    "≥70% — хорошо"),
            (DARK_COLORS['warning'], "40-69% — норма"),
            (DARK_COLORS['danger'],  "<40% — тревога"),
        ]:
            dot = QLabel("●")
            dot.setFont(QFont("Segoe UI", 14))
            dot.setStyleSheet(f"color: {dot_color};")
            lbl = QLabel(txt)
            lbl.setFont(QFont("Segoe UI", 11))
            lbl.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
            legend_row.addWidget(dot)
            legend_row.addWidget(lbl)
            legend_row.addSpacing(16)
        legend_row.addStretch()
        lay.addLayout(legend_row)

        self.chart_canvas = ProgressCanvas(w, width=7, height=4, dpi=100)
        self.chart_canvas.draw_chart(self._history)
        lay.addWidget(self.chart_canvas, stretch=1)

        return w

    # ── Weekly tab ────────────────────────────────────────────────────────────
    def _build_weekly_tab(self) -> QWidget:
        w   = QWidget()
        w.setStyleSheet("background: transparent;")
        outer_lay = QVBoxLayout(w)
        outer_lay.setSpacing(10)
        outer_lay.setContentsMargins(0, 0, 0, 0)

        hdr = QLabel("Детальная статистика по дням")
        hdr.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        hdr.setStyleSheet(f"color: {DARK_COLORS['text_primary']};")
        outer_lay.addWidget(hdr)

        # Column headers
        col_hdr = QFrame()
        col_hdr.setStyleSheet("background: transparent; border: none;")
        ch_lay = QHBoxLayout(col_hdr)
        ch_lay.setContentsMargins(18, 0, 18, 0)
        for lbl_text, min_w in [("Дата", 110), ("Внимание", 170), ("Усталость 😴", 60), ("Осанка 🧍", 60), ("Записей 📊", 0)]:
            l = QLabel(lbl_text)
            l.setFont(QFont("Segoe UI", 10))
            l.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
            if min_w:
                l.setMinimumWidth(min_w)
            ch_lay.addWidget(l)
        ch_lay.addStretch()
        outer_lay.addWidget(col_hdr)

        # Scrollable area with day cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent; border: none;")

        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        inner_lay = QVBoxLayout(inner)
        inner_lay.setSpacing(10)
        inner_lay.setContentsMargins(0, 0, 0, 0)

        for day_data in reversed(self._history):
            inner_lay.addWidget(DayCard(day_data))

        inner_lay.addStretch()
        scroll.setWidget(inner)
        outer_lay.addWidget(scroll, stretch=1)

        return w

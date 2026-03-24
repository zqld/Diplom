from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QPushButton, QFrame, 
                             QHBoxLayout, QGridLayout, QTabWidget, QWidget)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from src.progress_tracker import ProgressTracker
from src.analytics import AnalyticsEngine
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
    'good': '#4ADE80',
    'warning': '#FBBF24',
    'danger': '#F87171',
    'border': '#3A3A45',
}


class ProgressCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#1E1E25')
        super().__init__(self.fig)
        self.setParent(parent)


class ProgressWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("История прогресса")
        self.resize(900, 600)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {DARK_COLORS['bg_main']};
            }}
            QLabel {{
                font-family: 'Segoe UI', sans-serif;
                color: {DARK_COLORS['text_primary']};
            }}
        """)
        
        self.tracker = ProgressTracker()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)
        
        header = QLabel("📈 История прогресса")
        header.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        header.setStyleSheet(f"color: {DARK_COLORS['text_primary']};")
        layout.addWidget(header)
        
        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                background-color: {DARK_COLORS['bg_card']};
                border-radius: 12px;
                border: none;
                padding: 16px;
            }}
            QTabBar::tab {{
                background-color: {DARK_COLORS['bg_input']};
                color: {DARK_COLORS['text_secondary']};
                padding: 12px 24px;
                border-radius: 8px;
                margin-right: 8px;
                font-size: 13px;
                font-weight: 500;
            }}
            QTabBar::tab:selected {{
                background-color: {DARK_COLORS['accent']};
                color: #FFFFFF;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {DARK_COLORS['bg_card']};
            }}
        """)
        
        tabs.addTab(self._create_summary_tab(), "📊 Сводка")
        tabs.addTab(self._create_chart_tab(), "📈 Графики")
        tabs.addTab(self._create_weekly_tab(), "📅 За неделю")
        
        layout.addWidget(tabs)
        
        btn_close = QPushButton("Закрыть")
        btn_close.setFixedHeight(46)
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close.setStyleSheet(f"""
            QPushButton {{
                background-color: {DARK_COLORS['bg_card']};
                color: {DARK_COLORS['text_secondary']};
                border: 1px solid {DARK_COLORS['border']};
                border-radius: 10px;
                padding: 10px 24px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                border-color: {DARK_COLORS['accent']};
                color: {DARK_COLORS['accent']};
            }}
        """)
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close)
    
    def _create_summary_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)
        
        summary = self.tracker.get_weekly_summary()
        
        summary_cards = QGridLayout()
        summary_cards.setSpacing(16)
        
        cards_data = [
            ("Среднее внимание", f"{summary['avg_attention']}%", self._get_attention_color(summary['avg_attention'])),
            ("Событий усталости", str(summary['total_fatigue']), DARK_COLORS['warning']),
            ("Событий осанки", str(summary['total_posture']), DARK_COLORS['danger']),
            ("Тренд", summary['trend'].upper(), self._get_trend_color(summary['trend'])),
        ]
        
        for i, (title, value, color) in enumerate(cards_data):
            card = QFrame()
            card.setStyleSheet(f"""
                QFrame {{
                    background-color: {DARK_COLORS['bg_card']};
                    border-radius: 12px;
                    border: none;
                }}
            """)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(16, 14, 16, 14)
            card_layout.setSpacing(6)
            
            title_label = QLabel(title)
            title_label.setFont(QFont("Segoe UI", 11))
            title_label.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
            card_layout.addWidget(title_label)
            
            value_label = QLabel(value)
            value_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
            value_label.setStyleSheet(f"color: {color};")
            card_layout.addWidget(value_label)
            
            summary_cards.addWidget(card, i // 2, i % 2)
        
        layout.addLayout(summary_cards)
        
        if summary['active_days'] > 0:
            improvement_text = f"+{summary['improvement']}%" if summary['improvement'] > 0 else f"{summary['improvement']}%"
            improvement_card = QFrame()
            improvement_card.setStyleSheet(f"""
                QFrame {{
                    background-color: {DARK_COLORS['bg_input']};
                    border-radius: 12px;
                    border: none;
                }}
            """)
            imp_layout = QHBoxLayout(improvement_card)
            imp_layout.setContentsMargins(20, 16, 20, 16)
            
            imp_label = QLabel(f"📈 Изменение за неделю: {improvement_text}")
            imp_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Medium))
            imp_color = DARK_COLORS['good'] if summary['improvement'] > 0 else DARK_COLORS['danger']
            imp_label.setStyleSheet(f"color: {imp_color};")
            imp_layout.addWidget(imp_label)
            imp_layout.addStretch()
            
            days_label = QLabel(f"Активных дней: {summary['active_days']}/7")
            days_label.setFont(QFont("Segoe UI", 12))
            days_label.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
            imp_layout.addWidget(days_label)
            
            layout.addWidget(improvement_card)
        
        layout.addStretch()
        return widget
    
    def _create_chart_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)
        
        chart_header = QLabel("Динамика внимания за 7 дней")
        chart_header.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        chart_header.setStyleSheet(f"color: {DARK_COLORS['text_primary']};")
        layout.addWidget(chart_header)
        
        self.chart_canvas = ProgressCanvas(self, width=7, height=4, dpi=100)
        layout.addWidget(self.chart_canvas)
        
        self._plot_progress()
        
        return widget
    
    def _plot_progress(self):
        history = self.tracker.get_progress_history(7)
        
        if not history:
            self.chart_canvas.axes.text(0.5, 0.5, 'Нет данных', 
                                       transform=self.chart_canvas.axes.transAxes,
                                       ha='center', va='center', fontsize=14,
                                       color='#6A6A7A')
            self.chart_canvas.draw()
            return
        
        dates = [h['date'] for h in history]
        attention = [h['avg_attention'] for h in history]
        
        import datetime
        x = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in dates]
        
        self.chart_canvas.axes.clear()
        self.chart_canvas.axes.set_facecolor('#1E1E25')
        
        self.chart_canvas.axes.plot(x, attention, color='#6B8AFE', linewidth=2.5, 
                                    marker='o', markersize=8, label='Внимание %')
        
        self.chart_canvas.axes.axhline(y=70, color='#4ADE80', linestyle='--', 
                                       alpha=0.5, linewidth=1.5, label='Хорошо (70%)')
        self.chart_canvas.axes.axhline(y=40, color='#FBBF24', linestyle='--', 
                                       alpha=0.5, linewidth=1.5, label='Тревога (40%)')
        
        self.chart_canvas.axes.set_ylabel('Внимание %', color='#A0A0B0', fontsize=11)
        self.chart_canvas.axes.set_ylim(0, 100)
        self.chart_canvas.axes.set_yticks([0, 25, 50, 75, 100])
        
        self.chart_canvas.axes.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
        self.chart_canvas.axes.tick_params(colors='#6A6A7A', labelsize=10)
        
        self.chart_canvas.axes.grid(True, linestyle='-', alpha=0.2, color='#3A3A45')
        self.chart_canvas.axes.legend(loc='lower right', facecolor='#252530', 
                                      edgecolor='#3A3A45', labelcolor='#A0A0B0',
                                      fontsize=9)
        
        for spine in self.chart_canvas.axes.spines.values():
            spine.set_visible(False)
        self.chart_canvas.axes.spines['bottom'].set_visible(True)
        self.chart_canvas.axes.spines['bottom'].set_color('#3A3A45')
        
        self.chart_canvas.fig.tight_layout()
        self.chart_canvas.draw()
    
    def _create_weekly_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)
        
        header = QLabel("Детальная информация за неделю")
        header.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        header.setStyleSheet(f"color: {DARK_COLORS['text_primary']};")
        layout.addWidget(header)
        
        history = self.tracker.get_progress_history(7)
        
        for day_data in reversed(history):
            day_card = QFrame()
            day_card.setStyleSheet(f"""
                QFrame {{
                    background-color: {DARK_COLORS['bg_card']};
                    border-radius: 10px;
                    border: none;
                }}
            """)
            day_layout = QHBoxLayout(day_card)
            day_layout.setContentsMargins(16, 12, 16, 12)
            
            date_obj = datetime.datetime.strptime(day_data['date'], '%Y-%m-%d')
            date_str = date_obj.strftime('%d.%m (%a)')
            
            date_label = QLabel(date_str)
            date_label.setFont(QFont("Segoe UI", 13, QFont.Weight.Medium))
            date_label.setStyleSheet(f"color: {DARK_COLORS['text_primary']};")
            date_label.setMinimumWidth(100)
            day_layout.addWidget(date_label)
            
            if day_data['total_records'] > 0:
                att_label = QLabel(f"👁 {day_data['avg_attention']}%")
                att_label.setFont(QFont("Segoe UI", 12))
                att_label.setStyleSheet(f"color: {self._get_attention_color(day_data['avg_attention'])};")
                day_layout.addWidget(att_label)
                
                fatigue_label = QLabel(f"😴 {day_data['fatigue_events']}")
                fatigue_label.setFont(QFont("Segoe UI", 12))
                fatigue_label.setStyleSheet(f"color: {DARK_COLORS['warning']};")
                day_layout.addWidget(fatigue_label)
                
                posture_label = QLabel(f"🧍 {day_data['posture_events']}")
                posture_label.setFont(QFont("Segoe UI", 12))
                posture_label.setStyleSheet(f"color: {DARK_COLORS['danger']};")
                day_layout.addWidget(posture_label)
                
                records_label = QLabel(f"📊 {day_data['total_records']}")
                records_label.setFont(QFont("Segoe UI", 11))
                records_label.setStyleSheet(f"color: {DARK_COLORS['text_muted']};")
                day_layout.addWidget(records_label)
            else:
                no_data_label = QLabel("Нет данных")
                no_data_label.setFont(QFont("Segoe UI", 12))
                no_data_label.setStyleSheet(f"color: {DARK_COLORS['text_muted']}; font-style: italic;")
                day_layout.addWidget(no_data_label)
            
            day_layout.addStretch()
            layout.addWidget(day_card)
        
        layout.addStretch()
        return widget
    
    def _get_attention_color(self, value):
        if value >= 70:
            return DARK_COLORS['good']
        elif value >= 40:
            return DARK_COLORS['warning']
        else:
            return DARK_COLORS['danger']
    
    def _get_trend_color(self, trend):
        if trend == 'improving':
            return DARK_COLORS['good']
        elif trend == 'declining':
            return DARK_COLORS['danger']
        else:
            return DARK_COLORS['text_secondary']

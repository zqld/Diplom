from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QFrame, QDateTimeEdit, QWidget)
from PyQt6.QtCore import Qt, QDateTime
from PyQt6.QtGui import QFont
from neurofocus.utils.analytics import AnalyticsEngine, MplCanvas
from neurofocus.utils.data_export import DataExporter
from neurofocus.ui.theme import theme_manager
import pandas as pd
import matplotlib.dates as mdates


def get_chart_colors():
    c = theme_manager.colors
    return {
        'background': '#1E1E25' if theme_manager.current_theme == 'dark' else '#F0F0F0',
        'text': c['text_primary'],
        'text_secondary': c['text_secondary'],
        'grid': c['border'],
        'accent': c['accent'],
        'accent_light': c['accent_hover'],
        'good': c['good'],
        'warning': c['warning'],
        'danger': c['danger'],
        'posture_line': '#9CA3AF',
        'ear_line': '#8A8A9A',
        'yawn_marker': c['danger'],
        'threshold': c['warning'],
        'fill_good': 'rgba(74,222,128,0.2)',
        'fill_bad': 'rgba(248,113,113,0.2)',
    }


class ModernStatsCard(QFrame):
    def __init__(self, title, value="--", unit="", trend=None, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {theme_manager.colors['bg_card']};
                border-radius: 14px;
                border: none;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(6)
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Medium))
        title_label.setStyleSheet(f"color: {theme_manager.colors['text_muted']}; letter-spacing: 0.3px;")
        layout.addWidget(title_label)
        
        value_layout = QHBoxLayout()
        value_layout.setSpacing(8)
        
        self.value_label = QLabel(f"{value}{unit}")
        self.value_label.setFont(QFont("Segoe UI", 26, QFont.Weight.Bold))
        self.value_label.setStyleSheet(f"color: {theme_manager.colors['text_primary']};")
        self.value_label.setMinimumWidth(80)
        value_layout.addWidget(self.value_label)
        
        if trend:
            trend_label = QLabel(trend)
            trend_label.setFont(QFont("Segoe UI", 12))
            if trend.startswith("↑"):
                trend_label.setStyleSheet(f"color: {theme_manager.colors['good']};")
            elif trend.startswith("↓"):
                trend_label.setStyleSheet(f"color: {theme_manager.colors['danger']};")
            else:
                trend_label.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
            value_layout.addStretch()
            value_layout.addWidget(trend_label)
        else:
            value_layout.addStretch()
            
        layout.addLayout(value_layout)
    
    def refresh_theme(self):
        c = theme_manager.colors
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {c['bg_card']};
                border-radius: 14px;
                border: none;
            }}
        """)
        self.value_label.setStyleSheet(f"color: {c['text_primary']};")


class StatsWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Аналитика")
        self.resize(1300, 850)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {theme_manager.colors['bg_main']};
            }}
            QLabel {{
                font-family: 'Segoe UI', sans-serif;
            }}
            QDateTimeEdit {{
                background-color: {theme_manager.colors['bg_input']};
                border: 1px solid {theme_manager.colors['border']};
                border-radius: 8px;
                padding: 10px 14px;
                font-size: 13px;
                color: {theme_manager.colors['text_primary']};
            }}
            QDateTimeEdit:hover {{
                border-color: {theme_manager.colors['border_light']};
            }}
            QDateTimeEdit:focus {{
                border-color: {theme_manager.colors['accent']};
            }}
            QDateTimeEdit::drop-down {{
                border: none;
                width: 24px;
            }}
            QDateTimeEdit::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid {theme_manager.colors['text_muted']};
            }}
            QScrollBar:vertical {{
                background: {theme_manager.colors['bg_card']};
                width: 6px;
                border-radius: 3px;
            }}
            QScrollBar::handle:vertical {{
                background: {theme_manager.colors['border_light']};
                border-radius: 3px;
                min-height: 20px;
            }}
        """)

        self.engine = AnalyticsEngine()

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(28, 28, 28, 28)
        main_layout.setSpacing(24)

        left_panel = QFrame()
        left_panel.setStyleSheet("background-color: transparent;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(20)

        header = QFrame()
        header.setStyleSheet("background-color: transparent;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        title = QLabel("◈  Аналитика")
        title.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {theme_manager.colors['text_primary']}; letter-spacing: 0.5px;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        left_layout.addWidget(header)

        stats_cards = QHBoxLayout()
        stats_cards.setSpacing(16)
        
        self.card_records = ModernStatsCard("Всего записей", "0")
        stats_cards.addWidget(self.card_records)
        
        self.card_yawns = ModernStatsCard("Зевков", "0", " ↓")
        stats_cards.addWidget(self.card_yawns)
        
        self.card_posture = ModernStatsCard("Плохая осанка", "0%", "")
        stats_cards.addWidget(self.card_posture)
        
        self.card_attention = ModernStatsCard("Ср. внимание", "0%", "")
        stats_cards.addWidget(self.card_attention)
        
        left_layout.addLayout(stats_cards)

        chart_container = QFrame()
        chart_container.setStyleSheet(f"""
            QFrame {{
                background-color: {theme_manager.colors['bg_card']};
                border-radius: 16px;
                border: none;
            }}
        """)
        chart_layout = QVBoxLayout(chart_container)
        chart_layout.setContentsMargins(24, 24, 24, 24)
        chart_layout.setSpacing(16)
        
        chart_header = QLabel("Динамика показателей")
        chart_header.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        chart_header.setStyleSheet(f"color: {theme_manager.colors['text_primary']};")
        chart_layout.addWidget(chart_header)
        
        self.canvas = MplCanvas(self, width=7, height=6, dpi=100)
        self.canvas.setStyleSheet(f"""
            QWidget {{
                background-color: {theme_manager.colors['bg_card']};
                border-radius: 12px;
                border: none;
            }}
        """)
        chart_layout.addWidget(self.canvas)

        left_layout.addWidget(chart_container, stretch=1)

        right_panel = QFrame()
        right_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {theme_manager.colors['bg_card']};
                border-radius: 16px;
                border: none;
            }}
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(24, 24, 24, 24)
        right_layout.setSpacing(20)

        period_header = QLabel("Период")
        period_header.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        period_header.setStyleSheet(f"color: {theme_manager.colors['text_muted']}; letter-spacing: 0.8px;")
        right_layout.addWidget(period_header)

        start_frame = QFrame()
        start_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {theme_manager.colors['bg_input']};
                border-radius: 12px;
                border: none;
            }}
        """)
        start_layout = QVBoxLayout(start_frame)
        start_layout.setContentsMargins(16, 12, 16, 12)
        
        start_label = QLabel("От")
        start_label.setFont(QFont("Segoe UI", 11))
        start_label.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
        start_layout.addWidget(start_label)
        
        self.dt_start = QDateTimeEdit(QDateTime.currentDateTime().addSecs(-3600))
        self.dt_start.setDisplayFormat("dd.MM.yyyy  HH:mm")
        self.dt_start.setCalendarPopup(True)
        self.dt_start.setFixedHeight(40)
        start_layout.addWidget(self.dt_start)
        
        right_layout.addWidget(start_frame)

        end_frame = QFrame()
        end_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {theme_manager.colors['bg_input']};
                border-radius: 12px;
                border: none;
            }}
        """)
        end_layout = QVBoxLayout(end_frame)
        end_layout.setContentsMargins(16, 12, 16, 12)
        
        end_label = QLabel("До")
        end_label.setFont(QFont("Segoe UI", 11))
        end_label.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
        end_layout.addWidget(end_label)
        
        self.dt_end = QDateTimeEdit(QDateTime.currentDateTime())
        self.dt_end.setDisplayFormat("dd.MM.yyyy  HH:mm")
        self.dt_end.setCalendarPopup(True)
        self.dt_end.setFixedHeight(40)
        end_layout.addWidget(self.dt_end)
        
        right_layout.addWidget(end_frame)

        self.btn_apply = QPushButton("Обновить данные")
        self.btn_apply.setStyleSheet(f"""
            QPushButton {{
                background-color: {theme_manager.colors['accent']};
                color: #FFFFFF;
                border: none;
                border-radius: 12px;
                padding: 14px 20px;
                font-size: 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {theme_manager.colors['accent_hover']};
            }}
            QPushButton:pressed {{
                opacity: 0.85;
            }}
        """)
        self.btn_apply.setFixedHeight(48)
        self.btn_apply.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_apply.clicked.connect(self.plot_data)
        right_layout.addWidget(self.btn_apply)
        
        self.btn_export = QPushButton("📥 Экспорт CSV")
        self.btn_export.setStyleSheet(f"""
            QPushButton {{
                background-color: {theme_manager.colors['bg_input']};
                color: {theme_manager.colors['text_secondary']};
                border: 1px solid {theme_manager.colors['border']};
                border-radius: 12px;
                padding: 12px 20px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                border-color: {theme_manager.colors['accent']};
                color: {theme_manager.colors['accent']};
            }}
        """)
        self.btn_export.setFixedHeight(44)
        self.btn_export.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_export.clicked.connect(self.export_data)
        right_layout.addWidget(self.btn_export)

        summary_card = QFrame()
        summary_card.setStyleSheet(f"""
            QFrame {{
                background-color: {theme_manager.colors['bg_input']};
                border-radius: 14px;
                border: none;
            }}
        """)
        summary_layout = QVBoxLayout(summary_card)
        summary_layout.setContentsMargins(18, 16, 18, 16)
        summary_layout.setSpacing(10)
        
        summary_title = QLabel("Сводка")
        summary_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        summary_title.setStyleSheet(f"color: {theme_manager.colors['text_muted']};")
        summary_layout.addWidget(summary_title)
        
        self.summary_text = QLabel("Выберите период и нажмите кнопку для загрузки данных.")
        self.summary_text.setWordWrap(True)
        self.summary_text.setStyleSheet(f"color: {theme_manager.colors['text_muted']}; font-size: 12px; line-height: 1.6;")
        summary_layout.addWidget(self.summary_text)
        
        right_layout.addWidget(summary_card)

        right_layout.addStretch()

        btn_close = QPushButton("Закрыть")
        btn_close.setStyleSheet(f"""
            QPushButton {{
                background-color: {theme_manager.colors['bg_card']};
                color: {theme_manager.colors['text_secondary']};
                border: 1px solid {theme_manager.colors['border']};
                border-radius: 12px;
                padding: 14px 20px;
                font-size: 14px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {theme_manager.colors['bg_input']};
                border-color: {theme_manager.colors['border_light']};
            }}
        """)
        btn_close.setFixedHeight(48)
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close.clicked.connect(self.close)
        right_layout.addWidget(btn_close)

        main_layout.addWidget(left_panel, stretch=3)
        main_layout.addWidget(right_panel, stretch=1)

        self.plot_data()

    def plot_data(self):
        start_dt = self.dt_start.dateTime().toPyDateTime()
        end_dt = self.dt_end.dateTime().toPyDateTime()

        df = self.engine.load_data(start_dt, end_dt)

        if df is None or df.empty:
            self._show_empty_state("Нет данных за выбранный период")
            return

        ax1, ax2 = self.canvas.axes
        ax1.clear()
        ax2.clear()

        time_fmt = mdates.DateFormatter('%H:%M')

        if len(df) > 10:
            df['pitch_smooth'] = df['pitch'].rolling(window=5, min_periods=1).mean()
        else:
            df['pitch_smooth'] = df['pitch']

        ax1.plot(df['timestamp'], df['pitch_smooth'], color=get_chart_colors()['posture_line'], 
                 linewidth=2.5, label='Наклон головы', alpha=0.9)
        ax1.axhline(y=10, color=get_chart_colors()['threshold'], linestyle='--', 
                    alpha=0.7, linewidth=1.5, label='Порог')

        ax1.fill_between(df['timestamp'], df['pitch_smooth'], 10, 
                         where=(df['pitch_smooth'] < 10),
                         facecolor='#F87171', alpha=0.3, interpolate=True)
        
        ax1.fill_between(df['timestamp'], df['pitch_smooth'], 
                         df['pitch_smooth'].max() + 5,
                         where=(df['pitch_smooth'] >= 10),
                         facecolor='#4ADE80', alpha=0.2, interpolate=True)

        ax1.set_title('Осанка', color=get_chart_colors()['text'], fontsize=13, fontweight='600', pad=14, loc='left')
        ax1.set_ylabel('Градусы', color=get_chart_colors()['text_secondary'], fontsize=10, labelpad=8)
        ax1.set_facecolor(get_chart_colors()['background'])
        ax1.grid(True, linestyle='-', alpha=0.3, color=get_chart_colors()['grid'], zorder=0)
        ax1.xaxis.set_major_formatter(time_fmt)
        ax1.tick_params(colors=get_chart_colors()['text_secondary'], labelsize=9)
        
        for spine in ax1.spines.values():
            spine.set_visible(False)
        ax1.spines['bottom'].set_visible(True)
        ax1.spines['bottom'].set_color(get_chart_colors()['grid'])
        ax1.legend(loc='upper right', frameon=True, facecolor=get_chart_colors()['background'], 
                   edgecolor=get_chart_colors()['grid'], labelcolor=get_chart_colors()['text_secondary'],
                   fontsize=9, framealpha=1)

        if len(df) > 10:
            df['ear_smooth'] = df['ear'].rolling(window=10, min_periods=1).mean()
        else:
            df['ear_smooth'] = df['ear']

        ax2.plot(df['timestamp'], df['ear_smooth'], color=get_chart_colors()['ear_line'], 
                 linewidth=2.5, label='EAR (глаза)', alpha=0.9)

        yawns = df[df['fatigue_status'] == 'Yawning']
        if not yawns.empty:
            ax2.scatter(yawns['timestamp'], yawns['ear'], color='#F87171', 
                        s=60, marker='o', label='Зевок', zorder=5, alpha=0.8, edgecolors='#1E1E25', linewidths=1)

        closed_eyes = df[df['fatigue_status'] == 'Eyes Closed']
        if not closed_eyes.empty:
            ax2.scatter(closed_eyes['timestamp'], closed_eyes['ear'], color='#FBBF24', 
                        s=60, marker='x', label='Закрытые глаза', zorder=5, alpha=0.8)

        ax2.set_title('Усталость', color=get_chart_colors()['text'], fontsize=13, fontweight='600', pad=14, loc='left')
        ax2.set_ylabel('EAR', color=get_chart_colors()['text_secondary'], fontsize=10, labelpad=8)
        ax2.set_xlabel('Время', color=get_chart_colors()['text_secondary'], fontsize=10, labelpad=8)
        ax2.set_facecolor(get_chart_colors()['background'])
        ax2.grid(True, linestyle='-', alpha=0.3, color=get_chart_colors()['grid'], zorder=0)
        ax2.xaxis.set_major_formatter(time_fmt)
        ax2.tick_params(colors=get_chart_colors()['text_secondary'], labelsize=9)
        
        for spine in ax2.spines.values():
            spine.set_visible(False)
        ax2.spines['bottom'].set_visible(True)
        ax2.spines['bottom'].set_color(get_chart_colors()['grid'])
        ax2.legend(loc='upper right', frameon=True, facecolor=get_chart_colors()['background'], 
                   edgecolor=get_chart_colors()['grid'], labelcolor=get_chart_colors()['text_secondary'],
                   fontsize=9, framealpha=1)

        self.canvas.fig.tight_layout(pad=2.0)
        self.canvas.draw()

        total_records = len(df)
        bad_posture_count = len(df[df['posture_status'] == 'Bad Posture'])
        posture_percent = (bad_posture_count / total_records * 100) if total_records > 0 else 0
        
        avg_attention = 100
        if len(df) > 0:
            avg_ear = df['ear'].mean()
            avg_attention = max(0, min(100, int((avg_ear - 0.15) / (0.35 - 0.15) * 100)))

        self._update_cards(total_records, len(yawns), posture_percent, avg_attention, start_dt, end_dt)

    def _show_empty_state(self, message):
        ax1, ax2 = self.canvas.axes
        for ax in [ax1, ax2]:
            ax.clear()
            ax.set_facecolor('#1E1E25')
            ax.text(0.5, 0.5, message, transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14, 
                   color='#6A6A7A',
                   fontfamily='Segoe UI')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        self.canvas.draw()
        self.summary_text.setText("Нет данных за выбранный период.")
        self._update_cards(0, 0, 0, 0, None, None)

    def export_data(self):
        from PyQt6.QtWidgets import QMessageBox
        from neurofocus.utils.data_export import DataExporter
        
        start_dt = self.dt_start.dateTime().toPyDateTime()
        end_dt = self.dt_end.dateTime().toPyDateTime()
        
        exporter = DataExporter()
        filepath, message = exporter.export_to_csv(start_dt, end_dt)
        
        msg = QMessageBox(self)
        msg.setWindowTitle("Экспорт данных")
        msg.setStyleSheet(f"""
            QMessageBox {{
                background-color: {theme_manager.colors['bg_card']};
                color: {theme_manager.colors['text_primary']};
            }}
            QPushButton {{
                background-color: {theme_manager.colors['accent']};
                color: #FFFFFF;
                border: none;
                border-radius: 8px;
                padding: 8px 20px;
            }}
        """)
        
        if filepath:
            msg.setText(f"✓ {message}\n\nФайл: {filepath}")
            msg.setIcon(QMessageBox.Icon.Information)
        else:
            msg.setText(f"✗ {message}")
            msg.setIcon(QMessageBox.Icon.Warning)
        
        msg.exec()
    
    def _update_cards(self, records, yawns, posture_percent, attention, start_dt, end_dt):
        self.card_records.value_label.setText(f"{records}")
        self.card_yawns.value_label.setText(f"{yawns}")
        self.card_posture.value_label.setText(f"{posture_percent:.1f}%")
        self.card_attention.value_label.setText(f"{attention}%")
        
        if posture_percent > 30:
            self.card_posture.value_label.setStyleSheet(f"color: {theme_manager.colors['danger']};")
        elif posture_percent > 10:
            self.card_posture.value_label.setStyleSheet(f"color: {theme_manager.colors['warning']};")
        else:
            self.card_posture.value_label.setStyleSheet(f"color: {theme_manager.colors['good']};")
            
        if attention >= 80:
            self.card_attention.value_label.setStyleSheet(f"color: {theme_manager.colors['good']};")
        elif attention >= 50:
            self.card_attention.value_label.setStyleSheet(f"color: {theme_manager.colors['warning']};")
        else:
            self.card_attention.value_label.setStyleSheet(f"color: {theme_manager.colors['danger']};")

        if start_dt and end_dt:
            summary = f"""
            <div style="line-height: 1.7;">
            <b style="color: {theme_manager.colors['text_muted']};">Период анализа:</b><br>
            <span style="color: {theme_manager.colors['text_secondary']};">{start_dt.strftime('%H:%M:%S')} — {end_dt.strftime('%H:%M:%S')}</span>
            </div>
            """
            self.summary_text.setText(summary)
    
    def refresh_theme(self):
        c = theme_manager.colors
        chart_colors = get_chart_colors()
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {c['bg_main']};
            }}
            QLabel {{
                font-family: 'Segoe UI', sans-serif;
            }}
            QDateTimeEdit {{
                background-color: {c['bg_input']};
                border: 1px solid {c['border']};
                border-radius: 8px;
                padding: 10px 14px;
                font-size: 13px;
                color: {c['text_primary']};
            }}
            QDateTimeEdit:hover {{
                border-color: {c['border_light']};
            }}
            QDateTimeEdit:focus {{
                border-color: {c['accent']};
            }}
            QDateTimeEdit::drop-down {{
                border: none;
                width: 24px;
            }}
            QDateTimeEdit::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid {c['text_muted']};
            }}
            QScrollBar:vertical {{
                background: {c['bg_card']};
                width: 6px;
                border-radius: 3px;
            }}
            QScrollBar::handle:vertical {{
                background: {c['border_light']};
                border-radius: 3px;
                min-height: 20px;
            }}
        """)
        if hasattr(self, 'card_records'):
            for card in [self.card_records, self.card_yawns, self.card_posture, self.card_attention]:
                card.refresh_theme()
        self.plot_data()

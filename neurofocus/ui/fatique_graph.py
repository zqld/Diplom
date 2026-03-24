import numpy as np
from collections import deque
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from neurofocus.ui.theme import theme_manager


class MiniFatigueGraph(QWidget):
    def __init__(self, max_points: int = 60, parent=None):
        super().__init__(parent)
        self.max_points = max_points
        self.data_history = deque(maxlen=max_points)
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.figure = Figure(figsize=(3, 1.5), dpi=50)
        self.figure.patch.set_facecolor('none')
        
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setStyleSheet("background: transparent;")
        
        self.ax = self.figure.add_subplot(111)
        
        c = theme_manager.colors
        
        self.ax.set_facecolor('none')
        self.ax.tick_params(colors=c['text_muted'], labelsize=8)
        
        for spine in self.ax.spines.values():
            spine.set_color(c['border'])
        
        self.ax.set_xlim(0, self.max_points)
        self.ax.set_ylim(0, 100)
        self.ax.set_xticks([])
        self.ax.set_yticks([0, 50, 100])
        self.ax.set_ylabel('Усталость', fontsize=7, color=c['text_muted'])
        
        self.line, = self.ax.plot([], [], color=c['accent'], linewidth=1.5)
        self.fill = self.ax.fill_between([], [], 0, alpha=0.3, color=c['accent'])
        
        layout.addWidget(self.canvas)
    
    def update_colors(self):
        return self.refresh_theme()
    
    def refresh_theme(self):
        c = theme_manager.colors
        
        self.figure.patch.set_facecolor('none')
        self.ax.tick_params(colors=c['text_muted'], labelsize=8)
        
        for spine in self.ax.spines.values():
            spine.set_color(c['border'])
        
        self.ax.set_ylabel('Усталость', fontsize=7, color=c['text_muted'])
        
        self.line.set_color(c['accent'])
        
        x = list(range(len(self.data_history)))
        y = list(self.data_history)
        
        if self.fill in self.ax.collections:
            self.fill.remove()
        self.fill = self.ax.fill_between(
            x if x else [0],
            y if y else [0],
            0, alpha=0.3, color=c['accent']
        )
        
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
    
    def update_data(self, fatigue_score: float):
        self.data_history.append(fatigue_score)
        
        x = list(range(len(self.data_history)))
        y = list(self.data_history)
        
        self.line.set_data(x, y)
        
        if len(self.data_history) > 1:
            xs, ys = np.array(x), np.array(y)
            if self.fill in self.ax.collections:
                self.fill.remove()
            self.fill = self.ax.fill_between(xs, ys, 0, alpha=0.3, color=theme_manager.colors['accent'])
        
        self.ax.relim()
        self.ax.autoscale_view()
        
        if len(self.data_history) > 5:
            self.ax.set_xlim(max(0, len(self.data_history) - self.max_points), len(self.data_history))
        
        self.canvas.draw()
    
    def reset(self):
        self.data_history.clear()
        self.line.set_data([], [])
        if self.fill in self.ax.collections:
            self.fill.remove()
        self.fill = self.ax.fill_between([], [], 0, alpha=0.3, color=theme_manager.colors['accent'])
        self.canvas.draw()

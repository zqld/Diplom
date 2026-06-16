from PyQt6.QtWidgets import QApplication

def window_geometry(fraction=0.85):
    screen = QApplication.primaryScreen().geometry()
    w = int(screen.width() * fraction)
    h = int(screen.height() * fraction)
    x = screen.x() + (screen.width() - w) // 2
    y = screen.y() + (screen.height() - h) // 2
    return x, y, w, h

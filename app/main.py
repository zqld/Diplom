"""
NeuroFocus - Main Application Entry Point.

Usage:
    python -m app.main
    or
    python app/main.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurofocus.windows.main_window import MainWindow
from PyQt6.QtWidgets import QApplication
from neurofocus.utils.logger import logger


def main():
    """Main entry point for NeuroFocus application."""
    logger.info("Starting NeuroFocus application")
    
    app = QApplication(sys.argv)
    app.setApplicationName("NeuroFocus")
    app.setOrganizationName("NeuroFocus")
    
    window = MainWindow()
    window.show()
    
    logger.info("Main window displayed")
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

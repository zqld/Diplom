import sys
import os

if getattr(sys, 'frozen', False):
    try:
        os.add_dll_directory(sys._MEIPASS)
        mp_dir = os.path.join(sys._MEIPASS, 'mediapipe', 'python')
        if os.path.isdir(mp_dir):
            os.add_dll_directory(mp_dir)
    except Exception:
        pass

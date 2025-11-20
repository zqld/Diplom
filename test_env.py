#тестовый файл, потом можно удалить
import cv2
import mediapipe as mp
import tensorflow as tf
from PyQt6.QtWidgets import QApplication
import sys
import os

def check_environment():
    print("=== ПРОВЕРКА ОКРУЖЕНИЯ ===")
    
    # 1. Проверка OpenCV
    print(f"OpenCV version: {cv2.__version__}")
    
    # 2. Проверка MediaPipe
    try:
        mp_face_mesh = mp.solutions.face_mesh
        print("MediaPipe: OK")
    except Exception as e:
        print(f"MediaPipe Error: {e}")

    # 3. Проверка TensorFlow и загрузки модели
    model_path = os.path.join("models", "emotion_model.hdf5")
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print(f"Модель эмоций загружена успешно! Входной слой: {model.input_shape}")
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
    else:
        print(f"ФАЙЛ МОДЕЛИ НЕ НАЙДЕН! Проверьте путь: {model_path}")

    # 4. Проверка PyQt6
    try:
        app = QApplication(sys.argv)
        print("PyQt6: OK")
    except Exception as e:
        print(f"PyQt6 Error: {e}")

    print("=== ГОТОВО ===")

if __name__ == "__main__":
    check_environment()
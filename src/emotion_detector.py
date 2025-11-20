#метод preprocess_input — "Подготовка данных"
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class EmotionDetector:
    def __init__(self, model_path):
        """
        Инициализация детектора эмоций.
        :param model_path: Путь к файлу .hdf5
        """
        # Список эмоций согласно датасету FER-2013
        self.EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        
        # Загрузка модели
        try:
            self.model = load_model(model_path, compile=False)
            # Получаем размер входного изображения, который ожидает модель (обычно 64x64)
            self.target_size = self.model.input_shape[1:3] 
            print(f"Emotion Model Loaded. Input shape: {self.target_size}")
        except Exception as e:
            print(f"Error loading emotion model: {e}")
            self.model = None

    def predict_emotion(self, frame, face_landmarks):
        """
        Вырезает лицо, обрабатывает и предсказывает эмоцию.
        """
        if self.model is None:
            return "Error"

        h, w, _ = frame.shape
        
        # 1. Вычисляем Bounding Box (прямоугольник лица) по точкам MediaPipe
        # Берем координаты всех точек
        x_coords = [p.x for p in face_landmarks.landmark]
        y_coords = [p.y for p in face_landmarks.landmark]
        
        # Находим минимум и максимум
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Переводим относительные координаты (0..1) в пиксели
        start_x, end_x = int(min_x * w), int(max_x * w)
        start_y, end_y = int(min_y * h), int(max_y * h)

        # Добавляем отступы (padding), чтобы лицо влезло целиком
        padding_x = int((end_x - start_x) * 0.1)
        padding_y = int((end_y - start_y) * 0.1)
        
        start_x = max(0, start_x - padding_x)
        start_y = max(0, start_y - padding_y)
        end_x = min(w, end_x + padding_x)
        end_y = min(h, end_y + padding_y)

        # Если лицо не найдено или слишком маленькое
        if (end_x - start_x) < 10 or (end_y - start_y) < 10:
            return "No Face"

        # 2. Вырезаем лицо (Region of Interest - ROI)
        face_img = frame[start_y:end_y, start_x:end_x]
        
        # 3. Препроцессинг (как при обучении модели)
        # Перевод в оттенки серого
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Ресайз под вход модели (обычно 64x64)
        face_resized = cv2.resize(face_gray, self.target_size)
        
        # Нормализация (0..255 -> 0..1)
        face_normalized = face_resized.astype("float32") / 255.0
        
        # Добавляем размерности для Keras: (1, 64, 64, 1)
        # 1 - количество картинок (batch size)
        # 1 - количество каналов (чб)
        face_input = np.expand_dims(face_normalized, axis=0)
        face_input = np.expand_dims(face_input, axis=-1)

        # 4. Предсказание
        preds = self.model.predict(face_input, verbose=0)
        
        # Находим индекс эмоции с самой высокой вероятностью
        best_idx = np.argmax(preds)
        emotion = self.EMOTIONS[best_idx]
        confidence = preds[0][best_idx] # Уверенность модели (0..1)

        return emotion, confidence
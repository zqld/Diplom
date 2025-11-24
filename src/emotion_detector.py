#Обрати внимание на метод preprocess_input — это то, о чем нужно будет написать в дипломе в разделе "Подготовка данных".
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque, Counter

class EmotionDetector:
    def __init__(self, model_path):
        """
        Инициализация детектора эмоций со стабилизацией.
        """
        self.EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        
        # --- НАСТРОЙКА СГЛАЖИВАНИЯ ---
        # Храним последние 5 предсказаний. 
        # Если эмоции меняются слишком медленно, уменьши число до 3.
        self.emotion_history = deque(maxlen=5) 
        
        try:
            self.model = load_model(model_path, compile=False)
            self.target_size = self.model.input_shape[1:3] 
            print(f"Emotion Model Loaded. Input shape: {self.target_size}")
        except Exception as e:
            print(f"Error loading emotion model: {e}")
            self.model = None

    def predict_emotion(self, frame, face_landmarks):
        if self.model is None:
            return "Error", 0.0

        h, w, _ = frame.shape
        
        # 1. Координаты лица
        x_coords = [p.x for p in face_landmarks.landmark]
        y_coords = [p.y for p in face_landmarks.landmark]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        start_x, end_x = int(min_x * w), int(max_x * w)
        start_y, end_y = int(min_y * h), int(max_y * h)

        # --- УЛУЧШЕНИЕ 1: Увеличиваем область захвата (Padding) ---
        # Было 0.1 (10%), ставим 0.2 (20%), чтобы в кадр попадали брови и подбородок целиком
        padding_x = int((end_x - start_x) * 0.2)
        padding_y = int((end_y - start_y) * 0.2)
        
        start_x = max(0, start_x - padding_x)
        start_y = max(0, start_y - padding_y)
        end_x = min(w, end_x + padding_x)
        end_y = min(h, end_y + padding_y)

        if (end_x - start_x) < 10 or (end_y - start_y) < 10:
            return "No Face", 0.0

        # 2. Обработка изображения
        face_img = frame[start_y:end_y, start_x:end_x]
        
        try:
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, self.target_size)
            face_normalized = face_resized.astype("float32") / 255.0
            face_input = np.expand_dims(face_normalized, axis=0)
            face_input = np.expand_dims(face_input, axis=-1)

            # 3. Предсказание нейросети
            preds = self.model.predict(face_input, verbose=0)
            
            best_idx = np.argmax(preds)
            current_emotion = self.EMOTIONS[best_idx]
            confidence = preds[0][best_idx]

            # --- УЛУЧШЕНИЕ 2: Алгоритм голосования (Voting) ---
            self.emotion_history.append(current_emotion)
            
            # Считаем самую частую эмоцию в истории
            # Например: [Happy, Neutral, Happy, Happy, Neutral] -> Победил Happy
            most_common = Counter(self.emotion_history).most_common(1)[0][0]
            
            return most_common, confidence

        except Exception as e:
            print(f"Prediction Error: {e}")
            return "Error", 0.0
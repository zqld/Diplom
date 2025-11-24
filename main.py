import cv2
import os
import time
from src.face_core import FaceMeshDetector
from src.geometry import calculate_ear, calculate_mar # Импортируем нашу математику
from src.emotion_detector import EmotionDetector
from src.database import DatabaseManager # Импортируем менеджер базы данных

# --- КОНСТАНТЫ (Настроены под лицо) ---
EAR_THRESHOLD = 0.25  # Порог закрытия глаз
MAR_THRESHOLD = 0.60  # Порог зевания
EMOTION_UPDATE_RATE = 10 # Обновлять эмоции каждые N кадров
DB_SAVE_INTERVAL = 1.0 # Сохранять данные раз в 1 секунду

def main():
    # Инициализируем камеру (0 - обычно встроенная веб-камера)
    cap = cv2.VideoCapture(0)
    
    # Создаем экземпляр нашего детектора
    detector = FaceMeshDetector()

    model_path = os.path.join("models", "emotion_model.hdf5")
    emotion_ai = EmotionDetector(model_path)

    # Инициализация БД
    db = DatabaseManager("session_data.db") # Файл будет в data/session_data.db
    print("База данных подключена.")

    frame_counter = 0
    current_emotion = "Neutral" # Значение по умолчанию
    current_confidence = 0.0

    # Таймер для сохранения в БД
    last_save_time = time.time()

    print("Запуск системы... Нажмите 'q' для выхода.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Не удалось получить кадр с камеры.")
            break

        # Оптимизация: зеркалим кадр, чтобы было как в зеркале (удобнее пользователю)
        #frame = cv2.flip(frame, 1)

        # Обрабатываем кадр через наш класс
        # image - картинка с нарисованной сеткой
        # results - данные с координатами точек (пока не используем, но они есть)
        image, results = detector.process_frame(frame, draw=True)

                # Если лицо найдено
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0] # Объект landmarks
            landmark_points = landmarks.landmark # Список точек
            
            # ---- Логика диагностики (визуальная проверка) ----

            # --- МАТЕМАТИКА (Геометрический анализ (Работает каждый кадр)) ---
            ear = calculate_ear(landmark_points) # Глаза
            mar = calculate_mar(landmark_points) # Рот

            # Определяем статус усталости для БД
            fatigue_status = "Normal"
            if ear < EAR_THRESHOLD: # Порог закрытия глаз
                fatigue_status = "Eyes Closed"
                cv2.putText(image, "EYES CLOSED", (30, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            if mar > MAR_THRESHOLD: # Порог зевания
                fatigue_status = "Yawning"
                cv2.putText(image, "YAWNING", (30, 160), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
            # 2. Анализ эмоций (Работает раз в N кадров)
            # Это экономит ресурсы ПК
            if frame_counter % EMOTION_UPDATE_RATE == 0:
                try:
                    # Передаем оригинальный frame (чистый), и координаты точек
                    emotion, conf = emotion_ai.predict_emotion(frame, landmarks)
                    if emotion not in ["Error", "No Face"]:
                        current_emotion = emotion
                        current_confidence = conf
                except Exception as e:
                    print(f"AI Error: {e}")

            # --- СОХРАНЕНИЕ В БД (Раз в секунду) ---
            current_time = time.time()
            if current_time - last_save_time > DB_SAVE_INTERVAL:
                # Пишем в фоновом режиме, чтобы не тормозить видео
                # Для диплома можно писать синхронно (просто вызов функции), SQLite быстрый
                db.save_log(ear, mar, current_emotion, fatigue_status)
                last_save_time = current_time
                # Можно вывести в консоль точку, чтобы видеть, что запись идет
                print(".", end="", flush=True) 


            # --- ВЫВОД НА ЭКРАН (HUD) ---
            # Рисуем плашки с текстом прямо на видео
            
            # EAR (Глаза)
            cv2.putText(image, f"EAR: {ear:.2f}", (30, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # MAR (Рот)
            cv2.putText(image, f"MAR: {mar:.2f}", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Блок эмоций (справа сверху)
            emo_color = (255, 255, 0) if current_emotion in ["Happy", "Neutral"] else (0, 0, 255)
            cv2.putText(image, f"Emotion: {current_emotion}", (400, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, emo_color, 2)
#            cv2.putText(image, f"Conf: {current_confidence:.1%}", (400, 70), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, emo_color, 1)
            
        frame_counter += 1

        # Показываем результат
        cv2.imshow('Diploma Face Analysis System', image)

        # Выход по кнопке 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
#тестовый файл, потом можно удалить
from src.database import DatabaseManager, FaceLog

db = DatabaseManager("session_data.db")
session = db.Session()

# Получить последние 5 записей
logs = session.query(FaceLog).order_by(FaceLog.id.desc()).limit(50).all()

print(f"Всего записей в базе: {session.query(FaceLog).count()}")
print("--- Последние записи ---")
for log in logs:
    print(f"Time: {log.timestamp.strftime('%H:%M:%S')} | EAR: {log.ear:.2f} | Emotion: {log.emotion} | Status: {log.fatigue_status} | Posture: {log.posture_status}")
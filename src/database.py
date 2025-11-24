from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Создаем базовый класс для моделей
Base = declarative_base()

class FaceLog(Base):
    """
    Модель данных для одной записи анализа состояния.
    Соответствует строке в таблице 'face_logs'.
    """
    __tablename__ = 'face_logs'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now) # Время записи
    ear = Column(Float)         # Уровень открытости глаз
    mar = Column(Float)         # Уровень открытости рта
    pitch = Column(Float)       # Угол наклона головы
    emotion = Column(String)    # Эмоция
    fatigue_status = Column(String) # Текстовый статус (Норма/Сон/Зевок)
    posture_status = Column(String) # Статус осанки (Норма/Плохая)

    def __repr__(self):
        return f"<Log {self.timestamp} | Emo: {self.emotion} | P: {self.posture_status}>"
        #return f"<Log(time={self.timestamp}, emotion={self.emotion}, status={self.fatigue_status})>"

class DatabaseManager:
    def __init__(self, db_name="face_analysis.db"):
        """
        Менеджер базы данных. Создает файл БД в папке data/.
        """
        # Создаем папку data, если её нет
        if not os.path.exists("data"):
            os.makedirs("data")
            
        db_path = os.path.join("data", db_name)
        # Подключение (sqlite /// путь_к_файлу)
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        
        # Создание таблиц (если их нет)
        Base.metadata.create_all(self.engine)
        
        # Фабрика сессий
        self.Session = sessionmaker(bind=self.engine)

    def save_log(self, ear, mar, pitch, emotion, fatigue_status, posture_status):
        """Сохраняет одну запись в БД."""
        session = self.Session()
        try:
            log_entry = FaceLog(
                ear=ear, 
                mar=mar, 
                pitch=pitch,
                emotion=emotion,
                fatigue_status=fatigue_status,
                posture_status=posture_status
            )
            session.add(log_entry)
            session.commit()
        except Exception as e:
            print(f"Ошибка БД: {e}")
            session.rollback()
        finally:
            session.close()
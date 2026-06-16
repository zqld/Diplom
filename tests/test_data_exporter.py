import os
import csv
import sqlite3
import tempfile
from datetime import datetime, timedelta
import pytest
import pandas as pd
from src.data_exporter import DataExporter


@pytest.fixture
def db_with_data():
    """Создаёт временную БД с таблицей face_logs и тестовыми данными."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE face_logs (
            timestamp TEXT,
            ear REAL,
            mar REAL,
            pitch REAL,
            fatigue_status TEXT,
            posture_status TEXT
        )
    """)
    now = datetime.now()
    for i in range(20):
        ts = (now - timedelta(minutes=20 - i)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            "INSERT INTO face_logs VALUES (?, ?, ?, ?, ?, ?)",
            (ts, 0.28, 0.15, 5.0, "Awake", "Good")
        )
    conn.commit()
    conn.close()

    yield db_path
    os.unlink(db_path)


@pytest.fixture
def db_empty():
    """БД с таблицей face_logs, но без данных."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE face_logs (
            timestamp TEXT,
            ear REAL,
            mar REAL,
            pitch REAL,
            fatigue_status TEXT,
            posture_status TEXT
        )
    """)
    conn.commit()
    conn.close()

    yield db_path
    os.unlink(db_path)


class TestDataExporter:
    def test_export_all_data(self, db_with_data):
        exporter = DataExporter(db_path=db_with_data)
        filepath, message = exporter.export_to_csv()
        assert filepath is not None
        assert os.path.exists(filepath)
        df = pd.read_csv(filepath)
        assert len(df) == 20
        os.unlink(filepath)

    def test_export_with_date_filter(self, db_with_data):
        exporter = DataExporter(db_path=db_with_data)
        now = datetime.now()
        start = now - timedelta(minutes=10)
        end = now
        filepath, message = exporter.export_to_csv(start_date=start, end_date=end)
        assert filepath is not None
        assert os.path.exists(filepath)
        df = pd.read_csv(filepath)
        assert 1 <= len(df) <= 11  # последние ~10 минут
        os.unlink(filepath)

    def test_export_with_custom_filepath(self, db_with_data):
        exporter = DataExporter(db_path=db_with_data)
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        custom_path = tmp.name
        tmp.close()

        filepath, message = exporter.export_to_csv(filepath=custom_path)
        assert filepath == custom_path
        assert os.path.exists(custom_path)
        df = pd.read_csv(custom_path)
        assert len(df) == 20
        os.unlink(custom_path)

    def test_export_empty_db_returns_no_data(self, db_empty):
        exporter = DataExporter(db_path=db_empty)
        filepath, message = exporter.export_to_csv()
        assert filepath is None
        assert "Нет данных" in message

    def test_export_nonexistent_db_returns_no_data(self):
        """SQLite создаёт пустой .db при connect(), если папка существует.
        Таблицы face_logs в нём нет — _load_data возвращает None."""
        exporter = DataExporter(db_path="C:/nonexistent_dir/nope.db")
        filepath, message = exporter.export_to_csv()
        assert filepath is None
        assert "Нет данных" in message

    def test_export_wrong_table_name_before_fix(self):
        """Проверяем, что использование правильного имени таблицы работает."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = tmp.name
        tmp.close()

        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE face_logs (timestamp TEXT, ear REAL)
        """)
        conn.execute("INSERT INTO face_logs VALUES (?, ?)",
                     ("2025-01-01 12:00:00", 0.30))
        conn.commit()
        conn.close()

        exporter = DataExporter(db_path=db_path)
        filepath, message = exporter.export_to_csv()
        assert filepath is not None
        assert os.path.exists(filepath)
        df = pd.read_csv(filepath)
        assert len(df) == 1
        os.unlink(filepath)
        os.unlink(db_path)

    def test_export_csv_has_utf8_bom(self, db_with_data):
        """Проверяем, что файл сохраняется с BOM (utf-8-sig)."""
        exporter = DataExporter(db_path=db_with_data)
        filepath, _ = exporter.export_to_csv()
        with open(filepath, "rb") as f:
            raw = f.read(3)
        assert raw == b'\xef\xbb\xbf'  # BOM
        os.unlink(filepath)

    def test_export_respects_filename_param(self, db_with_data):
        """Если передан filename без filepath — сохраняется в exports_dir."""
        exporter = DataExporter(db_path=db_with_data)
        filepath, _ = exporter.export_to_csv(filename="my_custom_name.csv")
        assert filepath is not None
        assert "my_custom_name.csv" in filepath
        assert os.path.exists(filepath)
        os.unlink(filepath)

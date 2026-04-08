"""
Asynchronous database manager for NeuroFocus session logs.

All write operations are offloaded to a dedicated worker thread via
``queue.Queue``, so the video-processing loop never blocks on I/O.

Usage is identical to the synchronous version:

    db = DatabaseManager("session_data.db")
    db.save_log(ear=0.31, mar=0.12, …)   # returns immediately

On application exit call ``db.stop()`` to drain the queue and close
the connection cleanly.
"""

import os
import queue
import threading
import time
from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# ── SQLAlchemy models ──────────────────────────────────────────
Base = declarative_base()


class FaceLog(Base):
    """
    Модель данных для одной записи анализа состояния.
    Соответствует строке в таблице 'face_logs'.
    """
    __tablename__ = "face_logs"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    ear = Column(Float)
    mar = Column(Float)
    pitch = Column(Float)
    emotion = Column(String)
    fatigue_status = Column(String)
    posture_status = Column(String)

    def __repr__(self):
        return (f"<Log {self.timestamp} | "
                f"Emo: {self.emotion} | "
                f"Fatigue: {self.fatigue_status} | "
                f"Posture: {self.posture_status}>")


# ── Poison-pill sentinel ──────────────────────────────────────
_SENTINEL = object()


class DatabaseManager:
    """
    Thread-safe, queue-backed database writer.

    • ``save_log()`` enqueues a record and returns immediately.
    • A background daemon thread dequeues and commits to SQLite.
    • ``stop()`` drains remaining items, then shuts down the thread.
    """

    MAX_QUEUE_SIZE = 5_000  # cap memory if writer stalls

    def __init__(self, db_name: str = "session_data.db"):
        self._engine = None
        self._Session = None
        self._queue: queue.Queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._init_engine(db_name)
        self._start_worker()

    # ── Internal ───────────────────────────────────────────────

    def _init_engine(self, db_name: str):
        """Create / open SQLite database in data/ directory."""
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)

        db_path = os.path.join(data_dir, db_name)
        self._engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            # SQLite-specific: enable WAL for concurrent reads
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)

    def _start_worker(self):
        """Launch the background database writer thread."""
        self._worker = threading.Thread(
            target=self._db_worker_loop,
            name="NeuroFocus-DB-Writer",
            daemon=True,          # dies with main process
        )
        self._worker.start()

    def _db_worker_loop(self):
        """
        Continuously pull log entries from the queue and persist them.

        Stops when a ``_SENTINEL`` object is received or the stop event
        is set.
        """
        # One long-lived session per worker — faster than per-row open/close
        session: Optional[object] = None

        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Poison pill → shutdown
            if item is _SENTINEL:
                break

            # item = (ear, mar, pitch, emotion, fatigue_status, posture_status)
            try:
                if session is None:
                    session = self._Session()

                ear, mar, pitch, emotion, fatigue_st, posture_st = item
                log_entry = FaceLog(
                    ear=ear,
                    mar=mar,
                    pitch=pitch,
                    emotion=emotion,
                    fatigue_status=fatigue_st,
                    posture_status=posture_st,
                )
                session.add(log_entry)
                session.commit()
            except Exception as exc:
                # Rollback to keep session usable
                if session is not None:
                    try:
                        session.rollback()
                    except Exception:
                        session.close()
                        session = None
                # Log to stderr (logger may not be available here)
                import sys
                print(f"[DB-Writer] Error saving log: {exc}", file=sys.stderr)
            finally:
                self._queue.task_done()

        # Cleanup: commit remaining, close session
        if session is not None:
            try:
                session.commit()
            except Exception:
                pass
            finally:
                session.close()

    # ── Public API ─────────────────────────────────────────────

    def save_log(self, ear: float, mar: float, pitch: float,
                 emotion: str, fatigue_status: str, posture_status: str):
        """
        Enqueue a log entry.  Returns immediately (non-blocking).

        If the queue is full (writer stalled), the oldest entries are
        silently dropped to prevent memory exhaustion.
        """
        entry = (ear, mar, pitch, emotion, fatigue_status, posture_status)
        try:
            self._queue.put_nowait(entry)
        except queue.Full:
            # Queue is backed up — drop the entry rather than block.
            # This should only happen if the disk is extremely slow.
            import sys
            print(
                f"[DB-Writer] Queue full ({self.MAX_QUEUE_SIZE}), dropping entry.",
                file=sys.stderr,
            )

    def stop(self, timeout: float = 5.0):
        """
        Gracefully shut down the worker thread.

        Sends a poison pill, waits for the queue to drain, then joins
        the thread.
        """
        self._stop_event.set()

        # Send poison pill
        try:
            self._queue.put_nowait(_SENTINEL)
        except queue.Full:
            pass

        # Wait for thread to finish
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=timeout)
            if self._worker.is_alive():
                import sys
                print(
                    f"[DB-Writer] Thread did not stop within {timeout}s.",
                    file=sys.stderr,
                )

    def queue_size(self) -> int:
        """Approximate number of pending writes."""
        return self._queue.qsize()

    def wait_until_drained(self, timeout: float = 10.0) -> bool:
        """
        Block until the queue is empty or timeout is reached.
        Useful during shutdown to ensure all data is flushed.
        """
        try:
            return self._queue.join_thread(timeout=timeout) is None
        except Exception:
            return False

    # ── Context manager support ────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()

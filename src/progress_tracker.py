import os
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


class ProgressTracker:
    def __init__(self, db_path="data/session_data.db"):
        self.db_path = db_path
        self.progress_file = os.path.join(os.path.dirname(db_path), "progress.json")
        self._ensure_table()
    
    def _ensure_table(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT UNIQUE NOT NULL,
                    total_records INTEGER DEFAULT 0,
                    avg_ear REAL DEFAULT 0,
                    avg_pitch REAL DEFAULT 0,
                    avg_attention REAL DEFAULT 0,
                    fatigue_events INTEGER DEFAULT 0,
                    posture_events INTEGER DEFAULT 0,
                    face_lost_count INTEGER DEFAULT 0,
                    session_duration INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            conn.close()
        except Exception:
            pass
    
    def update_daily_progress(self, date=None):
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            start = f"{date} 00:00:00"
            end = f"{date} 23:59:59"
            
            # Используем правильное имя таблицы - face_logs
            df = pd.read_sql_query(
                "SELECT * FROM face_logs WHERE timestamp BETWEEN ? AND ?",
                conn,
                params=[start, end]
            )
            conn.close()
            
            if df.empty:
                return None
            
            total_records = len(df)
            
            avg_ear = df['ear'].mean() if 'ear' in df.columns else 0
            avg_pitch = df['pitch'].mean() if 'pitch' in df.columns else 0
            
            # Среднее внимание: учитываем fatigue_status (LSTM) + EAR как фолбек.
            # DB хранит статусы с суффиксом модели: "Drowsy [lstm]", "Yawning [lstm]", etc.
            if 'fatigue_status' in df.columns and 'ear' in df.columns:
                scores = []
                for _, row in df.iterrows():
                    status = str(row.get('fatigue_status', '')).lower()
                    if 'sleeping' in status or 'eyes closed' in status:
                        scores.append(15)
                    elif 'drowsy' in status:
                        scores.append(45)
                    elif 'yawning' in status:
                        scores.append(60)
                    else:
                        ear = float(row.get('ear', 0.30))
                        scores.append(max(0, min(100, int((ear - 0.15) / 0.20 * 100))))
                avg_attention = int(sum(scores) / len(scores)) if scores else 0
            elif 'ear' in df.columns:
                avg_attention = max(0, min(100, int((df['ear'].mean() - 0.15) / 0.20 * 100)))
            else:
                avg_attention = 0

            # Статусы усталости: DB хранит "Yawning [lstm]", "Drowsy [lstm]", etc.
            if 'fatigue_status' in df.columns:
                fatigue_events = int(df['fatigue_status'].str.contains(
                    r'Yawning|Eyes Closed|Drowsy|Sleeping|mild|moderate|severe|Засыпает|Усталость',
                    case=False, na=False, regex=True
                ).sum())
            else:
                fatigue_events = 0

            # Осанка: DB хранит "Bad Posture [face_mesh_geometric]", "Fair Posture [...]"
            if 'posture_status' in df.columns:
                posture_events = int(df['posture_status'].str.contains(
                    r'Bad Posture|Fair Posture|Плохая|bad posture',
                    case=False, na=False, regex=True
                ).sum())
            else:
                posture_events = 0
            
            session_duration = total_records
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO daily_progress 
                (date, total_records, avg_ear, avg_pitch, avg_attention, fatigue_events, posture_events, session_duration)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (date, total_records, avg_ear, avg_pitch, avg_attention, fatigue_events, posture_events, session_duration))
            
            conn.commit()
            conn.close()
            
            return {
                'date': date,
                'total_records': total_records,
                'avg_ear': avg_ear,
                'avg_pitch': avg_pitch,
                'avg_attention': avg_attention,
                'fatigue_events': fatigue_events,
                'posture_events': posture_events,
                'session_duration': session_duration
            }
        except Exception as e:
            return None
    
    def get_progress_history(self, days=7):
        try:
            conn = sqlite3.connect(self.db_path)
            
            dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
            dates.reverse()
            
            placeholders = ','.join(['?'] * len(dates))
            query = f"SELECT * FROM daily_progress WHERE date IN ({placeholders}) ORDER BY date ASC"
            
            df = pd.read_sql_query(query, conn, params=dates)
            conn.close()
            
            if df.empty:
                return self._generate_empty_history(dates)
            
            result = []
            for _, row in df.iterrows():
                result.append({
                    'date': row['date'],
                    'avg_attention': row['avg_attention'],
                    'fatigue_events': row['fatigue_events'],
                    'posture_events': row['posture_events'],
                    'total_records': row['total_records']
                })
            
            for date in dates:
                if date not in [r['date'] for r in result]:
                    result.append({
                        'date': date,
                        'avg_attention': 0,
                        'fatigue_events': 0,
                        'posture_events': 0,
                        'total_records': 0
                    })
            
            result.sort(key=lambda x: x['date'])
            
            return result
        except Exception:
            return []
    
    def _generate_empty_history(self, dates):
        return [{'date': d, 'avg_attention': 0, 'fatigue_events': 0, 'posture_events': 0, 'total_records': 0} for d in dates]
    
    def get_weekly_summary(self):
        history = self.get_progress_history(7)
        
        if not history:
            return {
                'avg_attention': 0,
                'total_fatigue': 0,
                'total_posture': 0,
                'trend': 'stable',
                'improvement': 0,
                'active_days': 0
            }
        
        active_days = [h for h in history if h['total_records'] > 0]
        active_count = len(active_days)
        
        if len(active_days) < 2:
            return {
                'avg_attention': active_days[0]['avg_attention'] if active_days else 0,
                'total_fatigue': sum(h['fatigue_events'] for h in active_days),
                'total_posture': sum(h['posture_events'] for h in active_days),
                'trend': 'stable',
                'improvement': 0,
                'active_days': active_count
            }
        
        first_half = active_days[:len(active_days)//2]
        second_half = active_days[len(active_days)//2:]
        
        avg_first = np.mean([d['avg_attention'] for d in first_half]) if first_half else 0
        avg_second = np.mean([d['avg_attention'] for d in second_half]) if second_half else 0
        
        improvement = avg_second - avg_first
        
        if improvement > 5:
            trend = 'improving'
        elif improvement < -5:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'avg_attention': int(np.mean([d['avg_attention'] for d in active_days])),
            'total_fatigue': sum(h['fatigue_events'] for h in history),
            'total_posture': sum(h['posture_events'] for h in history),
            'trend': trend,
            'improvement': round(improvement, 1),
            'active_days': active_count
        }

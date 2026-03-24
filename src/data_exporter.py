import os
import csv
from datetime import datetime
import pandas as pd


class DataExporter:
    def __init__(self, db_path="data/session_data.db"):
        self.db_path = db_path
        self.exports_dir = os.path.join(os.path.dirname(db_path), "exports")
        os.makedirs(self.exports_dir, exist_ok=True)
    
    def export_to_csv(self, start_date=None, end_date=None, filename=None):
        try:
            df = self._load_data(start_date, end_date)
            if df is None or df.empty:
                return None, "Нет данных для экспорта"
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"neurofocus_export_{timestamp}.csv"
            
            filepath = os.path.join(self.exports_dir, filename)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            
            return filepath, f"Экспортировано {len(df)} записей"
        
        except Exception as e:
            return None, f"Ошибка экспорта: {str(e)}"
    
    def export_to_excel(self, start_date=None, end_date=None, filename=None):
        try:
            df = self._load_data(start_date, end_date)
            if df is None or df.empty:
                return None, "Нет данных для экспорта"
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"neurofocus_export_{timestamp}.xlsx"
            
            filepath = os.path.join(self.exports_dir, filename)
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Данные', index=False)
                
                summary = self._generate_summary(df)
                summary.to_excel(writer, sheet_name='Сводка', index=False)
            
            return filepath, f"Экспортировано {len(df)} записей"
        
        except Exception as e:
            return None, f"Ошибка экспорта: {str(e)}"
    
    def _load_data(self, start_date=None, end_date=None):
        try:
            import sqlite3
            
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT * FROM monitoring_log"
            params = []
            
            if start_date:
                query += " WHERE timestamp >= ?"
                params.append(start_date.strftime("%Y-%m-%d %H:%M:%S"))
            
            if end_date:
                if start_date:
                    query += " AND timestamp <= ?"
                else:
                    query += " WHERE timestamp <= ?"
                params.append(end_date.strftime("%Y-%m-%d %H:%M:%S"))
            
            df = pd.read_sql_query(query, conn, params=params if params else None)
            conn.close()
            
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
        
        except Exception:
            return None
    
    def _generate_summary(self, df):
        if df is None or df.empty:
            return pd.DataFrame()
        
        summary_data = {
            'Показатель': [],
            'Значение': []
        }
        
        summary_data['Показатель'].append('Период')
        summary_data['Значение'].append(f"{df['timestamp'].min()} - {df['timestamp'].max()}")
        
        summary_data['Показатель'].append('Всего записей')
        summary_data['Значение'].append(str(len(df)))
        
        if 'ear' in df.columns:
            avg_ear = df['ear'].mean()
            summary_data['Показатель'].append('Средний EAR')
            summary_data['Значение'].append(f"{avg_ear:.3f}")
        
        if 'pitch' in df.columns:
            avg_pitch = df['pitch'].mean()
            summary_data['Показатель'].append('Средний наклон головы')
            summary_data['Значение'].append(f"{avg_pitch:.1f}°")
        
        if 'fatigue_status' in df.columns:
            bad_count = len(df[df['fatigue_status'].isin(['Fatigued', 'Tired', 'Mild'])])
            summary_data['Показатель'].append('Записей с усталостью')
            summary_data['Значение'].append(str(bad_count))
        
        if 'posture_status' in df.columns:
            bad_posture = len(df[df['posture_status'] == 'Bad Posture'])
            summary_data['Показатель'].append('Плохая осанка')
            summary_data['Значение'].append(f"{bad_posture} ({bad_posture/len(df)*100:.1f}%)")
        
        return pd.DataFrame(summary_data)
    
    def get_export_files(self):
        if not os.path.exists(self.exports_dir):
            return []
        
        files = []
        for f in os.listdir(self.exports_dir):
            if f.endswith(('.csv', '.xlsx')):
                filepath = os.path.join(self.exports_dir, f)
                size = os.path.getsize(filepath)
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                files.append({
                    'name': f,
                    'path': filepath,
                    'size': size,
                    'modified': mtime
                })
        
        return sorted(files, key=lambda x: x['modified'], reverse=True)

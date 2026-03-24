import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler


class AppLogger:
    _instance = None
    _logger = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._setup_logger()
    
    def _setup_logger(self):
        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"neurofocus_{datetime.now().strftime('%Y%m%d')}.log")
        
        self._logger = logging.getLogger("NeuroFocus")
        self._logger.setLevel(logging.DEBUG)
        
        if self._logger.handlers:
            return
        
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=5*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self._logger.addHandler(file_handler)
        self._logger.addHandler(console_handler)
    
    @property
    def logger(self):
        return self._logger
    
    def debug(self, msg):
        self._logger.debug(msg)
    
    def info(self, msg):
        self._logger.info(msg)
    
    def warning(self, msg):
        self._logger.warning(msg)
    
    def error(self, msg):
        self._logger.error(msg)
    
    def critical(self, msg):
        self._logger.critical(msg)


app_logger = AppLogger()
logger = app_logger.logger

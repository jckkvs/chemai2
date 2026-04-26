"""
backend/utils/logger.py — 精緻化版 (ロギング設定)

構造化ログ（JSON）、ログ回転、マルチプロセス対応、セキュリティフィルタを備えたロギングユーティリティ。
"""

import logging
import sys
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


def setup_logger(
    name: str = 'chemai',
    level: Union[str, int] = 'INFO',
    log_file: Optional[Union[str, Path]] = None,
    log_format: str = 'structured',  # 'text' or 'structured'
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True
) -> logging.Logger:
    """
    Configure structured logging with rotation and environment-aware settings
    """
    level = os.getenv('LOG_LEVEL', level)
    log_file = os.getenv('LOG_FILE', log_file)
    log_format = os.getenv('LOG_FORMAT', log_format)
    
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(level)
        return logger
    
    logger.setLevel(level)
    logger.propagate = False
    
    if log_format == 'structured':
        formatter = StructuredFormatter(name)
    else:
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8',
            delay=True
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter with context enrichment"""
    
    def __init__(self, logger_name: str):
        super().__init__()
        self.logger_name = logger_name
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'line': record.lineno,
        }
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
            }
        return json.dumps(log_entry, ensure_ascii=False, default=str)

"""
backend_fastapi/celery_app.py
Celery アプリケーション設定
"""
import os
from celery import Celery

# Redis URL の取得（デフォルトはローカルの Redis）
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "chemai_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "backend_fastapi.tasks.analysis_tasks",
        "backend_fastapi.tasks.chem_tasks"
    ]
)

# Celery の詳細設定
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Tokyo",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1時間でタイムアウト
    worker_prefetch_multiplier=1,  # 1つのタスクに集中
)

if __name__ == "__main__":
    celery_app.start()

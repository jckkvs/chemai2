FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install nicegui psutil

COPY backend /app/backend
COPY frontend_nicegui /app/frontend_nicegui

ENV PYTHONPATH=/app

CMD ["python", "frontend_nicegui/main.py"]

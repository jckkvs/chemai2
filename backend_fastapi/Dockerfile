FROM python:3.11-slim

WORKDIR /app

# 依存関係のインストール
COPY backend_fastapi/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend /app/backend
COPY backend_fastapi /app/backend_fastapi

WORKDIR /app/backend_fastapi

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

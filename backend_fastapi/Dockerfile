FROM python:3.11-slim

WORKDIR /app

# 依存関係のインストール
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    pandas==2.1.4 \
    scikit-learn==1.3.2 \
    python-multipart==0.0.6 \
    websockets==12.0

COPY backend /app/backend
COPY backend_fastapi /app/backend_fastapi

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend_fastapi.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

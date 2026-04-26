FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for RDKit and xTB
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire workspace to support local imports
COPY . .

# Expose FastAPI port
EXPOSE 8000

CMD ["python", "backend/api/main.py"]

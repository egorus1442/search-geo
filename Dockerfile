FROM python:3.11-slim

# GDAL и системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libgl1 \
    libglib2.0-0 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /data/index /data/tiles

CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

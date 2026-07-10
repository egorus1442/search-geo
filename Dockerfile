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

# Опциональный нейро-coarse (DINOv2/AnyLoc, COARSE_METHOD=dino|dino_vlad).
# CPU-колёса torch/torchvision из индекса PyTorch (иначе тянется CUDA);
# timm — отдельно (его нет в индексе PyTorch). Отключить: --build-arg WITH_DINO=0.
ARG WITH_DINO=1
RUN if [ "$WITH_DINO" = "1" ]; then \
        pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 \
            --index-url https://download.pytorch.org/whl/cpu && \
        pip install --no-cache-dir "timm>=1.0.0"; \
    fi

COPY . .

RUN mkdir -p /data/index /data/tiles

CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

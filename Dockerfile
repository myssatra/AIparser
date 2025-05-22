# Базовый образ с CUDA 12.8 и cuDNN
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Установка переменных окружения
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Установка необходимых системных пакетов и Python 3.11 из PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Создание виртуального окружения
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Обновление pip в виртуальном окружении
RUN pip install --upgrade pip

# Создание рабочей директории
WORKDIR /app

# Установка PyTorch и связанных библиотек
RUN pip install \
    torch==2.7.0+cu128 \
    torchaudio==2.7.0+cu128 \
    torchvision==0.22.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Копирование файлов проекта (если нужно)
COPY . .

RUN pip install -r requirements.txt

# Команда по умолчанию
CMD ["python", "main.py"]
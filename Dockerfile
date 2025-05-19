# Базовый образ с CUDA 12.8 и cuDNN 9
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Устанавливаем Python и зависимости
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    libpoppler-dev \
    poppler-utils \
    libtesseract-dev \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    ghostscript \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /AIparser

# Копируем requirements.txt
COPY requirements.txt .

RUN pip3 install --no-cache-dir torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# Устанавливаем зависимости Python
RUN pip3 install --no-cache-dir -r requirements.txt

# Копируем остальной код
COPY . .

# Открываем порт для Gradio
EXPOSE 7860

# Запускаем приложение
CMD ["python3", "main.py"]
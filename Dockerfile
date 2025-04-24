FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    python3-dev \
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

WORKDIR /AIparser

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "main.py"]
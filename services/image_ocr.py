# services/image_ocr.py
from PIL import Image
from pathlib import Path
import torch
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

def process_image(input_images):
    """Извлекает текст из списка изображений с использованием surya-ocr."""
    if not input_images:
        return "Ошибка: выберите хотя бы одно изображение."

    # Определяем устройство: GPU (CUDA) или CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Языки для распознавания
    langs = ["ru", "en"]
    print(f"Using languages: {langs}")

    # Инициализация моделей (один раз для всех изображений)
    detection_predictor = DetectionPredictor(device=device)
    recognition_predictor = RecognitionPredictor(device=device)

    results = []

    for input_image in input_images:
        input_path = Path(input_image)
        file_name = input_path.name

        try:
            # Открываем изображение
            image = Image.open(input_image)

            # Извлекаем текст с помощью surya-ocr
            with torch.no_grad():  # Отключаем градиенты для экономии памяти
                predictions = recognition_predictor([image], [langs], detection_predictor)

            # Извлекаем текст из predictions
            text_lines = predictions[0].text_lines
            text = "\n".join([line.text for line in text_lines if line.text])

            if not text.strip():
                results.append("Текст не распознан.")
            else:
                results.append(text)

        except Exception as e:
            results.append(f"Ошибка: {str(e)}")

    # Очистка памяти
    del detection_predictor
    del recognition_predictor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Объединяем результаты в одну строку с переносами
    return "\n\n".join(results)
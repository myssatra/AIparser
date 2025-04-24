# services/pdf_ocr.py
import ocrmypdf
from pathlib import Path
import PyPDF2
from .common import check_tesseract, check_ghostscript

def is_mute_pdf(pdf_path):
    """Проверяет, является ли PDF 'немым' (без встроенного текста) на первых 3 страницах."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf = PyPDF2.PdfReader(file)
            for page in pdf.pages[:3]:
                text = page.extract_text()
                if text and text.strip():
                    return False
            return True
    except Exception:
        return True

def process_pdf(input_pdfs):
    """Обрабатывает список PDF-файлов, возвращает только обработанные файлы и статусы."""
    # Проверяем зависимости
    tesseract_ok, tesseract_error = check_tesseract()
    if not tesseract_ok:
        return [], tesseract_error
    ghostscript_ok, ghostscript_error = check_ghostscript()
    if not ghostscript_ok:
        return [], ghostscript_error

    if not input_pdfs:
        return [], "Ошибка: выберите хотя бы один PDF-файл."

    output_files = []
    statuses = []

    for input_pdf in input_pdfs:
        input_path = Path(input_pdf)
        output_path = input_path.with_stem(f"{input_path.stem}_ocr")
        file_name = input_path.name

        # Проверяем, "немой" ли PDF
        if not is_mute_pdf(input_path):
            statuses.append(f"{file_name}: PDF уже содержит встроенный текст, OCR не требуется.")
            continue

        try:
            # Выполняем OCR
            ocrmypdf.ocr(
                input_file=input_path,
                output_file=output_path,
                language='rus+eng',
                force_ocr=True,
                deskew=True,
                clean=False
            )
            
            output_files.append(str(output_path))
            statuses.append(f"{file_name}: Обработано успешно!")
        
        except Exception as e:
            statuses.append(f"{file_name}: Ошибка: {str(e)}")

    # Объединяем статусы в одну строку с переносами
    status_text = "\n".join(statuses)
    return output_files, status_text
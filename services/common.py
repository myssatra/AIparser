# services/common.py
import shutil

def check_tesseract():
    """Проверяет наличие Tesseract."""
    if not shutil.which("tesseract"):
        return False, "Ошибка: Tesseract не найден. Установите его и добавьте в PATH."
    return True, ""

def check_ghostscript():
    """Проверяет наличие Ghostscript."""
    if not shutil.which("gswin64c") and not shutil.which("gs"):
        return False, "Ошибка: Ghostscript не найден. Установите его и добавьте в PATH."
    return True, ""
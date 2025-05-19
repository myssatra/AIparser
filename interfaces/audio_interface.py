# interfaces/audio_interface.py
import gradio as gr
from services.audio_transcription import transcribe_audio
import os

def build_audio_interface():
    """Создаёт интерфейс для транскрипции аудио и видео с возможностью скачивания результата."""
    with gr.Column():
        gr.Markdown("### Транскрипция аудио и видео")
        gr.Markdown("""
        Загрузите аудио или видео для транскрипции и диаризации (определения спикеров).
        Поддерживаемые форматы: MP3, WAV, AAC, FLAC, MP4, MOV, MKV, AVI.

        **Ограничения для загружаемых файлов:**
        - Максимальный размер файла: 500 МБ.
        - Максимальная длительность аудио/видео: 3 часа.
        
        **Выбор модели Whisper:**

        - Base: Быстрая и компактная, для чёткой речи, но менее точна в сложных условиях.
        - Small: Точнее Base, баланс скорости и качества, для аудио среднего качества.
        - Medium: Высокая точность, для шумных записей и акцентов, но медленнее.
        
        **Поддерживаемые форматы вывода: TXT, DOCX, PDF.**
        """)

        file_input = gr.File(label="Аудио или видео файл", file_types=["audio", "video"], type="filepath")
        model_choice = gr.Dropdown(
            choices=["base", "small", "medium"],
            value="base",
            label="Модель Whisper"
        )
        output_format = gr.Dropdown(
            choices=["txt", "docx", "pdf"],
            value="txt",
            label="Формат вывода"
        )
        use_diarization = gr.Checkbox(label="Использовать диаризацию (определение спикеров)", value=True)
        transcribe_button = gr.Button("Транскрибировать")
        
        output_text = gr.Textbox(label="Результат транскрипции", lines=10)
        output_file = gr.File(label="Скачать транскрипцию")
        log_text = gr.Textbox(label="Логи", lines=5, interactive=False)

        def process_audio(file_path, model_size, output_format, use_diarization):
            if file_path is None:
                return "Ошибка: Пожалуйста, выберите файл.", None, ""
            
            result_lines = []
            file_path_out = None
            log_lines = []
            for result in transcribe_audio(file_path, model_size, output_format, use_diarization):
                if isinstance(result, tuple):
                    file_path_out = result[1]  # Путь к временному файлу
                    result_lines.append(result[0])  # Полный текст
                    break
                elif isinstance(result, str) and "[Лог]" in result:
                    log_lines.append(result)  # Собираем логи
                else:
                    result_lines.append(result)  # Собираем строки транскрипции
            limited_result = "".join(result_lines[:10])  # Ограничиваем до 10 строк
            return limited_result, file_path_out, "\n".join(log_lines)

        transcribe_button.click(
            fn=process_audio,
            inputs=[file_input, model_choice, output_format, use_diarization],
            outputs=[output_text, output_file, log_text]
        )

if __name__ == "__main__":
    interface = gr.Interface(fn=build_audio_interface, title="Транскрипция аудио и видео")
    interface.launch()
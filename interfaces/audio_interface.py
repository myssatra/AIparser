# interfaces/audio_interface.py
import gradio as gr
from services.audio_transcription import transcribe_audio

def build_audio_interface():
    """Создаёт интерфейс для транскрипции аудио с возможностью скачивания результата."""
    with gr.Column():
        gr.Markdown("### Транскрипция аудио")
        gr.Markdown("""
        Загрузите аудиофайл для транскрипции и диаризации (определения спикеров).

        **Ограничения для загружаемых файлов:**
        - Максимальный размер файла: 500 МБ.
        - Максимальная длительность аудио: 3 часа.
        - Поддерживыаемые форматы:  MP4, MP3, WAV, MOV, MKV, AVI, AAC, FLAC.
        
        **Выбор модели Whisper:**

        - Base: Быстрая и компактная, для чёткой речи, но менее точна в сложных условиях.
        - Small: Точнее Base, баланс скорости и качества, для аудио среднего качества.
        - Medium: Высокая точность, для шумных записей и акцентов, но медленнее.
        
        **Поддерживаемые форматы вывода: TXT, DOCX, PDF.**
        """)

        audio_input = gr.Audio(type="numpy", label="Аудиофайл")
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

        def process_audio(audio, model_size, output_format, use_diarization):
            if audio is None:
                return "Ошибка: Пожалуйста, выберите аудиофайл.", None, ""
            result_lines = []
            file_path = None
            log_lines = []
            for result in transcribe_audio(audio, model_size, output_format, use_diarization):
                if isinstance(result, tuple):
                    file_path = result[1]  # Файл
                    result_lines.append(result[0])  # Полный текст
                    break
                elif isinstance(result, str) and "[Лог]" in result:
                    log_lines.append(result)  # Собираем логи
                else:
                    result_lines.append(result)  # Собираем строки транскрипции
            limited_result = "".join(result_lines[:10])  # Ограничиваем до 10 строк
            return limited_result, file_path if file_path else None, "\n".join(log_lines)

        transcribe_button.click(
            fn=process_audio,
            inputs=[audio_input, model_choice, output_format, use_diarization],
            outputs=[output_text, output_file]
        )

if __name__ == "__main__":
    interface = gr.Interface(fn=build_audio_interface, title="Транскрипция аудио")
    interface.launch()
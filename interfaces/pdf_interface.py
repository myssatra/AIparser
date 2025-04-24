# interfaces/pdf_interface.py
import gradio as gr
from services.pdf_ocr import process_pdf

def clear_pdf_inputs():
    """Очищает поле ввода для PDF."""
    return None

def build_pdf_interface():
    """Создает интерфейс для PDF с горизонтальным расположением."""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Ввод")
            pdf_input = gr.File(label="Загрузите PDF файлы", file_types=[".pdf"], file_count="multiple")
            with gr.Row():
                pdf_process_button = gr.Button("Обработать файлы")
                pdf_clear_button = gr.Button("Очистить форму")
        
        with gr.Column(scale=1):
            gr.Markdown("### Результат")
            pdf_output = gr.Files(label="Скачать обработанные PDF")
            pdf_status = gr.Textbox(label="Статус обработки", lines=10)
    
    pdf_process_button.click(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=[pdf_output, pdf_status]
    )
    pdf_clear_button.click(
        fn=clear_pdf_inputs,
        inputs=None,
        outputs=pdf_input
    )
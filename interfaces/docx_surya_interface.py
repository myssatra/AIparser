# interfaces/docx_surya_interface.py
import gradio as gr
from services.pdf_to_docx_surya import process_pdf_to_docx_surya

def clear_docx_surya_inputs():
    """Очищает поле ввода для PDF."""
    return None

def build_docx_surya_interface():
    """Создает интерфейс для преобразования PDF в DOCX с surya-ocr."""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Ввод")
            pdf_input = gr.File(label="Загрузите PDF файлы", file_types=[".pdf"], file_count="multiple")
            with gr.Row():
                pdf_process_button = gr.Button("Обработать файлы")
                pdf_clear_button = gr.Button("Очистить форму")
        
        with gr.Column(scale=1):
            gr.Markdown("### Результат")
            pdf_output = gr.Files(label="Скачать DOCX файлы")
            pdf_status = gr.Textbox(label="Статус обработки", lines=10)
    
    pdf_process_button.click(
        fn=process_pdf_to_docx_surya,
        inputs=pdf_input,
        outputs=[pdf_output, pdf_status]
    )
    pdf_clear_button.click(
        fn=clear_docx_surya_inputs,
        inputs=None,
        outputs=pdf_input
    )
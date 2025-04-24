# interfaces/image_interface.py
import gradio as gr
from services.image_ocr import process_image

def clear_image_inputs():
    """Очищает поле ввода для изображения."""
    return None

def build_image_interface():
    """Создает интерфейс для изображений с горизонтальным расположением."""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Ввод")
            image_input = gr.File(label="Загрузите изображения", file_types=[".jpg", ".jpeg", ".png"], file_count="multiple")
            with gr.Row():
                image_process_button = gr.Button("Обработать изображения")
                image_clear_button = gr.Button("Очистить форму")
        
        with gr.Column(scale=1):
            gr.Markdown("### Результат")
            image_output = gr.Textbox(label="Распознанный текст", lines=10)
    
    image_process_button.click(
        fn=process_image,
        inputs=image_input,
        outputs=image_output
    )
    image_clear_button.click(
        fn=clear_image_inputs,
        inputs=None,
        outputs=image_input
    )
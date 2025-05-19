# main.py
import gradio as gr
from interfaces.audio_interface import build_audio_interface
from interfaces.pdf_interface import build_pdf_interface
from interfaces.image_interface import build_image_interface
from interfaces.docx_interface import build_docx_interface
# from interfaces.docx_surya_interface import build_docx_surya_interface

css = """
.gr-row { gap: 20px; }
.gr-column { padding: 10px; }
.gr-button { margin-right: 10px; }
"""

with gr.Blocks(title="Система Распознавания: Файлы и Аудио", css=css) as app:
    gr.Markdown("## Система Распознавания: Файлы и Аудио")
    gr.Markdown("Выберите вкладку для обработки PDF, извлечения текста из изображений или конвертации PDF в DOCX.")
    
    with gr.Tabs():
        with gr.Tab("OCR для PDF"):
            build_pdf_interface()
        
        with gr.Tab("OCR для изображений"):
            build_image_interface()
        
        with gr.Tab("PDF в DOCX"):
            build_docx_interface()
        
        # with gr.Tab("PDF в DOCX (Surya Test)"):
        #     build_docx_surya_interface()
            
        with gr.Tab("Транскрипция аудио"):
            build_audio_interface()

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
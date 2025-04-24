import gradio as gr

def build_audio_interface():
     with gr.Column():
        gr.Markdown("### Транскрипция и диаризация аудио")
        gr.Markdown("Загрузите аудиофайл (MP3 или WAV) для получения транскрипции с разделением по говорящим.")
        
        audio_input = gr.File(label="Аудиофайл", file_types=[".mp3", ".wav"])
        transcribe_button = gr.Button("Транскрибировать")
        output_text = gr.Textbox(label="Результат", lines=10, interactive=False)
        
        transcribe_button.click(
            # fn=process_audio,
            inputs=audio_input,
            outputs=output_text
        )

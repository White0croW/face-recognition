import gradio as gr
from database import init_db, save_to_db
from image_utils import handle_search, handle_upload, process_single_image, process_zip_archive, search_images

# Gradio-интерфейс [[6]]
with gr.Blocks() as demo:
    gr.Markdown("# Менеджер изображений")

    with gr.Tab("Загрузка"):
        image_input = gr.Image(type="filepath", label="Изображение")
        zip_input = gr.File(label="ZIP-архив")
        output = gr.Textbox(label="Результат")
        process_btn = gr.Button("Обработать")
        process_btn.click(handle_upload, [image_input, zip_input], output)

    with gr.Tab("Поиск"):
        search_query = gr.Textbox(label="Поисковый запрос")
        search_output = gr.Gallery(label="Результаты")
        search_btn = gr.Button("Найти")
        search_btn.click(handle_search, search_query, search_output)

if __name__ == "__main__":
    demo.launch()

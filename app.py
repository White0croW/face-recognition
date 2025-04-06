import gradio as gr
from database import init_db, save_to_db
from image_utils import process_single_image, process_zip_archive, search_images

# Инициализация БД
conn = init_db()


def handle_upload(file, zip_file):
    """Обработка загрузки файлов"""
    if file:
        filename, save_path = process_single_image(file)
        save_to_db(conn, filename, save_path)
        return f"Сохранено: {filename}"
    elif zip_file:
        for filename, save_path in process_zip_archive(zip_file):
            save_to_db(conn, filename, save_path)
        return "Архив обработан"
    return "Файл не выбран"


def handle_search(query):
    """Обработка поиска"""
    results = search_images(query)
    return results if results else "Ничего не найдено"


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

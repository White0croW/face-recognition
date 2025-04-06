from main import handle_upload, handle_search
from PIL import Image
import io
import numpy as np


def test_handle_upload():
    # Тест загрузки изображения [[4]]
    img = Image.new("RGB", (100, 100))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    result = handle_upload(img_byte_arr, None)
    assert "Сохранено" in result


def test_handle_search():
    # Тест поиска изображений [[4]]
    results = handle_search("test")
    assert isinstance(results, list) or "Ничего не найдено" in results

import pytest
from image_utils import handle_upload, handle_search, process_image, validate_image
from PIL import Image
import io
import numpy as np
import os


@pytest.fixture
def sample_image():
    """Фикстура для тестового изображения"""
    img = Image.new("RGB", (100, 100), color=(73, 109, 137))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()


def test_process_image_valid(sample_image):
    """Тест обработки валидного изображения"""
    result = process_image(sample_image)
    assert isinstance(result, np.ndarray)
    assert result.shape == (100, 100, 3)  # Проверка размеров


def test_process_image_invalid():
    """Тест обработки невалидных данных"""
    assert process_image(b'invalid_data') is None


def test_validate_image(sample_image):
    """Тест валидации изображения"""
    img_array = process_image(sample_image)
    assert validate_image(img_array) is True
    assert validate_image(np.zeros((100, 100))) is False  # 2D массив
    assert validate_image(None) is False


def test_handle_upload(tmp_path, sample_image):
    """Тест загрузки изображения с временной директорией"""
    test_dir = tmp_path / "uploads"
    test_dir.mkdir()
    result = handle_upload(sample_image, str(test_dir))
    assert "Сохранено" in result
    assert len(os.listdir(test_dir)) == 1


def test_handle_search(tmp_path, sample_image):
    """Тест поиска изображений"""
    test_dir = tmp_path / "search_test"
    test_dir.mkdir()

    # Создаем тестовые файлы
    (test_dir / "test_image1.png").write_bytes(sample_image)
    (test_dir / "test_image2.png").write_bytes(sample_image)

    # Тестируем поиск
    results = handle_search("test", str(test_dir))
    assert len(results) == 2
    assert "Ничего не найдено" not in results
import pytest
from unittest.mock import MagicMock, patch
from ui import FaceRecognitionUI
from face_service import FaceService


@pytest.fixture
def mock_ui():
    mock_service = MagicMock(spec=FaceService)
    return FaceRecognitionUI(mock_service)


def test_render(mock_ui):
    with patch("ui.st.title") as mock_title:
        mock_ui.render()
        mock_title.assert_called_with("Система распознавания лиц")


def test_upload_valid_image(mock_ui):
    with patch("ui.st.file_uploader") as mock_uploader:
        mock_uploader.return_value = [MagicMock(type="image/png")]
        mock_ui._process_recognition = MagicMock()

        mock_ui._render_recognition()
        mock_ui.service.recognize_face.assert_called()


def test_handle_zip_upload(mock_ui):
    mock_zip = MagicMock()
    mock_zip.namelist.return_value = ["test.jpg"]
    mock_zip.read.return_value = b"image_data"

    with patch("ui.zipfile.ZipFile", return_value=mock_zip):
        result = mock_ui._process_zip(mock_zip)
        assert result == [b"image_data"]

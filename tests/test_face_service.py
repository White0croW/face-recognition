import pytest
from unittest.mock import patch, MagicMock
from face_service import FaceService
from database import SQLiteDB


@pytest.fixture
def mock_service():
    mock_db = MagicMock(spec=SQLiteDB)
    return FaceService(mock_db)


def test_add_image_to_db_valid(mock_service):
    with patch("face_service.cv2.imdecode", return_value=MagicMock()):
        mock_service._validate_image = MagicMock(return_value=True)
        mock_service._process_query_image = MagicMock(
            return_value=[(b"emb1", (10, 20, 30, 40))]
        )

        mock_service.add_image_to_db(b"test_image")
        mock_service.db.save_image.assert_called_once()


def test_add_image_to_db_invalid(mock_service):
    mock_service._validate_image = MagicMock(return_value=False)
    with pytest.raises(ValueError):
        mock_service.add_image_to_db(b"invalid_image")


def test_recognize_face(mock_service):
    mock_service.db.get_all_faces.return_value = [
        {
            "image": b"db_img",
            "embeddings": [b"emb1"],
            "face_locations": [(10, 20, 30, 40)],
        }
    ]
    mock_service._calculate_similarity = MagicMock(return_value=0.7)

    result = mock_service.recognize_face(b"test_image")
    assert len(result) == 1
    assert result[0]["similarity"] == 0.7

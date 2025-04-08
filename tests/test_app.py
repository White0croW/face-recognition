import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from unittest.mock import patch
from app import main


def test_main():
    with patch("app.SQLiteDB") as mock_db:
        with patch("app.FaceService") as mock_service:
            with patch("app.FaceRecognitionUI") as mock_ui:
                main()
                mock_db.assert_called_once_with("faces.db")
                mock_service.assert_called_once_with(mock_db())
                mock_ui.assert_called_once_with(mock_service())
                mock_ui().render.assert_called_once()

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
import sqlite3
from unittest.mock import patch, MagicMock
from database import SQLiteDB


@pytest.fixture
def mock_db():
    with patch("database.sqlite3.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        yield SQLiteDB(":memory:")


def test_create_tables(mock_db):
    mock_db._create_tables()
    mock_db.conn.cursor().execute.assert_called_with(
        "CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY AUTOINCREMENT, embeddings BLOB NOT NULL, face_locations BLOB NOT NULL, image BLOB NOT NULL)"
    )


def test_save_image(mock_db):
    mock_db.save_image(b"test_image", [b"emb1"], [(10, 20, 30, 40)])
    mock_db.conn.cursor().execute.assert_called_with(
        "INSERT INTO faces (embeddings, face_locations, image) VALUES (?, ?, ?)",
        (b"emb1", b"(10, 20, 30, 40)", b"test_image"),
    )


def test_get_all_faces(mock_db):
    mock_cursor = mock_db.conn.cursor()
    mock_cursor.fetchall.return_value = [(1, b"emb1", b"loc1", b"img1")]

    result = mock_db.get_all_faces()
    assert result == [
        {"id": 1, "embeddings": b"emb1", "face_locations": b"loc1", "image": b"img1"}
    ]

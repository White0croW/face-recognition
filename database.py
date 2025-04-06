import sqlite3


def init_db():
    """Инициализация БД [[3]][[5]]"""
    conn = sqlite3.connect("images.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            filename TEXT,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_path TEXT
        )
    """
    )
    conn.commit()
    return conn


def save_to_db(conn, filename, file_path):
    """Сохранение метаданных в БД [[9]]"""
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO images (filename, file_path) VALUES (?, ?)", (filename, file_path)
    )
    conn.commit()

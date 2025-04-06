# database.py (улучшенная структура)
import sqlite3
import numpy as np


def init_db():
    conn = sqlite3.connect("faces.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY,
            filename TEXT,
            file_path TEXT,
            embedding BLOB  # Хранение векторов [[7]]
        )
    """
    )
    conn.commit()
    return conn


def save_to_db(filename, file_path, embeddings):
    conn = sqlite3.connect("faces.db")
    cursor = conn.cursor()
    # Конвертация numpy в bytes [[3]]
    emb_blob = np.array(embeddings).tobytes() if embeddings else None
    cursor.execute(
        """
        INSERT INTO faces (filename, file_path, embedding)
        VALUES (?, ?, ?)
    """,
        (filename, file_path, emb_blob),
    )
    conn.commit()
    conn.close()


def search_images(query_emb, threshold=0.6):
    conn = sqlite3.connect("faces.db")
    cursor = conn.cursor()
    results = []

    # Поиск похожих лиц [[6]]
    cursor.execute("SELECT file_path, embedding FROM faces")
    for row in cursor.fetchall():
        path, emb_blob = row
        if not emb_blob:
            continue
        db_emb = np.frombuffer(emb_blob).reshape(-1)
        similarity = np.dot(query_emb, db_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(db_emb)
        )
        if similarity >= threshold:
            results.append((path, similarity))

    conn.close()
    return sorted(results, key=lambda x: x[1], reverse=True)

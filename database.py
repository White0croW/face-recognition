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
            image BLOB,
            embedding BLOB 
        )
    """
    )
    conn.commit()
    return conn


def save_image(filename, image_blob, embedding):
    conn = sqlite3.connect("faces.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO faces (filename, image, embedding)
        VALUES (?, ?, ?)
    """,
        (filename, image_blob, embedding.tobytes()),
    )
    conn.commit()
    conn.close()


def search_faces(query_emb, threshold=0.7):
    conn = sqlite3.connect("faces.db")
    cursor = cursor = conn.cursor()
    cursor.execute("SELECT filename, image, embedding FROM faces")
    results = []
    for row in cursor.fetchall():
        name, img_blob, emb_blob = row
        db_emb = np.frombuffer(emb_blob)
        similarity = np.dot(query_emb, db_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(db_emb)
        )
        if similarity >= threshold:
            results.append((name, img_blob))
    return results

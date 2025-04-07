import streamlit as st
from image_utils import process_image
from face_model import recognize_face
from database import get_db_connection
import zipfile
import io
import numpy as np
from PIL import Image

# Навигация через боковую панель [[7]]
menu = st.sidebar.selectbox("Меню", ["Заполнение БД", "Распознавание"])

db = get_db_connection()

if menu == "Заполнение БД":
    st.header("Добавление данных в БД")
    uploaded_files = st.file_uploader(
        "Загрузите фото или архив",
        type=["jpg", "png", "zip"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        process_uploaded_files(uploaded_files, db)

elif menu == "Распознавание":
    st.header("Распознавание лиц")
    col1, col2 = st.columns(2)

    with col1:
        # Загрузка фото [[5]]
        uploaded_file = st.file_uploader("Загрузите фото", type=["jpg", "png"])

    with col2:
        # Использование камеры [[5]]
        camera_input = st.camera_input("Сфотографируйтесь")

    if uploaded_file or camera_input:
        image_bytes = (
            uploaded_file.getvalue() if uploaded_file else camera_input.getvalue()
        )
        process_recognition(image_bytes, db)


def process_uploaded_files(files, db):
    """Обработка загруженных файлов для БД."""
    for file in files:
        if file.type == "application/zip":
            with zipfile.ZipFile(file) as z:
                for name in z.namelist():
                    if name.lower().endswith((".jpg", ".png")):
                        image_bytes = z.read(name)
                        process_and_save(image_bytes, db)
        else:
            process_and_save(file.getvalue(), db)
    st.success("Данные добавлены в БД")


def process_and_save(image_bytes, db):
    """Извлечение эмбеддинга и сохранение в БД."""
    embedding = recognize_face(image_bytes)
    if embedding is not None:
        db.insert_embedding(embedding, image_bytes)
    else:
        st.warning("Лицо не обнаружено")


def process_recognition(image_bytes, db):
    """Обработка распознавания."""
    embedding = recognize_face(image_bytes)
    if embedding is not None:
        matches = db.find_similar(embedding)
        st.write(f"Найдено совпадений: {len(matches)}")
        for match in matches:
            st.image(match["image"], caption=f"Сходство: {match['similarity']:.2f}")
    else:
        st.error("Лицо не обнаружено")

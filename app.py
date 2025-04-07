import streamlit as st
from PIL import Image
import zipfile
import os
import numpy as np
from database import init_db, save_image, search_faces
from face_model import FaceRecognizer

# Инициализация
init_db()
recognizer = FaceRecognizer()

st.title("Система распознавания лиц")

# Загрузка фото или архива [[7]]
uploaded_files = st.file_uploader(
    "Загрузите фото или ZIP-архив",
    type=["png", "jpg", "zip"],
    accept_multiple_files=True,
)

if uploaded_files:
    for file in uploaded_files:
        if file.type == "application/zip":
            # Обработка ZIP [[7]]
            with zipfile.ZipFile(file) as z:
                for fname in z.namelist():
                    if fname.lower().endswith((".png", ".jpg")):
                        img = Image.open(z.open(fname)).convert("RGB")
                        embedding = recognizer.get_embedding(img)
                        save_image(fname, img.tobytes(), embedding)
        else:
            # Обработка отдельного изображения [[3]]
            img = Image.open(file).convert("RGB")
            embedding = recognizer.get_embedding(img)
            save_image(file.name, img.tobytes(), embedding)
    st.success("Файлы сохранены в БД")

# Поиск по фото с камеры или загрузки
st.header("Поиск лица")
camera_input = st.camera_input("Сфотографируйтесь")
uploaded_search = st.file_uploader("Или загрузите фото", type=["png", "jpg"])

if camera_input or uploaded_search:
    img = Image.open(camera_input or uploaded_search).convert("RGB")
    query_emb = recognizer.get_embedding(img)
    results = search_faces(query_emb)  # Поиск по embedding [[3]]

    if results:
        st.write(f"Найдено {len(results)} совпадений:")
        cols = st.columns(3)
        for i, (name, img_blob) in enumerate(results):
            with cols[i % 3]:
                st.image(Image.open(io.BytesIO(img_blob)), caption=name)
    else:
        st.warning("Лицо не найдено")

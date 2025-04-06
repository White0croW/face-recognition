# app.py (Streamlit)
import streamlit as st
from PIL import Image
import zipfile
import os
import numpy as np
from database import init_db, save_to_db, search_images
from face_model import FaceRecognizer

# Инициализация [[5]]
init_db()
recognizer = FaceRecognizer()

# Настройки страницы [[8]]
st.set_page_config(page_title="Face Recognition", layout="wide")
st.title("Система распознавания лиц")

# Интерфейс [[6]]
tab1, tab2 = st.tabs(["Загрузка", "Поиск"])

with tab1:
    st.header("Загрузка изображений")
    uploaded_file = st.file_uploader(
        "Выберите изображение", type=["png", "jpg", "jpeg"]
    )
    zip_file = st.file_uploader("Загрузите ZIP-архив", type=["zip"])

    if st.button("Обработать"):
        try:
            if uploaded_file:
                # Обработка отдельного изображения [[3]]
                img = Image.open(uploaded_file).convert("RGB")
                filename = uploaded_file.name
                save_path = os.path.join("uploads", filename)
                os.makedirs("uploads", exist_ok=True)
                img.save(save_path)

                # Распознавание лиц [[6]]
                faces = recognizer.detect_faces(img)
                if not faces:
                    st.warning("Лица не обнаружены")
                else:
                    embeddings = [recognizer.recognize(img.crop(box)) for box in faces]
                    save_to_db(filename, save_path, embeddings)
                    st.success(f"Сохранено: {filename}")

            elif zip_file:
                # Обработка ZIP [[7]]
                with zipfile.ZipFile(zip_file) as z:
                    for fname in z.namelist():
                        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                            save_path = os.path.join("uploads", fname)
                            with open(save_path, "wb") as f:
                                f.write(z.read(fname))
                            save_to_db(fname, save_path, [])
                st.success("Архив обработан")

        except Exception as e:
            st.error(f"Ошибка: {str(e)}")  # [[9]]

with tab2:
    st.header("Поиск по лицам")
    query_image = st.file_uploader(
        "Загрузите образец для поиска", type=["png", "jpg", "jpeg"]
    )

    if query_image and st.button("Найти"):
        try:
            img = Image.open(query_image).convert("RGB")
            faces = recognizer.detect_faces(img)
            if not faces:
                st.warning("Лица не обнаружены в запросе")
            else:
                query_emb = recognizer.recognize(img.crop(faces[0]))
                results = search_images(query_emb)

                # Отображение результатов [[8]]
                cols = st.columns(4)
                for i, (path, score) in enumerate(results[:8]):
                    with cols[i % 4]:
                        st.image(
                            path,
                            caption=f"Совпадение: {score:.1f}%",
                            use_column_width=True,
                        )
        except Exception as e:
            st.error(f"Ошибка поиска: {str(e)}")

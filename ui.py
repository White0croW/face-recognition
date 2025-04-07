import streamlit as st
import zipfile
import numpy as np
import cv2
from io import BytesIO


class FaceRecognitionUI:
    def __init__(self, service):
        self.service = service

    def render(self):
        st.title("Система распознавания лиц")
        menu = st.sidebar.selectbox(
            "Меню", ["Добавить в БД", "Распознать", "Фотографии из БД"]
        )

        if menu == "Добавить в БД":
            self._render_db_upload()
        elif menu == "Распознать":
            self._render_recognition()
        else:
            self._render_photos()

    def _render_db_upload(self):
        st.header("Добавление данных")
        files = st.file_uploader(
            "Выберите файлы/архивы",
            type=["jpg", "png", "zip"],
            accept_multiple_files=True,
        )

        if files:
            total_images = self._count_total_images(files)
            progress = st.progress(0)
            success = 0
            errors = []

            for file in files:
                try:
                    if file.type in ["application/zip", "application/x-zip-compressed"]:
                        images = self._process_zip(file)
                        for img_bytes in images:
                            self.service.add_image_to_db(img_bytes)
                            success += 1
                            progress.progress(success / total_images)
                    else:
                        self.service.add_image_to_db(file.getvalue())
                        success += 1
                        progress.progress(success / total_images)
                except Exception as e:
                    errors.append(str(e))

            progress.empty()
            st.success(f"✅ Загружено изображений: {success}")
            if errors:
                st.error(f"❌ Ошибок: {len(errors)}")
                with st.expander("Подробнее"):
                    st.write("\n".join(errors))

    def _render_recognition(self):
        st.header("Распознавание")
        col1, col2 = st.columns(2)

        with col1:
            uploaded = st.file_uploader("Загрузить фото", type=["jpg", "png"])
        with col2:
            camera = st.camera_input("Использовать камеру")

        if uploaded or camera:
            image_bytes = uploaded.getvalue() if uploaded else camera.getvalue()
            self._process_recognition(image_bytes)

    def _process_recognition(self, image_bytes):
        col1, col2 = st.columns(2)

        # Отображение загруженного изображения слева
        with col1:
            st.subheader("Загруженное изображение:")
            st.image(image_bytes, use_container_width=True)

        # Поиск совпадений
        matches = self.service.recognize_face(image_bytes, threshold=0.6)

        # Отображение результатов справа
        with col2:
            if matches:
                st.subheader("Найденные совпадения:")
                for match in matches:
                    img_with_box = self._draw_face_box(
                        match["image"], match["face_location"]
                    )
                    st.image(
                        img_with_box, caption=f'Сходство: {match["similarity"]:.2f}'
                    )
            else:
                st.error("Совпадений не найдено")

    def _draw_face_box(self, image_bytes, face_location):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        top, left, bottom, right = face_location
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _render_photos(self):
        st.header("Фотографии из БД")
        images = self.service.db.get_all_faces()

        if not images:
            st.info("База данных пуста")
            return

        page_size = 10
        total_pages = (len(images) // page_size) + (1 if len(images) % page_size else 0)
        page = st.number_input("Страница", 1, total_pages, 1)

        start = (page - 1) * page_size
        end = start + page_size

        for img_info in images[start:end]:
            try:
                nparr = np.frombuffer(img_info["image"], np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img, caption=f'ID: {img_info["id"]}', use_container_width=True)
            except Exception as e:
                st.warning(f"Ошибка отображения изображения: {e}")

    def _count_total_images(self, files):
        count = 0
        for file in files:
            if file.type in ["application/zip", "application/x-zip-compressed"]:
                with zipfile.ZipFile(file) as z:
                    count += len(z.namelist())
            else:
                count += 1
        return count

    def _process_zip(self, file):
        with zipfile.ZipFile(file) as z:
            return [
                z.read(name)
                for name in z.namelist()
                if name.lower().endswith((".jpg", ".png"))
            ]

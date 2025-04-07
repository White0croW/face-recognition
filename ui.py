from deepface import DeepFace
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
            "Меню",
            ["Добавить в БД", "Распознать", "Фотографии из БД"],
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
            progress_bar = st.progress(0)
            success = 0
            errors = []

            for file in files:
                try:
                    if file.type in ["application/zip", "application/x-zip-compressed"]:
                        images = self._process_zip(file)
                        for img_bytes in images:
                            self.service.add_image_to_db(img_bytes)
                            success += 1
                            progress_bar.progress(success / total_images)
                    else:
                        self.service.add_image_to_db(file.getvalue())
                        success += 1
                        progress_bar.progress(success / total_images)
                except Exception as e:
                    errors.append(str(e))

            progress_bar.empty()
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
        total_images = self.service.db.get_image_count()
        if total_images == 0:
            st.error("База данных пуста")
            return

        # Отображаем загруженное изображение
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Загруженное изображение:")
            st.image(image_bytes, caption="Загруженное фото", use_container_width=True)

        progress_bar = st.progress(0)
        matches = []
        query_embedding = self.service._extract_embedding(image_bytes)

        if query_embedding is None:
            st.error("Лицо не обнаружено")
            return

        for i, db_img in enumerate(self.service.db.get_all_images()):
            try:
                db_embedding = self.service._extract_embedding(db_img)
                if db_embedding is None:
                    continue
                similarity = self.service._calculate_similarity(
                    query_embedding, db_embedding
                )
                if similarity >= 0.6:
                    matches.append({"image": db_img, "similarity": similarity})
            except Exception as e:
                print(f"Ошибка обработки изображения из БД: {e}")
            finally:
                progress_bar.progress((i + 1) / total_images)

        progress_bar.empty()
        self._display_matches(matches)

    def _display_matches(self, matches):
        if matches:
            col1, col2 = st.columns(2)
            with col2:
                st.subheader("Найденные совпадения:")
                for match in matches:
                    img_with_faces = self._detect_faces(match["image"])
                    st.image(
                        img_with_faces, caption=f"Сходство: {match['similarity']:.2f}"
                    )
        else:
            st.error("Совпадений не найдено")

    def _render_photos(self):
        st.header("Фотографии из БД")
        images = self.service.db.get_all_images()

        if not images:
            st.info("База данных пуста")
            return

        # Настройки пагинации
        page_size = 10  # Количество изображений на странице
        total_pages = (len(images) // page_size) + (
            1 if len(images) % page_size != 0 else 0
        )
        page_number = st.number_input(
            "Страница", min_value=1, max_value=total_pages, value=1, step=1
        )

        # Вычисление диапазона изображений для текущей страницы
        start_idx = (page_number - 1) * page_size
        end_idx = start_idx + page_size

        for i, img_bytes in enumerate(images[start_idx:end_idx], start=start_idx):
            try:
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img, caption=f"Изображение {i+1}", use_container_width=True)
            except Exception as e:
                st.warning(f"Ошибка отображения изображения {i}: {e}")

    def _detect_faces(self, image_bytes):
        """Обнаружение лиц на изображении"""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                return None

            # Преобразование из BGR в RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Обнаружение лиц
            face_locations = DeepFace.extract_faces(
                rgb_img, detector_backend="opencv", enforce_detection=False
            )

            # Рисование рамок вокруг лиц
            for face in face_locations:
                x, y, w, h = (
                    face["facial_area"]["x"],
                    face["facial_area"]["y"],
                    face["facial_area"]["w"],
                    face["facial_area"]["h"],
                )
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            return img
        except Exception as e:
            print(f"Ошибка при обработке изображения: {e}")
            return None

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
        """Извлечение изображений из ZIP-архива"""
        images = []
        with zipfile.ZipFile(file) as z:
            for name in z.namelist():
                with z.open(name) as f:
                    images.append(f.read())
        return images

from deepface import DeepFace


def recognize_face(image):
    """Извлекает эмбеддинг лица из изображения."""
    try:
        embedding = DeepFace.represent(
            image, model_name="ArcFace", enforce_detection=False
        )[0]["embedding"]
        return np.array(embedding)
    except Exception as e:
        print(f"Ошибка распознавания: {e}")
        return None

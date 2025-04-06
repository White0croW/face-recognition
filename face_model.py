# face_model.py (с интеграцией Hugging Face)
from transformers import AutoImageProcessor, AutoModel
import torch
import numpy as np
from PIL import Image


class FaceRecognizer:
    def __init__(self):
        # Модель для детекции лиц [[6]]
        self.detector = AutoModel.from_pretrained("huggingface/blazeface")
        # Модель для эмбеддингов [[7]]
        self.embedder = AutoModel.from_pretrained("facebook/face-embeddings-256")
        self.processor = AutoImageProcessor.from_pretrained(
            "facebook/face-embeddings-256"
        )

    def detect_faces(self, image: Image.Image):
        inputs = self.processor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.detector(**inputs)
        return outputs["boxes"].cpu().numpy().tolist()

    def recognize(self, face_image: Image.Image):
        inputs = self.processor(face_image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.embedder(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

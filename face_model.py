from transformers import AutoModel, AutoImageProcessor
import torch
import numpy as np
from PIL import Image


class FaceRecognizer:
    def __init__(self):
        # Загрузка модели для генерации embedding [[3]]
        self.model = AutoModel.from_pretrained("facebook/face-embeddings-256")
        self.processor = AutoImageProcessor.from_pretrained(
            "facebook/face-embeddings-256"
        )

    def get_embedding(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

"""
Inference - Inferência e Demonstração

Responsável por:
- Predição em imagens individuais
- Visualização de landmarks
- Demo/Inferência em tempo real
"""

import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path


class LandmarkPredictor:
    """
    Classe para inferência de landmarks em imagens
    """
    def __init__(self, model, device, image_size=640):
        self.model = model
        self.device = device
        self.image_size = image_size
        self.model.eval()
        
        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    def preprocess_image(self, image_path):
        """Carrega e pré-processa imagem"""
        # Carregar imagem
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Redimensionar
        image_resized = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_array = np.array(image_resized, dtype=np.float32) / 255.0
        
        # Normalizar
        image_array = (image_array - self.mean) / self.std
        
        # Para tensor
        image_tensor = torch.from_numpy(np.transpose(image_array, (2, 0, 1))).float()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor, original_size, np.array(image)

    @torch.no_grad()
    def predict(self, image_path, confidence_threshold=0.5):
        """Prediz landmarks em uma imagem"""
        # Pré-processar
        image_tensor, original_size, original_image = self.preprocess_image(image_path)
        
        # Predição
        landmarks, confidence, mask = self.model.predict_landmarks(
            image_tensor,
            confidence_threshold=confidence_threshold
        )
        
        # Converter para coordenadas da imagem original
        landmarks = landmarks[0].cpu().numpy()  # (68, 2)
        confidence = confidence[0].cpu().numpy()  # (68,)
        mask = mask[0].cpu().numpy()  # (68,)
        
        landmarks[:, 0] *= original_size[0]  # Escalar X
        landmarks[:, 1] *= original_size[1]  # Escalar Y
        
        return landmarks, confidence, mask, original_image

    def visualize_landmarks(self, image, landmarks, confidence, confidence_threshold=0.5):
        """Desenha landmarks na imagem"""
        image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        for i, (x, y) in enumerate(landmarks):
            conf = confidence[i]
            # Verde se confiança alta, vermelho se baixa
            color = (0, 255, 0) if conf > confidence_threshold else (0, 0, 255)
            cv2.circle(image_cv, (int(x), int(y)), 2, color, -1)
        
        return image_cv

    def predict_and_save(self, image_path, output_path=None, confidence_threshold=0.5):
        """Prediz landmarks e salva resultado"""
        print(f"\n🖼️  Processando: {image_path}")
        
        # Predição
        landmarks, confidence, mask, original_image = self.predict(
            image_path, 
            confidence_threshold
        )
        
        # Visualizar
        result_image = self.visualize_landmarks(
            original_image, 
            landmarks, 
            confidence, 
            confidence_threshold
        )
        
        # Salvar
        if output_path is None:
            output_path = Path(image_path).with_stem(Path(image_path).stem + '_landmarks')
        
        cv2.imwrite(str(output_path), result_image)
        
        # Estatísticas
        num_detected = mask.sum()
        avg_confidence = confidence[mask].mean() if num_detected > 0 else 0
        
        print(f"✅ Resultado salvo em: {output_path}")
        print(f"   Landmarks detectados: {num_detected}/68")
        print(f"   Confidence média: {avg_confidence:.4f}")
        
        return landmarks, confidence, mask

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import pandas as pd

from config.settings import (
    IBUG_LANDMARKS,
    DEFAULT_IMAGE_CONFIG,
    ImageConfig,
    get_dataset_path
)


class LandmarkDataset(Dataset):
    """
    Dataset para carregar imagens com landmarks faciais anotados.

    Formatos suportados:
    1. JSON format: {"image_path": "path/to/image.jpg", "landmarks": [[x1, y1], [x2, y2], ...]}
    2. PTS format: Arquivo de texto .pts com coordenadas dos landmarks.
    3. CSV format: "image_path,x1,y1,x2,y2,...,x68,y68,v1,v2,...,v68".
    """
    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        format: str = 'json',
        image_config: Optional[ImageConfig] = None,
        transform=None,
        augment=False
    ):
        """
        Args:
            root_dir (str): Diretório raiz onde as imagens estão armazenadas.
            annotation_file (str): Caminho para o arquivo de anotações.
            format (str): Formato do arquivo de anotações ('json', 'pts', 'csv').
            image_config (ImageConfig, opcional): Configurações para pré-processamento de imagens.
            transform (callable, opcional): Transformações a serem aplicadas nas imagens.
            augment (bool): Se True, aplica aumentos de dados nas imagens.
        """
        self.root_dir = root_dir
        self.annotation_file = annotation_file
        self.format = format
        self.image_config = image_config or DEFAULT_IMAGE_CONFIG
        self.transform = transform
        self.augment = augment

        self.samples = self._load_annotations()

    def _load_annotations(self) -> List[Dict]:
        match self.format:
            case 'json':
                return self._load_json_annotations()
            case 'pts':
                return self._load_pts_annotations()
            case 'csv':
                return self._load_csv_annotations()
            case _:
                raise ValueError(f"Formato de anotação não suportado: {self.format}")
            
    def _load_json_annotations(self) -> List[Dict]:
        with open(self.annotation_file, 'r') as f:
            data = json.load(f)
        samples = []
        for item in data:
            image_path = os.path.join(self.root_dir, item['image_path'])
            landmarks = np.array(item['landmarks'], dtype=np.float32)
            visibility = np.array(item.get('visibility', [1]*68), dtype=np.float32)
            bbox = item.get('bbox', None)
            samples.append({
                'image_path': str(image_path),
                'landmarks': landmarks,
                'visibility': visibility,
                'bbox': bbox
            })
        return samples
    
    def _load_pts_annotations(self) -> List[Dict]:
        samples = []
        
        # Suportar múltiplos subdiretórios (01_Indoor, 02_Outdoor, etc)
        root_path = Path(self.root_dir)
        
        # Se root_dir contém subdiretórios, processar recursivamente
        if any(root_path.iterdir()):
            image_extensions = ['.png', '.jpg', '.jpeg']
            
            # Buscar recursivamente por imagens
            for ext in image_extensions:
                for img_file in root_path.rglob(f'*{ext}'):
                    pts_file = img_file.with_suffix('.pts')
                    
                    if not pts_file.exists():
                        continue
                    
                    # Ler arquivo .pts
                    with open(pts_file, 'r') as f:
                        lines = f.readlines()
                    
                    # Extrair landmarks (pular header e footer)
                    landmarks = []
                    reading = False
                    for line in lines:
                        line = line.strip()
                        if line == '{':
                            reading = True
                            continue
                        if line == '}':
                            break
                        if reading:
                            try:
                                x, y = map(float, line.split())
                                landmarks.append([x, y])
                            except ValueError:
                                continue
                    
                    if len(landmarks) == 68:  # Validar 68 landmarks
                        landmarks = np.array(landmarks, dtype=np.float32)
                        visibility = np.ones(len(landmarks), dtype=np.float32)
                        
                        samples.append({
                            'image_path': str(img_file),
                            'landmarks': landmarks,
                            'visibility': visibility,
                            'bbox': None
                        })
        
        return samples
    
    def _load_csv_annotations(self) -> List[Dict]:
        df = pd.read_csv(self.annotation_file)
        samples = []
        
        for _, row in df.iterrows():
            image_path = os.path.join(self.root_dir, row['image_path'])
            landmarks = []
            visibility = []
            
            for i in range(68):
                x = row[f'x{i+1}']
                y = row[f'y{i+1}']
                v = row.get(f'v{i+1}', 1)  # Visibilidade padrão como 1
                landmarks.append([x, y])
                visibility.append(v)
            
            landmarks = np.array(landmarks, dtype=np.float32)
            visibility = np.array(visibility, dtype=np.float32)
            
            samples.append({
                'image_path': str(image_path),
                'landmarks': landmarks,
                'visibility': visibility,
                'bbox': None
            })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Carregar imagem
        image = Image.open(sample['image_path']).convert('RGB')
        original_size = image.size  # (W, H)
        
        # Landmarks originais
        landmarks = sample['landmarks'].copy()
        visibility = sample['visibility'].copy()
        
        # Redimensionar imagem
        image = image.resize(
            (self.image_config.width, self.image_config.height),
            Image.BILINEAR
        )
        
        # Normalizar coordenadas dos landmarks
        landmarks[:, 0] = landmarks[:, 0] / original_size[0]  # Normalizar X
        landmarks[:, 1] = landmarks[:, 1] / original_size[1]  # Normalizar Y
        
        # Converter para tensor
        image = np.array(image, dtype=np.float32) / 255.0
        
        # Aplicar normalização ImageNet se configurado
        if self.image_config.normalize:
            mean = np.array(self.image_config.mean).reshape(1, 1, 3)
            std = np.array(self.image_config.std).reshape(1, 1, 3)
            image = (image - mean) / std
        
        # Transpor para (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        
        # Data augmentation (se habilitado)
        if self.augment:
            image, landmarks, visibility = self._apply_augmentation(
                image, landmarks, visibility
            )
        
        # Aplicar transformações customizadas
        if self.transform is not None:
            image = self.transform(image)
        
        return {
            'image': torch.from_numpy(image).float(),
            'landmarks': torch.from_numpy(landmarks).float(),
            'visibility': torch.from_numpy(visibility).float(),
            'original_size': original_size
        }

    def _apply_augmentation(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        visibility: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aplica data augmentation
        
        Augmentations aplicados:
        - Random horizontal flip
        - Random brightness/contrast
        - Random rotation (pequeno ângulo)
        - Random noise
        """
        # Horizontal flip (50% chance)
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=2).copy()  # Flip horizontal (W axis)
            landmarks[:, 0] = 1.0 - landmarks[:, 0]  # Inverter coordenadas X
            
            # Trocar landmarks espelhados (ex: olho esquerdo ↔ olho direito)
            landmarks = self._swap_mirrored_landmarks(landmarks)
        
        # Random brightness/contrast
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)  # Contraste
            beta = np.random.uniform(-0.1, 0.1)  # Brilho
            image = np.clip(image * alpha + beta, 0, 1)
        
        # Random noise
        if np.random.rand() > 0.7:
            noise = np.random.normal(0, 0.02, image.shape)
            image = np.clip(image + noise, 0, 1)
        
        return image, landmarks, visibility

    def _swap_mirrored_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Troca landmarks espelhados após flip horizontal
        
        Baseado no mapeamento iBUG:
        - Sobrancelhas: esquerda (18-22) ↔ direita (23-27)
        - Olhos: esquerdo (37-42) ↔ direito (43-48)
        - Contorno facial é simétrico
        """
        swapped = landmarks.copy()
        
        # Sobrancelhas
        swapped[17:22], swapped[22:27] = landmarks[22:27].copy(), landmarks[17:22].copy()
        
        # Olhos
        swapped[36:42], swapped[42:48] = landmarks[42:48].copy(), landmarks[36:42].copy()
        
        # Nariz (31-36 são simétricos)
        swapped[31:36] = landmarks[[35, 34, 33, 32, 31]].copy()
        
        # Boca externa (49-60 são simétricos)
        mouth_outer = [54, 53, 52, 51, 50, 49, 60, 59, 58, 57, 56, 55]
        swapped[48:60] = landmarks[[48 + i for i in mouth_outer]].copy()
        
        # Boca interna (61-68 são simétricos)
        mouth_inner = [64, 63, 62, 61, 68, 67, 66, 65]
        swapped[60:68] = landmarks[[60 + i for i in mouth_inner]].copy()
        
        return swapped


def create_dataloader(
    root_dir: str,
    annotation_file: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    format: str = 'json',
    image_config: Optional[ImageConfig] = None,
    augment: bool = False
) -> DataLoader:
    """
    Cria um DataLoader para o dataset de landmarks
    
    Args:
        root_dir: Diretório raiz do dataset
        annotation_file: Arquivo de anotações
        batch_size: Tamanho do batch
        shuffle: Embaralhar dados
        num_workers: Número de workers para carregamento paralelo
        format: Formato das anotações
        image_config: Configurações de imagem
        augment: Habilitar data augmentation
    
    Returns:
        DataLoader configurado
    """
    dataset = LandmarkDataset(
        root_dir=root_dir,
        annotation_file=annotation_file,
        format=format,
        image_config=image_config,
        augment=augment
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader

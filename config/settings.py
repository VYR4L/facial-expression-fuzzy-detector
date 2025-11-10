from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_SETS_DIR = ROOT_DIR / "datasets"

@dataclass
class LandmarkPoint:
    """
    Classe para representar um ponto de referência 2D em uma imagem.
    """
    name: str
    coordinates: Tuple[float, float]


@dataclass
class ImageConfig:
    """
    Classe para armazenar configurações relacionadas a imagens.
    """
    width: int
    height: int
    channels: int = 3
    color_mode: str = 'RGB'  # e.g., 'RGB', 'Grayscale'
    normalize: bool = True
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet mean
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)   # ImageNet std


@dataclass
class ModelConfig:
    """
    Classe para armazenar configurações relacionadas ao modelo.
    """
    model_name: str
    input_shape: Tuple[int, int, int] # e.g., (height, width, channels)
    num_classes: int
    learning_rate: float
    epochs: int
    batch_size: int


@dataclass
class TrainingConfig:
    """
    Classe para armazenar configurações relacionadas ao treinamento.
    """
    train_split: float
    validation_split: float
    shuffle: bool
    augmentations: Dict[str, bool]  # e.g., {'flip': True, 'rotate': False}


# Configurações padrão para imagens
# IBUG Dataset Landmarks
IBUG_LANDMARKS: List[LandmarkPoint] = [
    LandmarkPoint("Contorno facial", (1, 17)),
    LandmarkPoint("Sobrancelha esquerda", (18, 22)),
    LandmarkPoint("Sobrancelha direita", (23, 27)),
    LandmarkPoint("Nariz", (28, 36)),
    LandmarkPoint("Olho esquerdo", (37, 42)),
    LandmarkPoint("Olho direito", (43, 48)),
    LandmarkPoint("Boca", (49, 68)),
]

DEFAULT_IMAGE_CONFIG = ImageConfig(
    width=640,
    height=640,
    channels=3,
    color_mode='RGB',
    normalize=True,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

DEFAULT_MODEL_CONFIG = ModelConfig(
    model_name='Yolov11',
    input_shape=(640, 640, 3),
    num_classes=10,
    learning_rate=0.001,
    epochs=50,
    batch_size=32
)

DEFAULT_TRAINING_CONFIG = TrainingConfig(
    train_split=0.8,
    validation_split=0.1,
    shuffle=True,
    augmentations={'flip': True, 'rotate': True, 'zoom': False}
)


def get_dataset_path(dataset_name: str) -> Path:
    """
    Retorna o caminho completo para um conjunto de dados específico.
    """
    return DATA_SETS_DIR / dataset_name


def list_available_datasets() -> List[str]:
    """
    Lista todos os conjuntos de dados disponíveis no diretório de datasets.
    """
    return [d.name for d in DATA_SETS_DIR.iterdir() if d.is_dir()]

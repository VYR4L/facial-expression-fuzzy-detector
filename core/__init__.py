"""
Core components do YOLOv11 para detecção de landmarks faciais
"""
from .backbone import YOLOv11Backbone
from .neck import YOLOv11Neck
from .head import LandmarkDetectionHead, YOLOv11LandmarkDetector

__all__ = [
    'YOLOv11Backbone',
    'YOLOv11Neck',
    'LandmarkDetectionHead',
    'YOLOv11LandmarkDetector'
]

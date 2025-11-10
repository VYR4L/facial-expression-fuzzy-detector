import sys
from pathlib import Path

# Adicionar o diretório raiz ao PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import torch
from core.head import YOLOv11LandmarkDetector


def test_model():
    """Testa a arquitetura do modelo"""
    print("=" * 60)
    print("Testando YOLOv11 para detecção de landmarks faciais")
    print("=" * 60)
    
    # Criar modelo
    model = YOLOv11LandmarkDetector(
        in_channels=3,
        base_channels=64,
        num_landmarks=68
    )
    
    # Colocar em modo de avaliação
    model.eval()
    
    # Criar imagem de exemplo (batch_size=2, 3 canais, 640x640)
    dummy_input = torch.randn(2, 3, 640, 640)
    
    print(f"\n📥 Input shape: {dummy_input.shape}")
    print(f"   (batch_size, channels, height, width)")
    
    # Forward pass
    with torch.no_grad():
        landmarks = model(dummy_input)
    
    print(f"\n📤 Output shape: {landmarks.shape}")
    print(f"   (batch_size, num_landmarks, 3)")
    print(f"   3 = [x, y, confidence]")
    
    # Testar predição com threshold
    landmarks_xy, confidence, mask = model.predict_landmarks(
        dummy_input,
        confidence_threshold=0.5
    )
    
    print(f"\n🎯 Landmarks (x, y): {landmarks_xy.shape}")
    print(f"   Confidence scores: {confidence.shape}")
    print(f"   Valid mask: {mask.shape}")
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Estatísticas do modelo:")
    print(f"   Total de parâmetros: {total_params:,}")
    print(f"   Parâmetros treináveis: {trainable_params:,}")
    print(f"   Tamanho aproximado: {total_params * 4 / (1024**2):.2f} MB (float32)")
    
    # Exemplo de landmarks detectados
    print(f"\n💡 Exemplo de landmarks da primeira imagem:")
    print(f"   Landmark 1 (contorno facial):")
    print(f"      x={landmarks[0, 0, 0]:.4f}, y={landmarks[0, 0, 1]:.4f}, conf={landmarks[0, 0, 2]:.4f}")
    print(f"   Landmark 37 (olho esquerdo):")
    print(f"      x={landmarks[0, 36, 0]:.4f}, y={landmarks[0, 36, 1]:.4f}, conf={landmarks[0, 36, 2]:.4f}")
    print(f"   Landmark 49 (boca):")
    print(f"      x={landmarks[0, 48, 0]:.4f}, y={landmarks[0, 48, 1]:.4f}, conf={landmarks[0, 48, 2]:.4f}")
    
    print("\n✅ Teste concluído com sucesso!")
    print("\n📝 Próximos passos:")
    print("   1. Preparar dataset com landmarks anotados")
    print("   2. Implementar função de loss (ex: MSE + confidence loss)")
    print("   3. Criar script de treinamento")
    print("   4. Integrar com o sistema fuzzy para classificação de emoções")
    print("=" * 60)


def show_architecture():
    """Mostra a arquitetura do modelo"""
    print("\n" + "=" * 60)
    print("ARQUITETURA DO MODELO")
    print("=" * 60)
    
    print("""
    YOLOv11 para Detecção de Landmarks Faciais
    
    📐 BACKBONE (Feature Extraction):
    ├── Stem: Conv 3→64, stride=2
    ├── Stage 1: Conv + C3K2 (64→128)
    ├── Stage 2: Conv + C3K2 (128→256)
    ├── Stage 3: Conv + C3K2 (256→512)  → P3 (1/16)
    ├── Stage 4: Conv + C3K2 (512→1024) → P4 (1/32)
    └── Stage 5: Conv + C3K2 + SPFF     → P5 (1/64)
    
    🔀 NECK (Feature Fusion - PANet):
    ├── Top-Down Pathway:
    │   ├── P5 → P4 (Upsample + Concat + C3K2)
    │   └── P4 → P3 (Upsample + Concat + C3K2)
    └── Bottom-Up Pathway:
        ├── P3 → P4 (Downsample + Concat + C3K2)
        ├── P4 → P5 (Downsample + Concat + C3K2)
        └── Refinar com C2PSA (Spatial Attention)
    
    🎯 HEAD (Landmark Detection):
    ├── Multi-scale detection (N3, N4, N5)
    ├── Fusão de predições
    └── Output: 68 landmarks × 3 (x, y, confidence)
    
    📊 SAÍDA FINAL:
    └── Tensor (B, 68, 3)
        ├── [:, :, 0] → Coordenada X normalizada [0, 1]
        ├── [:, :, 1] → Coordenada Y normalizada [0, 1]
        └── [:, :, 2] → Confidence score [0, 1]
    """)
    print("=" * 60)


if __name__ == "__main__":
    show_architecture()
    test_model()

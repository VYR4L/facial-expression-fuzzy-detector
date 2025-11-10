import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from utils.dataset_loader import LandmarkDataset

if __name__ == "__main__":
    print("=" * 60)
    print("Testando LandmarkDataset")
    print("=" * 60)
    
    # Exemplo com PTS format
    dataset = LandmarkDataset(
        root_dir="datasets/300W",
        annotation_file="",  # Não usado para PTS
        format='pts',
        augment=False
    )

    print(f"\n📊 Dataset carregado:")
    print(f"   Número de amostras: {len(dataset)}")
    print(f"   Formato: PTS (300W)")
    
    # Testar primeira amostra
    print(f"\n🔍 Testando primeira amostra...")
    sample = dataset[0]
    
    print(f"\n✅ Amostra carregada com sucesso!")
    print(f"   Image shape: {sample['image'].shape}")
    print(f"   Landmarks shape: {sample['landmarks'].shape}")
    print(f"   Visibility shape: {sample['visibility'].shape}")
    print(f"   Original size: {sample['original_size']}")
    
    # Mostrar alguns landmarks
    print(f"\n💡 Exemplos de landmarks (primeiros 5):")
    for i in range(5):
        x, y = sample['landmarks'][i]
        vis = sample['visibility'][i]
        print(f"   Landmark {i+1}: x={x:.4f}, y={y:.4f}, visible={vis:.0f}")
    
    # Testar múltiplas amostras
    print(f"\n🔄 Testando 10 amostras aleatórias...")
    import random
    indices = random.sample(range(len(dataset)), min(10, len(dataset)))
    
    for idx in indices:
        sample = dataset[idx]
        assert sample['image'].shape == (3, 640, 640), f"Shape incorreto: {sample['image'].shape}"
        assert sample['landmarks'].shape == (68, 2), f"Landmarks incorretos: {sample['landmarks'].shape}"
        assert sample['visibility'].shape == (68,), f"Visibility incorreto: {sample['visibility'].shape}"
    
    print(f"   ✅ Todas as 10 amostras passaram nos testes!")
    
    print("\n" + "=" * 60)
    print("✅ Teste concluído com sucesso!")
    print("=" * 60)
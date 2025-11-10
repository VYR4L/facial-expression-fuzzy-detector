import sys
from pathlib import Path

# Adicionar o diretório raiz ao PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from utils.wing_loss import WingLoss, LandmarkLoss, AdaptativeWingLoss
import torch


if __name__ == "__main__":
    print("Testando Loss Functions para Landmark Detection\n")
    
    # Criar loss
    criterion = LandmarkLoss(
        wing_omega=10,
        wing_epsilon=2,
        lambda_coord=5.0,
        lambda_conf=1.0,
        use_weights=True
    )
    
    # Dados de exemplo
    batch_size = 4
    num_landmarks = 68
    
    # Predições do modelo (após sigmoid na confidence)
    predictions = torch.randn(batch_size, num_landmarks, 3)
    predictions[:, :, :2] = torch.sigmoid(predictions[:, :, :2])  # Normalizar coords
    
    # Ground truth
    targets = {
        'landmarks': torch.rand(batch_size, num_landmarks, 2),  # Coords [0, 1]
        'visibility': torch.randint(0, 2, (batch_size, num_landmarks)).float()  # 0 ou 1
    }
    
    # Calcular loss
    losses = criterion(predictions, targets)
    
    print(f"📊 Losses calculadas:")
    print(f"   Total Loss: {losses['loss']:.4f}")
    print(f"   Localization Loss: {losses['loc_loss']:.4f}")
    print(f"   Confidence Loss: {losses['conf_loss']:.4f}")
    
    print(f"\n✅ Loss function implementada com sucesso!")
    print(f"\n💡 Pesos por região facial:")
    print(f"   Contorno: 1.0x")
    print(f"   Sobrancelhas: 1.5x")
    print(f"   Olhos: 2.0x ⭐")
    print(f"   Nariz: 1.0x")
    print(f"   Boca: 2.0x ⭐")
"""
Main - Sistema de Detecção de Landmarks Faciais com YOLOv11

Uso:
    python main.py --mode train --epochs 100 --batch-size 32
    python main.py --mode test --weights best.pth
    python main.py --mode demo --image path/to/face.jpg
"""

import argparse
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path

# Importar componentes do projeto
from core import YOLOv11LandmarkDetector
from utils import LandmarkDataset, LandmarkLoss, Trainer, Evaluator, LandmarkPredictor
from config import DEFAULT_IMAGE_CONFIG


def setup_model_and_criterion(args, device):
    """Configura modelo e função de loss"""
    model = YOLOv11LandmarkDetector(
        in_channels=3,
        base_channels=args.base_channels,
        num_landmarks=68
    ).to(device)
    
    criterion = LandmarkLoss(
        wing_omega=10.0,
        wing_epsilon=2.0,
        lambda_coord=5.0,
        lambda_conf=1.0,
        use_weights=True
    )
    
    return model, criterion


def setup_dataloaders(args):
    """Configura dataloaders para treino e validação"""
    # Carregar dataset
    print("\n📂 Carregando dataset...")
    full_dataset = LandmarkDataset(
        root_dir=args.data_dir,
        annotation_file="",
        format='pts',
        image_config=DEFAULT_IMAGE_CONFIG,
        augment=True
    )
    
    # Split treino/validação
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Desabilitar augmentation no dataset de validação
    val_dataset.dataset.augment = False
    
    # Criar dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def train_model(args):
    """Função principal de treinamento"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Configurar componentes
    train_loader, val_loader = setup_dataloaders(args)
    model, criterion = setup_model_and_criterion(args, device)
    
    # Carregar checkpoint se especificado
    if args.resume:
        print(f"📥 Carregando checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Criar optimizer e scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10
    )
    
    # Criar e executar trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        save_dir=args.save_dir
    )
    
    trainer.train()


def test_model(args):
    """Função para testar o modelo"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("Testando Modelo YOLOv11 para Landmarks Faciais")
    print("=" * 70)
    
    # Carregar modelo e critério
    model, criterion = setup_model_and_criterion(args, device)
    
    print(f"\n📥 Carregando pesos: {args.weights}")
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Carregar dataset de teste
    print("\n📂 Carregando dataset de teste...")
    test_dataset = LandmarkDataset(
        root_dir=args.data_dir,
        annotation_file="",
        format='pts',
        image_config=DEFAULT_IMAGE_CONFIG,
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Criar evaluator e avaliar
    evaluator = Evaluator(model, criterion, device)
    results = evaluator.evaluate(test_loader)
    evaluator.print_results(results)


def demo(args):
    """Demonstração com uma imagem"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("Demo - Detecção de Landmarks Faciais")
    print("=" * 70)
    
    # Carregar modelo
    model, _ = setup_model_and_criterion(args, device)
    
    print(f"\n📥 Carregando pesos: {args.weights}")
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Criar predictor
    predictor = LandmarkPredictor(model, device, image_size=640)
    
    # Predizer e salvar
    predictor.predict_and_save(
        args.image,
        args.output,
        args.conf_threshold
    )
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='YOLOv11 Facial Landmarks Detection')
    
    # Modo de operação
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'demo'],
                        help='Modo de operação')
    
    # Dataset
    parser.add_argument('--data-dir', type=str, default='datasets/300W',
                        help='Diretório do dataset')
    
    # Modelo
    parser.add_argument('--base-channels', type=int, default=64,
                        help='Canais base da backbone')
    parser.add_argument('--weights', type=str, default='checkpoints/best_model.pth',
                        help='Caminho para os pesos do modelo')
    parser.add_argument('--resume', type=str, default='',
                        help='Retomar treinamento de checkpoint')
    
    # Treinamento
    parser.add_argument('--epochs', type=int, default=100,
                        help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Tamanho do batch')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate inicial')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Número de workers para DataLoader')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Diretório para salvar checkpoints')
    
    # Demo
    parser.add_argument('--image', type=str, default='',
                        help='Caminho da imagem para demo')
    parser.add_argument('--output', type=str, default='',
                        help='Caminho para salvar resultado')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='Threshold de confiança')
    
    args = parser.parse_args()
    
    # Executar modo apropriado
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'test':
        test_model(args)
    elif args.mode == 'demo':
        if not args.image:
            print("❌ Erro: --image é obrigatório no modo demo")
            return
        demo(args)


if __name__ == "__main__":
    main()

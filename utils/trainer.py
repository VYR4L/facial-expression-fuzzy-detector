"""
Trainer - Gerenciador de Treinamento do Modelo

Responsável por:
- Training loop
- Validação
- Checkpoints
- Logging
"""

import torch
import torch.nn as nn
from pathlib import Path
import time
from tqdm import tqdm
from torch.amp import autocast, GradScaler


class Trainer:
    """
    Classe para gerenciar o treinamento do modelo
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs,
        save_dir='checkpoints',
        accumulation_steps=4,
        use_amp=True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp and torch.cuda.is_available()
        # ✅ CORRIGIDO: GradScaler atualizado para PyTorch 2.9+
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_loc_loss': [],
            'train_conf_loss': []
        }

    def train_epoch(self, epoch):
        """Treina uma época"""
        self.model.train()
        total_loss = 0
        total_loc = 0
        total_conf = 0
        
        train_pbar = tqdm(self.train_loader, desc=f"Época {epoch+1}/{self.num_epochs} [Train]", leave=True)
        
        for batch_idx, batch in enumerate(train_pbar):
            images = batch['image'].to(self.device)
            landmarks = batch['landmarks'].to(self.device)
            visibility = batch['visibility'].to(self.device)
            
            # ✅ CORRIGIDO: autocast atualizado para PyTorch 2.9+
            with autocast('cuda', enabled=self.use_amp):
                predictions = self.model(images)
                
                # Calcular loss
                targets = {
                    'landmarks': landmarks,
                    'visibility': visibility
                }
                losses = self.criterion(predictions, targets)
                loss = losses['loss'] / self.accumulation_steps
            
            # ✅ MIXED PRECISION: Backward com gradient scaling
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Atualizar pesos a cada N steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # ✅ LIMPAR CACHE A CADA 10 BATCHES
            if (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
            
            # Métricas
            total_loss += losses['loss'].item()
            total_loc += losses['loc_loss']
            total_conf += losses['conf_loss']
            
            # Atualizar progress bar
            train_pbar.set_postfix({
                'loss': f'{losses["loss"].item():.2f}',
                'loc': f'{losses["loc_loss"]:.3f}',
                'conf': f'{losses["conf_loss"]:.3f}'
            })
        
        # Limpar gradientes restantes
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()
        
        num_batches = len(self.train_loader)
        return total_loss / num_batches, total_loc / num_batches, total_conf / num_batches

    @torch.no_grad()
    def validate(self, epoch):
        """Valida o modelo"""
        self.model.eval()
        total_loss = 0
        total_loc_loss = 0
        total_conf_loss = 0
        
        pbar = tqdm(self.val_loader, desc=f'Época {epoch+1}/{self.num_epochs} [Val]')
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            landmarks = batch['landmarks'].to(self.device)
            visibility = batch['visibility'].to(self.device)
            
            # Forward pass
            predictions = self.model(images)
            
            # Calcular loss
            targets = {
                'landmarks': landmarks,
                'visibility': visibility
            }
            losses = self.criterion(predictions, targets)
            
            # Acumular losses
            total_loss += losses['loss'].item()
            total_loc_loss += losses['loc_loss'].item()
            total_conf_loss += losses['conf_loss'].item()
            
            pbar.set_postfix({
                'loss': losses['loss'].item(),
                'loc': losses['loc_loss'].item(),
                'conf': losses['conf_loss'].item()
            })
        
        avg_loss = total_loss / len(self.val_loader)
        avg_loc_loss = total_loc_loss / len(self.val_loader)
        avg_conf_loss = total_conf_loss / len(self.val_loader)
        
        return avg_loss, avg_loc_loss, avg_conf_loss

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Salva checkpoint do modelo"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }
        
        # Salvar checkpoint regular
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Salvar melhor modelo
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f'✅ Melhor modelo salvo com val_loss={val_loss:.4f}')

    def train(self):
        """Loop de treinamento completo"""
        print("=" * 70)
        print("Iniciando Treinamento do YOLOv11 para Landmarks Faciais")
        print("=" * 70)
        print(f"📊 Dataset:")
        print(f"   Treino: {len(self.train_loader.dataset)} amostras")
        print(f"   Validação: {len(self.val_loader.dataset)} amostras")
        print(f"\n⚙️  Configurações:")
        print(f"   Épocas: {self.num_epochs}")
        print(f"   Batch size: {self.train_loader.batch_size}")
        print(f"   Accumulation steps: {self.accumulation_steps}")
        print(f"   Effective batch size: {self.train_loader.batch_size * self.accumulation_steps}")
        print(f"   Mixed Precision (AMP): {'Ativado' if self.use_amp else 'Desativado'}")
        print(f"   Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"   Device: {self.device}")
        print("=" * 70 + "\n")
        
        for epoch in range(self.num_epochs):
            start_time = time.time()
            
            # Treinar
            train_loss, train_loc, train_conf = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_loc_loss'].append(train_loc)
            self.history['train_conf_loss'].append(train_conf)
            
            # Validar
            val_loss, val_loc, val_conf = self.validate(epoch)
            self.history['val_loss'].append(val_loss)
            
            # Atualizar learning rate
            self.scheduler.step(val_loss)
            
            epoch_time = time.time() - start_time
            
            # Log
            print(f"\n📈 Época {epoch+1}/{self.num_epochs} - {epoch_time:.2f}s")
            print(f"   Train Loss: {train_loss:.4f} (loc: {train_loc:.4f}, conf: {train_conf:.4f})")
            print(f"   Val Loss:   {val_loss:.4f} (loc: {val_loc:.4f}, conf: {val_conf:.4f})")
            print(f"   LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Salvar checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if (epoch + 1) % 10 == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
            
            print()
        
        print("=" * 70)
        print(f"✅ Treinamento concluído!")
        print(f"   Melhor Val Loss: {self.best_val_loss:.4f}")
        print(f"   Checkpoints salvos em: {self.save_dir}")
        print("=" * 70)

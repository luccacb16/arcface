import argparse
from PIL import Image
import wandb
import os
import random
import numpy as np
import math
import warnings

from sklearn.metrics import precision_recall_fscore_support

import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomCrop, RandomHorizontalFlip, RandomRotation, ColorJitter
from torch.utils.data import Dataset
from torch.amp import autocast

transform = Compose([
    Resize([112, 112]),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

aug_transform = Compose([
    RandomCrop([112, 112]),
    RandomHorizontalFlip(),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --------------------------------------------------------------------------------------------------------

class CustomDataset(Dataset):
    def __init__(self, images_df, transform=None, dtype=torch.bfloat16):
        self.labels = images_df['id'].values
        self.image_paths = images_df['path'].values
        self.transform = transform
        self.dtype = dtype

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        image = image.to(self.dtype)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, label
    
# --------------------------------------------------------------------------------------------------------

# EVAL

def test(model, val_dataloader, criterion, dtype=torch.bfloat16, device='cuda'):
    correct = 0
    total = 0
    total_loss = 0.0
    
    all_labels = []
    all_preds = []
    
    model.eval()

    with autocast(dtype=dtype, device_type=device):
        with torch.no_grad():
            for data in val_dataloader:
                inputs, labels = data
                inputs, labels = inputs.to(device=device, dtype=dtype), labels.to(device=device)

                outputs = model(inputs, labels)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

    model.train()
    
    accuracy = correct / total
    loss = total_loss / len(val_dataloader)
    
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)

    return accuracy, precision, recall, f1, loss

class ArcFaceLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, reduction_epochs=None, reduction_factor=0.1, warmup_epochs=5, last_epoch=-1, warmup_lr=2.5e-2):
        self.reduction_epochs = reduction_epochs if reduction_epochs is not None else [20, 28]
        self.reduction_factor = reduction_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        super(ArcFaceLRScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = self.last_epoch / max(1, self.warmup_epochs - 1)
            return [self.warmup_lr + (base_lr - self.warmup_lr) * warmup_factor for base_lr in self.initial_lrs]
        else:
            num_reductions = sum(epoch <= self.last_epoch for epoch in self.reduction_epochs)
            factor = self.reduction_factor ** num_reductions
            return [base_lr * factor for base_lr in self.initial_lrs]

class WarmUpCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, epochs, warmup_epochs, min_lr, max_lr, last_epoch=-1):
        self.epochs = epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_epochs = warmup_epochs
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [self.min_lr + (self.max_lr - self.min_lr) * alpha for _ in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (self.max_lr - self.min_lr) * cosine_decay for _ in self.base_lrs]

def save_model_artifact(checkpoint_path, epoch):
    artifact = wandb.Artifact(f'epoch_{epoch}', type='model')
    artifact.add_file(os.path.join(checkpoint_path, f'epoch_{epoch}.pt'))
    wandb.log_artifact(artifact)
    
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
    
def load_checkpoint(model_class: torch.nn.Module, path: str, pretrain: bool = False, freeze: bool = False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        checkpoint = torch.load(path)
    
    model = model_class(pretrain=pretrain, n_classes=checkpoint['n_classes'], emb_size=checkpoint['emb_size'], s=checkpoint['s'], m=checkpoint['m'])
    
    if pretrain:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Carregar sem logits
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        if freeze:
            # Congelar input_layer e body
            for param in model.input_layer.parameters():
                param.requires_grad = False

            for param in model.body.parameters():
                param.requires_grad = False
        
    return model

# --------------------------------------------------------------------------------------------------------
    
def parse_args():
    parser = argparse.ArgumentParser(description="Treinar a rede neural como classificador")
    parser.add_argument('--model', type=str, default='ArcFaceResNet50', help='Modelo a ser utilizado (default: ArcFaceResNet50)')
    parser.add_argument('--batch_size', type=int, default=128, help='Tamanho do batch (default: 128)')
    parser.add_argument('--accumulation', type=int, default=512, help='Acumulação de gradientes (default: 512)')
    parser.add_argument('--epochs', type=int, default=32, help='Número de epochs (default: 32)')
    parser.add_argument('--emb_size', type=int, default=512, help='Tamanho do embedding (default: 512)')
    parser.add_argument('--num_workers', type=int, default=1, help='Número de workers para o DataLoader (default: 1)')
    parser.add_argument('--train_dir', type=str, default='./data/CASIA/train', help='Caminho para o diretório de treino (default: ./data/CASIA/train)')
    parser.add_argument('--test_dir', type=str, default='./data/CASIA/test', help='Caminho para o diretório de teste (default: ./data/CASIA/test)')
    parser.add_argument('--eval_dir', type=str, default='./data/LFW', help='Caminho para o diretório de eval (default: ./data/LFW)')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/', help='Caminho para salvar os checkpoints (default: ./checkpoints/)')
    parser.add_argument('--compile', action='store_true', help='Se deve compilar o modelo (default: False)')
    parser.add_argument('--wandb', action='store_true', help='Se está rodando com o Weights & Biases (default: False)')
    parser.add_argument('--random_state', type=int, default=42, help='Seed para o random (default: 42)')
    parser.add_argument('--lr', type=float, default=1e-1, help='Taxa de aprendizado inicial (default: 1e-1)')
    parser.add_argument('--s', type=float, default=64.0, help="Fator de escala dos embeddings (default: 64)")
    parser.add_argument('--m', type=float, default=0.5, help="Margem dos embeddings (default: 0.5)")
    parser.add_argument('--reduction_factor', type=float, default=0.1, help="Fator de redução da taxa de aprendizado (default: 0.1)")
    parser.add_argument('--reduction_epochs', nargs='+', type=int, default=[20, 28], help="Epochs para redução da taxa de aprendizado (default: [20, 28])")
    parser.add_argument('--warmup_epochs', type=int, default=5, help="Epochs para warmup da taxa de aprendizado (default: 5)")
    parser.add_argument('--warmup_lr', type=float, default=2.5e-2, help="Taxa de aprendizado inicial para warmup (default: 2.5e-2)")
    parser.add_argument('--pretrain', action='store_true', help='Se está rodando em modo de pré-treino (default: False)')
    parser.add_argument('--restore_path', type=str, default=None, help='Caminho para o checkpoint a ser restaurado (default: None)')
    parser.add_argument('--freeze', action='store_true', help='Se deve congelar as camadas de input_layer e body (default: False)')
        
    return parser.parse_args()
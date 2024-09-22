import argparse
from PIL import Image
import wandb
import os
import random
import numpy as np

from sklearn.metrics import precision_recall_fscore_support

import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomCrop, RandomHorizontalFlip, RandomRotation, ColorJitter
from torch.utils.data import Dataset
from torch.amp import autocast

transform = Compose([
    Resize([112, 112]),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

aug_transform = Compose([
    Resize([128, 128]),
    RandomCrop([112, 112]),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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

def evaluate(model, val_dataloader, criterion, dtype=torch.bfloat16, device='cuda'):
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
    def __init__(self, optimizer, reduction_epochs=None, reduction_factor=0.1, warmup_epochs=5, last_epoch=-1, warmup_lr=1e-6):
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

# --------------------------------------------------------------------------------------------------------
    
def parse_args():
    parser = argparse.ArgumentParser(description="Treinar a rede neural como classificador")
    parser.add_argument('--batch_size', type=int, default=128, help='Tamanho do batch (default: 128)')
    parser.add_argument('--accumulation', type=int, default=512, help='Acumulação de gradientes (default: 512)')
    parser.add_argument('--epochs', type=int, default=32, help='Número de epochs (default: 32)')
    parser.add_argument('--emb_size', type=int, default=512, help='Tamanho do embedding (default: 512)')
    parser.add_argument('--num_workers', type=int, default=1, help='Número de workers para o DataLoader (default: 1)')
    parser.add_argument('--train_df', type=str, default='./data/CASIA/train.csv', help='Caminho para o CSV de treino (default: ./data/CASIA/train.csv)')
    parser.add_argument('--test_df', type=str, default='./data/CASIA/test.csv', help='Caminho para o CSV de teste (default: ./data/CASIA/test.csv)')
    parser.add_argument('--images_path', type=str, default='./data/', help='Caminho para as imagens do dataset (default: ./data/CASIA/casia-faces/)')
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
        
    return parser.parse_args()
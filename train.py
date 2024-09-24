import numpy as np
import torchvision.datasets
from tqdm import tqdm
import os
import wandb
from collections import Counter

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
import wandb.wandb_torch

from models.arcfaceresnet50 import ArcFaceResNet50
from models.irse50 import IR_SE_50

from utils import parse_args, transform, aug_transform, evaluate, ArcFaceLRScheduler, FocalLoss, save_model_artifact, set_seed

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
DTYPE = torch.bfloat16
if torch.cuda.is_available():
    gpu_properties = torch.cuda.get_device_properties(0)

    if gpu_properties.major < 8:
        DTYPE = torch.float16
        
USING_WANDB = False

model_map = {
    'arcfaceresnet50': ArcFaceResNet50,
    'irse50': IR_SE_50
}

# --------------------------------------------------------------------------------------------------------
    
def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: torch.optim.lr_scheduler,
    accumulation_steps: int,
    epochs: int,
    dtype: torch.dtype,
    device: str,
    checkpoint_path: str
):
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        num_batches = len(train_dataloader) // accumulation_steps
        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
            
        for step, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device, dtype=dtype), labels.to(device)
            
            with autocast(dtype=dtype, device_type=device):
                outputs = model(inputs, labels)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                progress_bar.update(1)
                
            running_loss += loss.item() * accumulation_steps
        
        progress_bar.close()
        
        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy, epoch_precision, epoch_recall, epoch_f1, val_loss = evaluate(model, test_dataloader, criterion, dtype=dtype, device=device)
        
        if USING_WANDB:
            wandb.log({
                'epoch': epoch+1,
                'train_loss': epoch_loss,
                'val_loss': val_loss,
                'accuracy': epoch_accuracy,
                'precision': epoch_precision,
                'recall': epoch_recall,
                'f1': epoch_f1,
                'lr': optimizer.param_groups[0]['lr']
            })
            
        print(f"Epoch [{epoch+1}/{epochs}] | loss: {epoch_loss:.6f} | val_loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"Metrics: accuracy: {epoch_accuracy:.4f} | precision: {epoch_precision:.4f} | recall: {epoch_recall:.4f} | f1: {epoch_f1:.4f}\n")
        model.save_checkpoint(checkpoint_path, f'epoch_{epoch+1}.pt')
        if USING_WANDB: 
            save_model_artifact(checkpoint_path, epoch+1)
            
        scheduler.step()
        
# --------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()
    
    model = args.model
    batch_size = args.batch_size
    accumulation = args.accumulation
    epochs = args.epochs
    emb_size = args.emb_size
    num_workers = args.num_workers
    train_dir = args.train_dir
    test_dir = args.test_dir
    CHECKPOINT_PATH = args.checkpoint_path
    compile = args.compile
    USING_WANDB = args.wandb
    random_state = args.random_state
    lr = args.lr
    s = args.s
    m = args.m
    reduction_factor = args.reduction_factor
    reduction_epochs = args.reduction_epochs
    warmup_epochs = args.warmup_epochs
    warmup_lr = args.warmup_lr
    
    # Seed para reproducibilidade
    set_seed(random_state)
    
    accumulation_steps = accumulation // batch_size
    
    config = {
        'model': model,
        'batch_size': batch_size,
        'accumulation': accumulation,
        'epochs': epochs,
        'emb_size': emb_size,
        'num_workers': num_workers,
        'train_dir': train_dir,
        'test_dir': test_dir,
        'checkpoint_path': CHECKPOINT_PATH,
        'compile': compile,
        'random_state': random_state,
        'lr': lr,
        's': s,
        'm': m,
        'reduction_factor': reduction_factor,
        'reduction_epochs': reduction_epochs,
        'warmup_epochs': warmup_epochs,
        'warmup_lr': warmup_lr
    }

    if USING_WANDB:
        wandb.login(key=os.environ['WANDB_API_KEY'])
        wandb.init(project='arcface', config=config)

    # ------
    
    # Train
    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=aug_transform)
    
    n_classes = len(train_dataset.classes)    
    class_counts = Counter(train_dataset.targets)
    weights = [1.0 / class_counts[i] if class_counts[i] > 0 else 0 for i in range(len(class_counts))]
    sample_weights = torch.DoubleTensor([weights[label] for label in train_dataset.targets])
    weighted_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=num_workers, 
        sampler=weighted_sampler)

    # Test
    test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=num_workers)
    
    # Loss
    criterion = FocalLoss(gamma=2)
    
    # Modelo
    if model not in model_map:
        raise ValueError(f'Modelo {model} não encontrado')
    
    model = model_map[model](emb_size=emb_size, n_classes=n_classes, s=s, m=m).to(device)
        
    if compile:
        model = torch.compile(model)
    
    # Scaler, Otimizador e Scheduler
    scaler = GradScaler(init_scale=2**14)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    scheduler = ArcFaceLRScheduler(optimizer, warmup_lr=warmup_lr, warmup_epochs=warmup_epochs+1, reduction_epochs=reduction_epochs, reduction_factor=reduction_factor, last_epoch=-1)

    # -----
    
    print(f'\nModel: {model.__class__.__name__} | Params: {model.num_params/1e6:.2f}M')
    print(f'Device: {device}')
    print(f'Device name: {torch.cuda.get_device_name()}')
    print(f'Using tensor type: {DTYPE}')
    
    print(f'\nImagens: {len(train_dataset)} | Identidades: {n_classes} | imgs/id: {len(train_dataset) / n_classes:.2f}\n')
    
    train(
        model              = model,
        train_dataloader   = train_dataloader,
        test_dataloader    = test_dataloader,
        criterion          = criterion,
        optimizer          = optimizer,
        scaler             = scaler,
        scheduler          = scheduler,
        accumulation_steps = accumulation_steps,
        epochs             = epochs,
        dtype              = DTYPE,
        device             = device,
        checkpoint_path    = CHECKPOINT_PATH
    )
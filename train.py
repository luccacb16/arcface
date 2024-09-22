import pandas as pd
from tqdm import tqdm
import os
import wandb
from collections import Counter

import torch
import torchvision
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler
from torch.amp import GradScaler, autocast
import wandb.wandb_torch

from models.arcfaceresnet50 import ArcFaceResNet50

from utils import parse_args, transform, aug_transform, CustomDataset, evaluate, ArcFaceLRScheduler, FocalLoss, save_model_artifact, set_seed

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

DTYPE = torch.bfloat16
if torch.cuda.is_available():
    gpu_properties = torch.cuda.get_device_properties(0)

    if gpu_properties.major < 8:
        DTYPE = torch.float16
        
USING_WANDB = False

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def train(
    rank,
    world_size,
    model: nn.Module,
    train_dataset: CustomDataset,
    test_dataset: CustomDataset,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: torch.optim.lr_scheduler,
    accumulation_steps: int,
    epochs: int,
    batch_size: int,
    num_workers: int,
    dtype: torch.dtype,
    checkpoint_path: str
):
    setup(rank, world_size)
    device = f'cuda:{rank}'
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    
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
        sampler=weighted_sampler
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_norm = 0.0
        optimizer.zero_grad()
        
        num_batches = len(train_dataloader) // accumulation_steps
        if rank == 0:
            progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
            
        for step, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device, dtype=dtype), labels.to(device)
            
            with autocast(dtype=dtype, device_type='cuda'):
                outputs = model(inputs, labels)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if rank == 0:
                    progress_bar.update(1)
                
            running_loss += loss.item() * accumulation_steps
        
        if rank == 0:
            progress_bar.close()
        
        # Reduce loss
        epoch_loss = torch.tensor(running_loss / len(train_dataloader), device=device)
        epoch_loss = reduce_tensor(epoch_loss, world_size)
        
        if rank == 0:
            epoch_accuracy, epoch_precision, epoch_recall, epoch_f1, val_loss = evaluate(model.module, test_dataloader, criterion, dtype=dtype, device=device)
            
            # Reduce evaluation metrics
            metrics = torch.tensor([epoch_accuracy, epoch_precision, epoch_recall, epoch_f1, val_loss], device=device)
            metrics = reduce_tensor(metrics, world_size)
            epoch_accuracy, epoch_precision, epoch_recall, epoch_f1, val_loss = metrics.tolist()
        
            if USING_WANDB:
                wandb.log({
                    'epoch': epoch+1,
                    'train_loss': epoch_loss.item(),
                    'val_loss': val_loss,
                    'accuracy': epoch_accuracy,
                    'precision': epoch_precision,
                    'recall': epoch_recall,
                    'f1': epoch_f1,
                    'lr': optimizer.param_groups[0]['lr']
                })
            
            print(f"Epoch [{epoch+1}/{epochs}] | loss: {epoch_loss.item():.6f} | val_loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"Metrics: accuracy: {epoch_accuracy:.4f} | precision: {epoch_precision:.4f} | recall: {epoch_recall:.4f} | f1: {epoch_f1:.4f}\n")
            model.module.save_checkpoint(checkpoint_path, f'epoch_{epoch+1}.pt')
            if USING_WANDB: 
                save_model_artifact(checkpoint_path, epoch+1)
            
        scheduler.step()
    
    cleanup()

if __name__ == '__main__':
    args = parse_args()
    
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
    world_size = args.world_size if args.world_size else torch.cuda.device_count()
    
    # Seed for reproducibility
    set_seed(random_state)
    
    accumulation_steps = accumulation // batch_size
    
    config = {
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
        'world_size': world_size
    }

    if USING_WANDB:
        wandb.login(key=os.environ['WANDB_API_KEY'])
        wandb.init(project='arcface', config=config)

    # ImageFolder
    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=aug_transform)
    test_dataset = torchvision.datasets.ImageFolder(test_dir, transform=transform)
    n_classes = len(train_dataset.classes)

    # Loss
    criterion = FocalLoss(gamma=2)
    
    # Model
    model = ArcFaceResNet50(emb_size=emb_size, n_classes=n_classes, s=s, m=m)
        
    if compile:
        model = torch.compile(model)
    
    # Scaler, Optimizer and Scheduler
    scaler = GradScaler()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = ArcFaceLRScheduler(optimizer, warmup_epochs=warmup_epochs+1, reduction_epochs=reduction_epochs, reduction_factor=reduction_factor, last_epoch=-1)

    print(f'\nModel: {model.__class__.__name__} | Params: {model.num_params/1e6:.2f}M')
    print(f'Number of GPUs: {world_size}')
    print(f'Using tensor type: {DTYPE}')
    
    print(f'\nImages: {len(train_dataset)} | Identities: {n_classes} | imgs/id: {len(train_dataset) / n_classes:.2f}\n')
    
    torch.multiprocessing.spawn(
        train,
        args=(
            world_size,
            model,
            train_dataset,
            test_dataset,
            criterion,
            optimizer,
            scaler,
            scheduler,
            accumulation_steps,
            epochs,
            batch_size,
            num_workers,
            DTYPE,
            CHECKPOINT_PATH
        ),
        nprocs=world_size,
        join=True
    )
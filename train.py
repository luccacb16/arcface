import pandas as pd
from tqdm import tqdm
import os
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import wandb.wandb_torch

from models.arcfaceresnet50 import ArcFaceResNet50

from utils import parse_args, transform, aug_transform, CustomDataset, evaluate, ArcFaceLRScheduler, FocalLoss, save_model_artifact, set_seed

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
DTYPE = torch.bfloat16
if torch.cuda.is_available():
    gpu_properties = torch.cuda.get_device_properties(0)

    if gpu_properties.major < 8:
        DTYPE = torch.float16
        
USING_WANDB = False

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
        epoch_norm = 0.0
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
                scaler.unscale_(optimizer)
                epoch_norm += torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                progress_bar.update(1)
                
            running_loss += loss.item() * accumulation_steps
        
        progress_bar.close()
        
        epoch_loss = running_loss / len(train_dataloader)
        epoch_norm = epoch_norm / num_batches
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
            
        print(f"Epoch [{epoch+1}/{epochs}] | loss: {epoch_loss:.6f} | val_loss: {val_loss:.6f} | norm: {epoch_norm:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"Metrics: accuracy: {epoch_accuracy:.4f} | precision: {epoch_precision:.4f} | recall: {epoch_recall:.4f} | f1: {epoch_f1:.4f}\n")
        model.save_checkpoint(checkpoint_path, f'epoch_{epoch+1}.pt')
        if USING_WANDB: 
            save_model_artifact(checkpoint_path, epoch+1)
            
        scheduler.step()
        
# --------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()
    
    batch_size = args.batch_size
    accumulation = args.accumulation
    epochs = args.epochs
    emb_size = args.emb_size
    num_workers = args.num_workers
    TRAIN_DF_PATH = args.train_df
    TEST_DF_PATH = args.test_df
    IMAGES_PATH = args.images_path
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
    
    # Seed para reproducibilidade
    set_seed(random_state)
    
    accumulation_steps = accumulation // batch_size
    
    config = {
        'batch_size': batch_size,
        'accumulation': accumulation,
        'epochs': epochs,
        'emb_size': emb_size,
        'num_workers': num_workers,
        'train_df_path': TRAIN_DF_PATH,
        'test_df_path': TEST_DF_PATH,
        'images_path': IMAGES_PATH,
        'checkpoint_path': CHECKPOINT_PATH,
        'compile': compile,
        'random_state': random_state,
        'lr': lr,
        's': s,
        'm': m,
        'reduction_factor': reduction_factor,
        'reduction_epochs': reduction_epochs
    }

    if USING_WANDB:
        wandb.login(key=os.environ['WANDB_API_KEY'])
        wandb.init(project='arcface', config=config)

    # ------
    
    # Dados
    train_df = pd.read_csv(TRAIN_DF_PATH)
    train_df['path'] = train_df['path'].apply(lambda x: os.path.join(IMAGES_PATH, x))
    n_classes = train_df['id'].nunique()
    
    test_df = pd.read_csv(TEST_DF_PATH)
    test_df['path'] = test_df['path'].apply(lambda x: os.path.join(IMAGES_PATH, x))
    test_df = test_df.sample(n=1024, random_state=random_state).reset_index(drop=True)
    
    # Datasets e Loaders
    train_dataset = CustomDataset(train_df, transform=aug_transform, dtype=DTYPE)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    
    test_dataset = CustomDataset(test_df, transform=transform, dtype=DTYPE)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    # Loss
    criterion = FocalLoss(gamma=2)
    
    # Modelo
    model = ArcFaceResNet50(emb_size=emb_size, n_classes=n_classes, s=s, m=m).to(device)
        
    if compile:
        model = torch.compile(model)
    
    # Scaler, Otimizador e Scheduler
    scaler = GradScaler()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    scheduler = ArcFaceLRScheduler(optimizer, warmup_epochs=warmup_epochs+1, reduction_epochs=reduction_epochs, reduction_factor=reduction_factor, last_epoch=-1)

    # -----
    
    print(f'\nModel: {model.__class__.__name__} | Params: {model.num_params/1e6:.2f}M')
    print(f'Device: {device}')
    print(f'Device name: {torch.cuda.get_device_name()}')
    print(f'Using tensor type: {DTYPE}')
    
    print(f'\nImagens: {len(train_df)} | Identidades: {n_classes} | imgs/id: {len(train_df) / n_classes:.2f}\n')
    
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
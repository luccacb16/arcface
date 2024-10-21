import pandas as pd
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
import wandb

from models.arcfaceresnet50 import ArcFaceResNet50
from models.inception_resnet_v1 import InceptionResNetV1
from models.ir_se_50 import IR_SE_50
from models.arcfaceresnet101 import ArcFaceResNet101

from utils import load_checkpoint, parse_args, transform, aug_transform, test, ArcFaceLRScheduler, FocalLoss, save_model_artifact, set_seed, WarmUpCosineAnnealingLR
from eval_utils import evaluate, EvalDataset

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
DTYPE = torch.bfloat16
if torch.cuda.is_available():
    gpu_properties = torch.cuda.get_device_properties(0)

    if gpu_properties.major < 8:
        DTYPE = torch.float16
        
USING_WANDB = False
SAVE_STEP = 5

model_map = {
    'arcfaceresnet50': ArcFaceResNet50,
    'inceptionresnetv1': InceptionResNetV1,
    'irse50': IR_SE_50,
    'arcfaceresnet100': ArcFaceResNet101
}

# --------------------------------------------------------------------------------------------------------
    
def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    eval_dataloader: DataLoader,
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
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                progress_bar.update(1)
                
            running_loss += loss.item() * accumulation_steps
        
        progress_bar.close()
        
        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy, epoch_precision, epoch_recall, epoch_f1, val_loss = test(model, test_dataloader, criterion, dtype=dtype, device=device)
        eval_val, eval_accuracy = evaluate(model, eval_dataloader, target_far=1e-3, device=device, dtype=dtype)

        if USING_WANDB:
            wandb.log({
                'epoch': epoch+1,
                'train_loss': epoch_loss,
                'val_loss': val_loss,
                'eval_val': eval_val,
                'eval_accuracy': eval_accuracy,
                'accuracy': epoch_accuracy,
                'precision': epoch_precision,
                'recall': epoch_recall,
                'f1': epoch_f1,
                'lr': optimizer.param_groups[0]['lr'],
                'eval_val': eval_val,
                'eval_accuracy': eval_accuracy
            })
            
        print(f"Epoch [{epoch+1}/{epochs}] | loss: {epoch_loss:.6f} | val_loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"Metrics: accuracy: {epoch_accuracy:.4f} | precision: {epoch_precision:.4f} | recall: {epoch_recall:.4f} | f1: {epoch_f1:.4f}")
        print(f"[LFW]: VAL@FAR1e-3: {eval_val:.4f} | Accuracy: {eval_accuracy:.4f}\n")
       
        model.save_checkpoint(checkpoint_path, f'epoch_{epoch+1}.pt')
        if (epoch+1) % SAVE_STEP == 0 and USING_WANDB:        
            save_model_artifact(checkpoint_path, epoch+1)
            
        scheduler.step()
        
# --------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()
    
    model_name = args.model
    batch_size = args.batch_size
    accumulation = args.accumulation
    epochs = args.epochs
    emb_size = args.emb_size
    num_workers = args.num_workers
    train_dir = args.train_dir
    test_dir = args.test_dir
    eval_dir = args.eval_dir
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
    pretrain = args.pretrain
    restore_path = args.restore_path
    
    # Seed para reproducibilidade
    set_seed(random_state)
    
    accumulation_steps = accumulation // batch_size
    
    config = {
        'model': model_name,
        'batch_size': batch_size,
        'accumulation': accumulation,
        'epochs': epochs,
        'emb_size': emb_size,
        'num_workers': num_workers,
        'train_dir': train_dir,
        'test_dir': test_dir,
        'eval_dir': eval_dir,
        'checkpoint_path': CHECKPOINT_PATH,
        'compile': compile,
        'random_state': random_state,
        'lr': lr,
        's': s,
        'm': m,
        'reduction_factor': reduction_factor,
        'reduction_epochs': reduction_epochs,
        'warmup_epochs': warmup_epochs,
        'warmup_lr': warmup_lr,
        'pretrain': pretrain,
        'restore_path': restore_path
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
    
    # Eval
    eval_pairs_df = pd.read_csv(os.path.join(eval_dir, 'pairsDevTest.csv')) 
    eval_dataset = EvalDataset(eval_dir=eval_dir, pairs_df=eval_pairs_df, transform=transform)
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=num_workers)
    
    # Loss
    if pretrain:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = FocalLoss(gamma=2)
    
    # Modelo
    if model_name.lower() not in model_map:
        raise ValueError(f'Modelo {model_name} não encontrado!')
    
    if pretrain:
        model = model_map[model_name.lower()](emb_size=emb_size, n_classes=n_classes, s=s, m=m, pretrain=pretrain).to(device)
    else:    
        model = load_checkpoint(
            model_class=model_map[model_name.lower()],
            path=restore_path,
            pretrain=pretrain
        ).to(device)
        
    if compile:
        model = torch.compile(model)
    
    # Scaler, Otimizador e Scheduler
    scaler = GradScaler(init_scale=2**14)

    # Adaptação para melhor convergência
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = WarmUpCosineAnnealingLR(optimizer, epochs=epochs, warmup_epochs=warmup_epochs, min_lr=warmup_lr, max_lr=lr)

    # -----
    
    print(f'Training mode: {"Pretrain" if pretrain else "ArcFace"}')
    print(f'Model: {model.__class__.__name__} | Params: {model.num_params/1e6:.2f}M')
    print(f'Device: {device}')
    print(f'Device name: {torch.cuda.get_device_name()}')
    print(f'Using tensor type: {DTYPE}')
    
    print(f'\n[TRAIN] Imagens: {len(train_dataset)} | Identidades: {n_classes}')
    print(f'[TEST] Imagens: {len(test_dataset)} | Identidades: {len(test_dataset.classes)}')
    print(f'[EVAL] Imagens: {len(eval_dataset)} | Identidades: {len(eval_dataset.classes)}\n')
    
    train(
        model              = model,
        train_dataloader   = train_dataloader,
        test_dataloader    = test_dataloader,
        eval_dataloader    = eval_dataloader,
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
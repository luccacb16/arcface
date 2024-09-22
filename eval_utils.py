import torch
import torch.nn as nn
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# get_pairs -> calcular distancias -> ROC -> accuracy -> dist stats

def get_pairs(ids_df: pd.DataFrame, n_pairs: int | None = None, random_state: int = 42) -> pd.DataFrame:
    if n_pairs is None:
        n_pairs = len(ids_df)

    # Configura o gerador de números aleatórios com uma semente fixa
    rng = np.random.default_rng(random_state)
    
    pairs = []
    i = 0
    while i < n_pairs // 2:
        # Escolhe aleatoriamente uma linha/índice do dataframe usando o RNG configurado
        row = ids_df.sample(1, random_state=rng).iloc[0]
        
        # Encontra outras imagens com o mesmo ID
        same_id_df = ids_df[(ids_df['id'] == row['id']) & ~(ids_df.index == row.name)]
        if same_id_df.empty:
            continue  # Se não houver outras imagens com o mesmo ID, tenta novamente
        
        # Seleciona aleatoriamente uma imagem com o mesmo ID e uma com um ID diferente
        same = same_id_df.sample(1, random_state=rng)
        diff = ids_df[ids_df['id'] != row['id']].sample(1, random_state=rng)

        # Adiciona os pares ao resultado final
        pairs.append([row['path'], same['path'].values[0], 1])  # 1 significa que são do mesmo ID
        pairs.append([row['path'], diff['path'].values[0], 0])  # 0 significa que são de IDs diferentes
        
        i += 1

    return pd.DataFrame(pairs, columns=['img1', 'img2', 'label'])

class PairsDataset(Dataset):
    def __init__(self, pairs_df, transform=None):
        self.pairs_df = pairs_df
        self.transform = transform

    def __len__(self):
        return len(self.pairs_df)

    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]
        
        img1 = Image.open(row['img1'])
        img2 = Image.open(row['img2'])
        label = row['label']

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label
    
def cosine_distance(x1, x2):
    return 1 - torch.nn.functional.cosine_similarity(x1, x2)

def calculate_distances(model: nn.Module, pairs: pd.DataFrame, batch_size=32, transform=None, device='cuda', metric='L2') -> pd.DataFrame:
    metric_map = {
        'L2': torch.nn.functional.pairwise_distance,
        'cosine': cosine_distance
    }
    if metric not in metric_map:
        raise ValueError(f'Metric {metric} not found')
    
    model.to(device)
    model.eval()
    
    dataset = PairsDataset(pairs, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    distances = []

    with torch.no_grad():
        for img1_batch, img2_batch, _ in dataloader:
            img1_batch, img2_batch = img1_batch.to(device), img2_batch.to(device)
            
            outputs1 = model.get_embedding(img1_batch)
            outputs2 = model.get_embedding(img2_batch)
            
            batch_distances = metric_map[metric](outputs1, outputs2)
            
            distances.extend(batch_distances.cpu().numpy())

    pairs['distance'] = distances
    return pairs

def plot_distribution_and_ROC(pairs: pd.DataFrame, model_name: str, target_far=1e-3, metric='L2') -> tuple:
    col_name = 'distance'
    
    # Cria uma figura com dois subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico de distribuição de distâncias
    pairs[pairs['label'] == 1][col_name].plot.kde(ax=ax[0])
    pairs[pairs['label'] == 0][col_name].plot.kde(ax=ax[0])
    ax[0].set_title(f'Distances distribution ({model_name}) - {metric}')
    ax[0].legend(['Positive', 'Negative'])
    
    # Cálculo da curva ROC e AUC
    fpr, tpr, thresholds = roc_curve(pairs['label'], -pairs['distance'])
    roc_auc = auc(fpr, tpr)
    
    # Encontra o threshold ótimo
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Threshold para FAR 1e-3
    far_idx = np.where(fpr <= target_far)[0][-1]
    far_threshold = thresholds[far_idx]

    # Plot da curva ROC
    ax[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title(f'Receiver Operating Characteristic for {model_name}')
    ax[1].legend(loc="lower right")

    # Mostra a figura completa
    plt.tight_layout()
    plt.show()
    
    return -optimal_threshold, -far_threshold
    
def accuracy(pairs: pd.DataFrame, threshold: float) -> float:
    return sum((pairs['distance'] < threshold) == pairs['label']) / len(pairs)

def VAL(pairs: pd.DataFrame, threshold: float) -> float:
    true_positives = pairs['label'] == 1
    classified_as_true = pairs['distance'] < threshold
    val = sum(true_positives & classified_as_true) / sum(true_positives)
    
    return val

def distance_stats(pairs: pd.DataFrame) -> tuple:
    col_name = 'distance'
    pos_mean = pairs[pairs['label'] == 1][col_name].mean()
    pos_std = pairs[pairs['label'] == 1][col_name].std()

    neg_mean = pairs[pairs['label'] == 0][col_name].mean()
    neg_std = pairs[pairs['label'] == 0][col_name].std()
    
    return pos_mean, pos_std, neg_mean, neg_std

def eval_epoch(
    model: nn.Module, 
    val_df: pd.DataFrame, 
    transform, 
    n_pairs: int = 1024, 
    batch_size: int = 32,
    device='cuda', 
    target_far: float = 1e-3,
    metric='L2',
    random_state=42
) -> None:
    
    pairs_df = get_pairs(val_df, n_pairs=n_pairs, random_state=random_state)
    pairs_dist = calculate_distances(model, pairs_df, transform=transform, device=device, batch_size=batch_size, metric=metric)
    
    _, low_far = plot_distribution_and_ROC(pairs_dist, model.__class__.__name__, metric=metric, target_far=target_far)
    
    acc = accuracy(pairs_dist, low_far)
    val = VAL(pairs_dist, low_far)
    
    print(f'Target FAR: {target_far:.0e} | Threshold: {low_far:.4f}')
    print(f'[{model.__class__.__name__}] Accuracy: {acc:.4f}')
    print(f'[{model.__class__.__name__}] VAL: {val:.4f}\n')
    
    # Calcula as estatísticas das distâncias
    pos_mean, pos_std, neg_mean, neg_std = distance_stats(pairs_dist)
    print(f'[{model.__class__.__name__}] Positive mean: {pos_mean:.4f} ± {pos_std:.4f}')
    print(f'[{model.__class__.__name__}] Negative mean: {neg_mean:.4f} ± {neg_std:.4f}')
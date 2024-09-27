import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet101
import warnings

from .arcface_layer import ArcMarginProduct

class ArcFaceResNet101(nn.Module):
    def __init__(self, n_classes=0, emb_size=512, s=64.0, m=0.5):
        super(ArcFaceResNet101, self).__init__()
        resnet = resnet101()

        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(2048, emb_size)
        self.bn2 = nn.BatchNorm1d(emb_size)
        
        self.arcface = ArcMarginProduct(in_features=emb_size, out_features=n_classes, s=s, m=m)

        self._initialize_weights()
        
        self.emb_size = emb_size
        self.n_classes = n_classes
        
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def forward(self, x, labels=None):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn2(x)
        
        if labels is not None:
            x = self.arcface(x, labels)
            
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def save_checkpoint(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path)
            
        model_state_dict = {k.replace('_orig_mod.', ''): v for k, v in self.state_dict().items()}
        
        checkpoint = {
            'state_dict': model_state_dict,
            'n_classes': self.n_classes,
            'emb_size': self.emb_size,
            'm': self.arcface.m,
            's': self.arcface.s
        }
        torch.save(checkpoint, os.path.join(path, filename))
    
    @staticmethod
    def load_checkpoint(path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            checkpoint = torch.load(path)
    
        model = ArcFaceResNet101(n_classes=checkpoint['n_classes'], emb_size=checkpoint['emb_size'], s=checkpoint['s'], m=checkpoint['m'])
        model.load_state_dict(checkpoint['state_dict'])
        return model
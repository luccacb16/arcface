import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet50
import warnings

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input, label):
        m = torch.tensor(self.m, device=input.device, dtype=input.dtype)

        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        phi = cosine * torch.cos(m) - sine * torch.sin(m)
        phi = torch.where(cosine > 0, phi, cosine)
        
        one_hot = torch.zeros(cosine.size(0), self.out_features, device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class ArcFaceResNet50(nn.Module):
    def __init__(self, n_classes=0, emb_size=512, s=64.0, m=0.5):
        super(ArcFaceResNet50, self).__init__()
        resnet = resnet50()

        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        self.fc1 = nn.Linear(2048, emb_size)
        self.bn1 = nn.BatchNorm1d(emb_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        
        self.arcface = ArcMarginProduct(emb_size, n_classes, s, m)

        self._initialize_weights()
        
        self.emb_size = emb_size
        self.n_classes = n_classes
        
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, labels=None):
        x = self.get_embedding(x)
        
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
                
    def get_embedding(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        return x
    
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
        
    def load_checkpoint(path):
        # Suprimindo o FutureWarning específico para torch.load
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            checkpoint = torch.load(path)
    
        model = ArcFaceResNet50(n_classes=checkpoint['n_classes'], emb_size=checkpoint['emb_size'], s=checkpoint['s'], m=checkpoint['m'])
        model.load_state_dict(checkpoint['state_dict'])
        return model
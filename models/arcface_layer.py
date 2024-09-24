import torch
import torch.nn as nn
from torch.nn import functional as F

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
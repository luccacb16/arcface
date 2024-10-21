import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sigmoid, Dropout, Sequential, Module
import os

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class Bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(Bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth)
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class IR_SE_50(nn.Module):
    def __init__(self, n_classes: int = 0, emb_size: int = 512, s: float = 64.0, m: float = 0.5, pretrain=False):
        super(IR_SE_50, self).__init__()
        
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        
        blocks = [
            [3, 64, 64],
            [4, 64, 128],
            [14, 128, 256],
            [3, 256, 512]
        ]
        
        modules = []
        for block in blocks:
            num_units, in_channel, depth = block
            for i in range(num_units):
                stride = 2 if i == 0 else 1
                modules.append(Bottleneck_IR_SE(in_channel, depth, stride))
                in_channel = depth
                
        self.body = Sequential(*modules)
        
        self.output_layer = Sequential(BatchNorm2d(512),
                                        Dropout(),
                                        Flatten(),
                                        Linear(512 * 7 * 7, emb_size),
                                        BatchNorm1d(emb_size))
        
        self.n_classes = n_classes
        self.emb_size = emb_size
        self.s = s
        self.m = m
        self.pretrain = pretrain
        
        if self.pretrain:
            self.logits = Linear(emb_size, n_classes)
        else:
            self.logits = ArcMarginProduct(in_features=emb_size, out_features=n_classes, s=s, m=m)
            
        self._initialize_weights()
        
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x, labels=None):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        
        if self.pretrain:
            return self.logits(x)
        else:
            if labels is not None:
                return self.logits(x, labels)

        return x
                    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (BatchNorm1d, BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def save_checkpoint(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path)
        
        checkpoint = {
            'state_dict': self.state_dict(),
            'n_classes': self.n_classes,
            'emb_size': self.emb_size,
            's': self.s,
            'm': self.m
        }
        torch.save(checkpoint, os.path.join(path, filename))

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = torch.cos(torch.tensor(m))
        self.sin_m = torch.sin(torch.tensor(m))
        self.th = torch.cos(torch.tensor(torch.pi - m))
        self.mm = torch.sin(torch.tensor(torch.pi - m)) * m

    def forward(self, input, label):
        cosine = nn.functional.linear(nn.functional.normalize(input), nn.functional.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
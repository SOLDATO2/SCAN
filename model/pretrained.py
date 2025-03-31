import torch
import torch.nn as nn

class PretrainedResNeXt3DFeatureExtractor(nn.Module):
    def __init__(self):
        super(PretrainedResNeXt3DFeatureExtractor, self).__init__()
        #pip install pytorchvideo
        self.model = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=True)
        #remove a cabeça de classificação
        self.model = nn.Sequential(*list(self.model.children())[:-1])
    
    def forward(self, x):
        # x: [B, 3, T, H, W]
        features = self.model(x)
        return features.view(features.size(0), -1)

import torch
import torch.nn as nn
import torchvision.models as models

class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss: sqrt((x - y)^2 + eps^2)
    """
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    
    def forward(self, x, y):
        diff = x - y
        error = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(error)

class MeanShift(nn.Conv2d):
    """
    Subtrai mean e divide por std do ImageNet nas imagens RGB.
    sign=-1 => subtrai mean
    """
    def __init__(self, mean, std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = sign * torch.Tensor(mean) / torch.Tensor(std)
        for p in self.parameters():
            p.requires_grad = False

class VGG19FeatureExtractor(nn.Module):
    """
    vgg19(pretrained=True).features -> cortar até certo conv_index
    Subtrai mean/std do ImageNet, usa MSE nas features.
    """
    def __init__(self, conv_index='44'):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        modules = [m for m in vgg]
        # '22' => [:8], '33' => [:16], '44' => [:26], '54' => [:35]
        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8])   # conv2_2
        elif conv_index == '33':
            self.vgg = nn.Sequential(*modules[:16]) # conv3_3
        elif conv_index == '44':
            self.vgg = nn.Sequential(*modules[:26]) # conv4_4
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35]) # conv5_4
        else:
            self.vgg = nn.Sequential(*modules[:26])

        # Mean e std de ImageNet
        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std  = (0.229, 0.224, 0.225)
        self.sub_mean = MeanShift(vgg_mean, vgg_std)
        
        # Congela parâmetros
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x em [-1,1], converter p/ [0,1] e subtrair mean:
        x_01 = (x + 1.0) * 0.5
        x_01 = torch.clamp(x_01, 0, 1)
        x_01 = self.sub_mean(x_01)
        return self.vgg(x_01)

class PerceptualLoss(nn.Module):
    """
    Faz forward no VGG19 Feature Extractor e calcula MSE das features.
    """
    def __init__(self, conv_index='44'):
        super().__init__()
        self.vgg_extractor = VGG19FeatureExtractor(conv_index=conv_index)
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        with torch.no_grad():
            target_feat = self.vgg_extractor(target).detach()
        pred_feat = self.vgg_extractor(pred)
        loss = self.mse(pred_feat, target_feat)
        return loss

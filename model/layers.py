import torch
import torch.nn as nn
from model.attention import CALayer, SpatialAttention

class ConvNorm(nn.Module):
    """
    Convolução + ReflectionPad2d + (BN ou IN opcional)
    """
    def __init__(self, in_feat, out_feat, kernel_size, stride=1, norm=False):
        super(ConvNorm, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(in_feat, out_feat, stride=stride, kernel_size=kernel_size, bias=True)

        self.norm = None
        if norm == 'IN':
            self.norm = nn.InstanceNorm2d(out_feat, track_running_stats=True)
        elif norm == 'BN':
            self.norm = nn.BatchNorm2d(out_feat)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        if self.norm is not None:
            out = self.norm(out)
        return out

class UpConvNorm(nn.Module):
    """
    Upsample feito como : 'shuffle', 'transpose' ou 'bilinear'.
    Agora usando nn.PixelShuffle no modo 'shuffle'.
    """
    def __init__(self, in_channels, out_channels, mode='shuffle', norm=False):
        super(UpConvNorm, self).__init__()
        if mode == 'transpose':
            self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        elif mode == 'shuffle':
            self.upconv = nn.Sequential(
                ConvNorm(in_channels, 4*out_channels, kernel_size=3, stride=1, norm=norm),
                nn.PixelShuffle(2)
            )
        else:
            self.upconv = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                ConvNorm(in_channels, out_channels, kernel_size=1, stride=1, norm=norm))
    
    def forward(self, x):
        return self.upconv(x)

class RCAB(nn.Module):
    """
    Residual Channel Attention Block (RCAB),
    agora incluindo SpatialAttention.
    """
    def __init__(self, in_feat, out_feat, kernel_size, reduction, bias=True,
                 norm=False, act=nn.ReLU(True), downscale=False, return_ca=False):
        super(RCAB, self).__init__()
        self.downscale = downscale
        self.return_ca = return_ca

        stride = 2 if downscale else 1

        self.conv1 = ConvNorm(in_feat, out_feat, kernel_size, stride=stride, norm=norm)
        self.act = act
        self.conv2 = ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm)
        self.ca = CALayer(out_feat, reduction=16)
        self.sa = SpatialAttention(kernel_size=7)

        if downscale:
            self.downConv = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = self.ca(out)
        out = self.sa(out)

        if self.downscale:
            res = self.downConv(res)
        out = out + res

        if self.return_ca:
            return out, None
        else:
            return out

class ResidualGroup(nn.Module):
    """
    Grupo de n_resblocks, estilo RCAB
    """
    def __init__(self, Block, n_resblocks, n_feat, kernel_size, reduction, act, norm=False):
        super(ResidualGroup, self).__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_resblocks):
            blk = Block(n_feat, n_feat, kernel_size, reduction, bias=True, norm=norm, act=act)
            self.blocks.append(blk)
        self.post_conv = ConvNorm(n_feat, n_feat, kernel_size, stride=1, norm=norm)

    def forward(self, x):
        out = x
        for blk in self.blocks:
            out = blk(out)
        out = self.post_conv(out)
        return x + out

class Interpolation(nn.Module):
    """
    Interpolação: une feats do par (f1,f3), passa por n_resgroups.
    """
    def __init__(self, n_resgroups, n_resblocks, n_feats,
                 reduction=16, act=nn.LeakyReLU(0.2, True), norm=False):
        super(Interpolation, self).__init__()
        self.headConv = nn.Conv2d(n_feats*2, n_feats, kernel_size=3, padding=1)
        self.resgroups = nn.ModuleList()
        for _ in range(n_resgroups):
            rg = ResidualGroup(RCAB, n_resblocks, n_feats, kernel_size=3,
                               reduction=reduction, act=act, norm=norm)
            self.resgroups.append(rg)
        self.tailConv = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)

    def forward(self, f1, f3):
        x = torch.cat([f1, f3], dim=1)
        x = self.headConv(x)

        out = x
        for rg in self.resgroups:
            out = rg(out)

        out = out + x
        out = self.tailConv(out)
        return out

class Encoder(nn.Module):
    """
    Encoder original do SCAN_EncDec:
     - 3 -> 32 -> 64 -> 128 -> 192
     - Bloco Interpolation com 5 resgroups * 12 RCAB cada
    """
    def __init__(self, in_channels=3, nf_start=32, norm=False):
        super(Encoder, self).__init__()
        relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.body = nn.Sequential(
            ConvNorm(in_channels, nf_start, 7, stride=1, norm=norm),
            relu,
            ConvNorm(nf_start, nf_start*2, 5, stride=2, norm=norm),
            relu,
            ConvNorm(nf_start*2, nf_start*4, 5, stride=2, norm=norm),
            relu,
            ConvNorm(nf_start*4, nf_start*6, 5, stride=2, norm=norm)
        )
        self.interpolate = Interpolation(
            n_resgroups=5,
            n_resblocks=12,
            n_feats=nf_start*6,
            reduction=16,
            act=relu,
            norm=norm
        )

    def forward(self, x1, x2):
        feats1 = self.body(x1)
        feats2 = self.body(x2)
        feats = self.interpolate(feats1, feats2)
        return feats

class Decoder(nn.Module):
    """
    Decoder original do SCAN_EncDec:
    - 192 -> 128 -> 64 -> 32 -> 3
    - 3 UpConvNorm + Conv final
    """
    def __init__(self, in_channels=192, out_channels=3, norm=False, up_mode='shuffle'):
        super(Decoder, self).__init__()
        relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.body = nn.Sequential(
            UpConvNorm(in_channels, 128, mode=up_mode, norm=norm),
            nn.LeakyReLU(0.2, inplace=True),
            UpConvNorm(128, 64, mode=up_mode, norm=norm),
            nn.LeakyReLU(0.2, inplace=True),
            UpConvNorm(64, 32, mode=up_mode, norm=norm),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=7, padding=3)
        )

    def forward(self, feats):
        return self.body(feats)

class SCAN_EncDec(nn.Module):
    """
    - Recebe input_6c => separa x1 (3 canais) + x2 (3 canais)
    - Envia cada um ao encoder
    - Interpola + Passa no decoder => saida 3 canais
    """
    def __init__(self, nf_start=32):
        super(SCAN_EncDec, self).__init__()
        self.encoder = Encoder(in_channels=3, nf_start=nf_start, norm=False)
        self.decoder = Decoder(
            in_channels=nf_start*6,
            out_channels=3,
            norm=False,
            up_mode='shuffle'
        )

    def forward(self, x):
        x1 = x[:, :3, ...]
        x2 = x[:, 3:, ...]
        
        #x1, m1 = sub_mean(x1)
        #x2, m2 = sub_mean(x2)

        feats = self.encoder(x1, x2)
        out = self.decoder(feats)
        
        #out = out + (m1 + m2) / 2.0
        
        return out

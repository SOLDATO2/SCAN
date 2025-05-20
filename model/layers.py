import torch
import torch.nn as nn
# Atenção de canal e espacial comentadas/removidas
# from model.attention import CALayer, SpatialAttention

class ConvNorm(nn.Module):
    """
    Convolução + ReflectionPad2d + (BN ou IN opcional)
    """
    def __init__(self, in_feat, out_feat, kernel_size, stride=1, norm=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_feat, out_feat,
                              kernel_size=kernel_size,
                              stride=stride,
                              bias=True)
        if norm == 'IN':
            self.norm = nn.InstanceNorm2d(out_feat, track_running_stats=True)
        elif norm == 'BN':
            self.norm = nn.BatchNorm2d(out_feat)
        else:
            self.norm = None

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class UpConvNorm(nn.Module):
    """
    Upsample usando ConvTranspose2d.
    """
    def __init__(self, in_ch, out_ch, norm=False):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_ch, out_ch,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1)
        # opcional: norm após o upsample
        self.norm = nn.BatchNorm2d(out_ch) if norm == 'BN' else None

    def forward(self, x):
        x = self.upconv(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class Encoder(nn.Module):
    """
    Encoder simples:
      - extrai features de cada frame
      - concatena e reduz canais via conv
    """
    def __init__(self, in_channels=3, nf_start=32, norm=False):
        super().__init__()
        relu = nn.LeakyReLU(0.2, inplace=True)
        # corpo de down-sampling
        self.body = nn.Sequential(
            ConvNorm(in_channels, nf_start, 7, stride=1, norm=norm),
            relu,
            ConvNorm(nf_start, nf_start*2, 5, stride=2, norm=norm),
            relu,
            ConvNorm(nf_start*2, nf_start*4, 5, stride=2, norm=norm),
            relu,
            ConvNorm(nf_start*4, nf_start*6, 5, stride=2, norm=norm),
            relu,
        )
        # conv para fundir features de I1 e I2
        out_feats = nf_start * 6
        self.fuse = nn.Conv2d(out_feats * 2,
                               out_feats,
                               kernel_size=3,
                               padding=1)

    def forward(self, x1, x2):
        f1 = self.body(x1)           # [B, out_feats, H/8, W/8]
        f2 = self.body(x2)           # [B, out_feats, H/8, W/8]
        x = torch.cat([f1, f2], dim=1)  # [B, 2*out_feats, H/8, W/8]
        return self.fuse(x)             # [B, out_feats, H/8, W/8]


class Decoder(nn.Module):
    """
    Decoder simples:
      - 3× upsample (ConvTranspose2d) + LeakyReLU
      - conv final 7×7 → 3 canais
    """
    def __init__(self, in_ch, out_ch=3, norm=False, up_mode='transpose'):
        super().__init__()
        relu = nn.LeakyReLU(0.2, inplace=True)
        self.body = nn.Sequential(
            UpConvNorm(in_ch, 128, norm=norm),
            relu,
            UpConvNorm(128, 64, norm=norm),
            relu,
            UpConvNorm(64,  32, norm=norm),
            relu,
            nn.Conv2d(32, out_ch, kernel_size=7, padding=3)
        )

    def forward(self, x):
        return self.body(x)


class SCAN_EncDec(nn.Module):
    """
    Encoder–Decoder puro sem residual groups nem atenção.
    Entrada: [B,6,H,W] (I1||I2)
    Saída:   [B,3,H,W]
    """
    def __init__(self, nf_start=32):
        super().__init__()
        self.encoder = Encoder(in_channels=3, nf_start=nf_start, norm=False)
        self.decoder = Decoder(in_ch=nf_start * 6,
                               out_ch=3,
                               norm=False)

    def forward(self, x):
        # separa I1 e I2
        x1, x2 = x[:, :3], x[:, 3:]

        # padding dinâmico em inferência, se precisar
        if not self.training:
            from model.common import InOutPaddings
            pad_in, pad_out = InOutPaddings(x1)
            x1, x2 = pad_in(x1), pad_in(x2)

        feats = self.encoder(x1, x2)
        out   = self.decoder(feats)

        if not self.training:
            out = pad_out(out)

        return out

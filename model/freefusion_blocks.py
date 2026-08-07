"""
Building blocks ported from FreeFusion (references/FreeFusion/MMFNet.py).
Adapted channel dimensions from [32, 64, 128] to [48, 96, 192] for Text-IF.
"""
import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1,
                 bias=True, norm=False, relu=True):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False
        padding = kernel_size // 2
        layers = []
        layers.append(nn.Conv2d(in_channel, out_channel, kernel_size,
                                padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class SALayer(nn.Module):
    """SE-style channel attention with PReLU activation."""
    def __init__(self, channel, reduction=4, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=bias),
            nn.PReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.conv_du(x)


class FFBlock(nn.Module):
    """Feature Fusion Block with learnable channel-wise attention weights."""
    def __init__(self, in_channels, out_channels):
        super(FFBlock, self).__init__()
        self.conv_1_1 = nn.Sequential(
            BasicConv(in_channels * 2, in_channels * 2, kernel_size=3, relu=True),
            BasicConv(in_channels * 2, out_channels, kernel_size=3, relu=True),
            BasicConv(out_channels, out_channels, kernel_size=3, relu=True),
        )
        self.conv_1_2 = nn.Sequential(
            BasicConv(in_channels * 2, in_channels * 2, kernel_size=3, relu=True),
            BasicConv(in_channels * 2, out_channels, kernel_size=3, relu=True),
            BasicConv(out_channels, out_channels, kernel_size=3, relu=True),
        )
        self.channel_weights_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.channel_weights_2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_2 = BasicConv(in_channels * 2, out_channels, kernel_size=3, relu=True)

    def forward(self, en1, en2):
        cat_1_1 = torch.cat([en1, en2], dim=1)
        shallow_conv_1_1 = self.conv_1_1(cat_1_1)
        shallow_conv_1_2 = self.conv_1_2(cat_1_1)

        channel_weights_1 = self.channel_weights_1(shallow_conv_1_1)
        channel_weights_2 = self.channel_weights_2(shallow_conv_1_2)
        x_1 = en1 * channel_weights_1
        x_2 = en2 * channel_weights_2
        cat_2 = torch.cat([x_1, x_2], dim=1)
        fus = self.conv_2(cat_2)
        return fus


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention producing a single-channel mask in [0, 1].

    Channel-wise max and avg pools of the input are concatenated (2 channels)
    and fed through a 7x7 conv. Parameter count is independent of input
    channels: 2 * 1 * 7 * 7 = 98 + 0 (bias=False).
    """
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)             # [B, 1, H, W]
        mx, _ = torch.max(x, dim=1, keepdim=True)            # [B, 1, H, W]
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))  # [B, 1, H, W]


class FFBlockSCA(nn.Module):
    """Feature Fusion Block with Spatial-Channel Joint Attention.

    Identical structure to FFBlock when use_spatial=False. When
    use_spatial=True, adds a shared CBAM-style spatial attention branch
    that modulates both modalities jointly via a single [B, 1, H, W] mask.
    Per-modality channel attention (channel_weights_1/2) is preserved, so
    modality specificity is encoded by the channel branch while the spatial
    branch captures scene-level foreground/background structure.

    Args:
        in_channels: input channel count per modality (C).
        out_channels: output channel count.
        use_spatial: when True, enable the spatial attention branch. When
            False, this module is numerically and structurally identical
            to FFBlock (clean ablation control).
    """
    def __init__(self, in_channels, out_channels, use_spatial: bool = True):
        super(FFBlockSCA, self).__init__()
        self.use_spatial = use_spatial

        # Identical to FFBlock
        self.conv_1_1 = nn.Sequential(
            BasicConv(in_channels * 2, in_channels * 2, kernel_size=3, relu=True),
            BasicConv(in_channels * 2, out_channels, kernel_size=3, relu=True),
            BasicConv(out_channels, out_channels, kernel_size=3, relu=True),
        )
        self.conv_1_2 = nn.Sequential(
            BasicConv(in_channels * 2, in_channels * 2, kernel_size=3, relu=True),
            BasicConv(in_channels * 2, out_channels, kernel_size=3, relu=True),
            BasicConv(out_channels, out_channels, kernel_size=3, relu=True),
        )
        self.channel_weights_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.channel_weights_2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_2 = BasicConv(in_channels * 2, out_channels, kernel_size=3, relu=True)

        # New: shared spatial attention branch
        if use_spatial:
            self.spatial_attn = SpatialAttention(kernel_size=7)

    def forward(self, en1, en2):
        cat_1_1 = torch.cat([en1, en2], dim=1)
        sconv_1 = self.conv_1_1(cat_1_1)
        sconv_2 = self.conv_1_2(cat_1_1)

        w_c1 = self.channel_weights_1(sconv_1)
        w_c2 = self.channel_weights_2(sconv_2)

        if self.use_spatial:
            ctx = torch.cat([sconv_1, sconv_2], dim=1)       # [B, 2C, H, W]
            w_s = self.spatial_attn(ctx)                     # [B, 1, H, W]
            x_1 = en1 * w_c1 * w_s
            x_2 = en2 * w_c2 * w_s
        else:
            x_1 = en1 * w_c1
            x_2 = en2 * w_c2

        return self.conv_2(torch.cat([x_1, x_2], dim=1))


class FDBlock(nn.Module):
    """Feature Decoupled Block.
    SE-attends modality features and subtracts from fused features,
    producing a residual that approximates the other modality.
    """
    def __init__(self, in_channels):
        super(FDBlock, self).__init__()
        self.sa_0 = SALayer(in_channels[0], reduction=4, bias=False)
        self.sa_1 = SALayer(in_channels[1], reduction=4, bias=False)
        self.sa_2 = SALayer(in_channels[2], reduction=4, bias=False)

    def forward(self, fus_fea, modality_fea):
        m_0 = self.sa_0(modality_fea[0])
        x_0 = fus_fea[0] - m_0

        m_1 = self.sa_1(modality_fea[1])
        x_1 = fus_fea[1] - m_1

        m_2 = self.sa_2(modality_fea[2])
        x_2 = fus_fea[2] - m_2

        return [x_0, x_1, x_2]


class ReconHead(nn.Module):
    """Lightweight shared reconstruction decoder.
    Maps 3-level features (deepest to shallowest) to a 3-channel image.

    Args:
        in_channels: [L3_channels, L2_channels, L1_channels] = [192, 96, 48]
        out_channels: output image channels (default 3 for RGB)
    """
    def __init__(self, in_channels, out_channels=3):
        super(ReconHead, self).__init__()
        c3, c2, c1 = in_channels

        # Stage 3 -> 2
        self.up3_2 = nn.Sequential(
            nn.Conv2d(c3, c2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(c2 * 2, c2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Stage 2 -> 1
        self.up2_1 = nn.Sequential(
            nn.Conv2d(c2, c1, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Output
        self.head = nn.Conv2d(c1, out_channels, 3, padding=1)

    def forward(self, features):
        """
        Args:
            features: [feat_L3, feat_L2, feat_L1]
                      shapes: [B, 192, H/4, W/4], [B, 96, H/2, W/2], [B, 48, H, W]
        Returns:
            [B, out_channels, H, W]
        """
        x = self.up3_2(features[0])                    # [B, 96, H/2, W/2]
        x = self.fuse2(torch.cat([x, features[1]], 1))  # [B, 96, H/2, W/2]

        x = self.up2_1(x)                               # [B, 48, H, W]
        x = self.fuse1(torch.cat([x, features[2]], 1))  # [B, 48, H, W]

        return self.head(x)                              # [B, 3, H, W]

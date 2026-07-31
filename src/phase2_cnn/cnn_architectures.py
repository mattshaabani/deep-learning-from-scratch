"""
src/phase2_cnn/cnn_architectures.py

Three CNN architectures of increasing sophistication, all at
comparable depth, to isolate the effect of BatchNorm and residual
connections:

    PlainCNN       -- conv/relu/pool stack, no BatchNorm, no skip connections
    BatchNormCNN   -- same architecture + BatchNorm after every conv
    ResNetCNN      -- same depth budget, using residual blocks

Channel lists are built dynamically (not hardcoded) so any depth is
supported, from very shallow (depth=2) to very deep (depth=20+) --
this is what lets us run the depth ablation study without index errors.

Usage:
    from src.phase2_cnn.cnn_architectures import PlainCNN, BatchNormCNN, ResNetCNN
    model = ResNetCNN(num_classes=4, depth=12)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_channel_list(depth: int, max_channels: int = 128) -> list[int]:
    """
    Generate a channel progression for PlainCNN/BatchNormCNN at any depth.

    Starts at 3 (RGB input channels), doubles through the standard
    32 -> 64 -> 128 progression, then holds at max_channels for any
    additional depth requested. Always returns a list with at least
    depth+1 entries so channels[depth] is always safely indexable.
    """
    channels = [3, 32, 64]
    while len(channels) <= depth:
        channels.append(max_channels)
    return channels


def _build_resnet_channel_list(depth: int, max_channels: int = 128) -> list[int]:
    """
    Same progression logic as _build_channel_list, but starting from
    32 (the stem's output channels) rather than 3, since ResNetCNN's
    stem already handles the initial 3->32 conversion separately.
    """
    channels = [32, 64]
    while len(channels) < depth:
        channels.append(max_channels)
    return channels[:depth]


class PlainCNN(nn.Module):
    """
    Plain stacked CNN, no BatchNorm, no residual connections.
    Baseline for demonstrating vanishing gradients at depth.
    """
    def __init__(self, num_classes: int = 10, depth: int = 4):
        super().__init__()
        channels = _build_channel_list(depth)

        layers = []
        spatial_size = 32
        for i in range(depth):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            # Only pool if doing so keeps spatial size at 2x2 or larger
            if i % 2 == 1 and spatial_size >= 4:
                layers.append(nn.MaxPool2d(2))
                spatial_size //= 2

        self.features = nn.Sequential(*layers)
        self.pool     = nn.AdaptiveAvgPool2d((4, 4))
        self.fc       = nn.Linear(channels[depth] * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


class BatchNormCNN(nn.Module):
    """
    Same architecture as PlainCNN, but with BatchNorm after every
    convolution, before the activation -- the standard placement.

    BatchNorm formula (per channel, per batch):
        x_hat = (x - batch_mean) / sqrt(batch_var + eps)
        y = gamma * x_hat + beta          (gamma, beta are LEARNED)

    Normalizing activations keeps gradients from exploding/vanishing
    as they compound across layers, and lets us use higher learning
    rates than a plain (unnormalized) network can tolerate.
    """
    def __init__(self, num_classes: int = 10, depth: int = 4):
        super().__init__()
        channels = _build_channel_list(depth)

        layers = []
        spatial_size = 32
        for i in range(depth):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.ReLU())
            if i % 2 == 1 and spatial_size >= 4:
                layers.append(nn.MaxPool2d(2))
                spatial_size //= 2

        self.features = nn.Sequential(*layers)
        self.pool     = nn.AdaptiveAvgPool2d((4, 4))
        self.fc       = nn.Linear(channels[depth] * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


class ResidualBlock(nn.Module):
    """
    A single residual block: output = F(x) + x (identity shortcut)

    F(x) here is conv -> BN -> relu -> conv -> BN. If input and output
    channel counts differ, the shortcut path uses a 1x1 conv to match
    dimensions (a "projection shortcut") -- otherwise it's a pure
    identity connection.

    Why this solves vanishing gradients: during backprop, the gradient
    of the identity path is exactly 1 (d(x)/dx = 1), providing an
    unimpeded "gradient highway" straight back to earlier layers,
    regardless of how small F(x)'s own gradient might be.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity     # THE residual connection
        return F.relu(out)


class ResNetCNN(nn.Module):
    """
    ResNet-style CNN built from ResidualBlocks, at a comparable
    parameter/depth budget to PlainCNN and BatchNormCNN for fair
    comparison.
    """
    def __init__(self, num_classes: int = 10, depth: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        channels = _build_resnet_channel_list(depth)
        blocks = []
        in_ch = 32
        for i, out_ch in enumerate(channels):
            stride = 2 if i > 0 and i % 2 == 0 else 1
            blocks.append(ResidualBlock(in_ch, out_ch, stride=stride))
            in_ch = out_ch

        self.blocks = nn.Sequential(*blocks)
        self.pool   = nn.AdaptiveAvgPool2d((4, 4))
        self.fc     = nn.Linear(in_ch * 4 * 4, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters -- useful for fair architecture comparisons."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
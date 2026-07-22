"""
MobileNetV1 Model Architecture Definition.
Includes Depthwise Separable Convolution blocks and the full MobileNetV1 classifier.
"""

import torch
import torch.nn as nn

__all__ = ["DepthwiseConvBlock", "Mobilenet"]


class DepthwiseConvBlock(nn.Module):
    """
    Depthwise Separable Convolution Block.
    Consists of a 3x3 Depthwise Convolution (grouped) followed by BatchNorm + ReLU,
    and a 1x1 Pointwise Convolution followed by BatchNorm + ReLU.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()

        # Depthwise convolution layer (spatial filtering)
        self.depthwise_layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Pointwise convolution layer (channel combination)
        self.pointwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_layer(x)
        x = self.pointwise(x)
        return x


class Mobilenet(nn.Module):
    """
    MobileNetV1 Neural Network Architecture.
    
    Args:
        num_classes (int): Number of output classification categories. Defaults to 23.
    """
    def __init__(self, num_classes: int = 23):
        super().__init__()

        # Stem layer: Initial standard 3x3 convolution with stride 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Sequence of Depthwise Separable Convolutional Blocks
        layers = [
            DepthwiseConvBlock(32, 64, stride=1),
            DepthwiseConvBlock(64, 128, stride=2),
            DepthwiseConvBlock(128, 128, stride=1),
            DepthwiseConvBlock(128, 256, stride=2),
            DepthwiseConvBlock(256, 256, stride=1),
            DepthwiseConvBlock(256, 512, stride=2),
        ]

        # 5 identical depthwise blocks (512 -> 512, stride 1)
        for _ in range(5):
            layers.append(DepthwiseConvBlock(512, 512, stride=1))

        layers.extend([
            DepthwiseConvBlock(512, 1024, stride=2),
            DepthwiseConvBlock(1024, 1024, stride=1)
        ])
        self.depthwise_blocks = nn.Sequential(*layers)

        # Global average pooling and final linear classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.depthwise_blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


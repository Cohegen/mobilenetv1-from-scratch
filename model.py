import torch
import torch.nn as nn

class DepthwiseConvBlock(nn.Module):
  def __init__(self,in_channels,out_channels,stride):
    super().__init__()

    ##depthwise convolution layer
    self.depthwise_layer = nn.Sequential(
        nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=stride,padding=1,groups=in_channels,bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True)
    )

    ##pointwise layer
    self.pointwise = nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

  def forward(self,x:torch.Tensor):
    x = self.depthwise_layer(x)
    x= self.pointwise(x)
    return x

class Mobilenet(nn.Module):
  def __init__(self,num_classes=23):
    super().__init__()

    #stem layer
    self.conv1 = nn.Sequential(
        nn.Conv2d(3,32,kernel_size=3,stride=2,padding=1,bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True)
    )

    #depthwise blocks
    layers = [
        DepthwiseConvBlock(32,64,1),
        DepthwiseConvBlock(64,128,2),
        DepthwiseConvBlock(128,128,1),
        DepthwiseConvBlock(128,256,2),
        DepthwiseConvBlock(256,256,1),
        DepthwiseConvBlock(256,512,2),
    ]

    ##adding five identical depthwise blocks
    for _ in range(5):
      layers.append(DepthwiseConvBlock(512,512,1))

    layers.extend([
          DepthwiseConvBlock(512,1024,2),
          DepthwiseConvBlock(1024,1024,1)
    ])
    self.depthwise_blocks= nn.Sequential(*layers)

    self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    self.fc = nn.Linear(1024,num_classes)

  def forward(self,x:torch.Tensor)->torch.Tensor:
    x = self.conv1(x)
    x = self.depthwise_blocks(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    return x

# MobileNetV1 PyTorch Implementation

A modular, robust, and clean PyTorch implementation of **MobileNetV1** featuring depthwise separable convolutions, automated data preparation pipelines with split-level data augmentations, and an extensible training framework with early stopping and learning rate scheduling support.

---

## 💡 Background Information

### Overview & Motivation
**MobileNetV1** was introduced by Google researchers (*Howard et al., 2017*) in the landmark paper:
> **"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"** ([arXiv:1704.04861](https://arxiv.org/abs/1704.04861))

Prior to MobileNetV1, state-of-the-art deep convolutional networks (such as VGG-16, ResNet-50, and Inception) focused primarily on maximizing accuracy. However, these models were computationally expensive, requiring billions of Floating Point Operations (FLOPs) and tens of millions of parameters. This made them impractical for resource-constrained edge environments such as:
- **Mobile & Embedded Devices** (smartphones, IoT devices)
- **Real-time Robotics & Autonomous Drones**
- **Low-latency On-device Computer Vision** (augmented reality, smart cameras)

MobileNetV1 addresses this bottleneck by proposing a factorized convolution structure—**Depthwise Separable Convolutions**—that reduces computational complexity and model size by **8x to 9x** compared to standard convolutions with only a minimal (~1%) trade-off in classification accuracy.

---

## 🏗 MobileNetV1 Architecture Deep Dive

### 1. Core Paradigm: Depthwise Separable Convolutions
Standard convolutions apply spatial filtering and channel combination simultaneously across all input channels in a single step. MobileNetV1 factorizes standard convolution into two separate stages:

1. **Depthwise Convolution (Spatial Filtering)**:
   Applies a single $3 \times 3$ convolutional filter per input channel independently (grouped convolution where `groups = in_channels`).
2. **Pointwise Convolution (Channel Combination)**:
   Applies a $1 \times 1$ convolution across all channels to project and combine features into a new channel space.

```mermaid
graph TD
    subgraph Standard Convolution Block
        A1[Input Feature Map] --> B1["Standard 3x3 Conv"]
        B1 --> C1[Batch Normalization]
        C1 --> D1[ReLU]
    end

    subgraph Depthwise Separable Conv Block
        A2[Input Feature Map] --> B2["3x3 Depthwise Conv (groups=in_channels)"]
        B2 --> C2[Batch Normalization]
        C2 --> D2[ReLU]
        D2 --> E2["1x1 Pointwise Conv"]
        E2 --> F2[Batch Normalization]
        F2 --> G2[ReLU]
    end
```

---

### 2. Mathematical Cost Comparison

Let:
- $D_F \times D_F$ be the spatial height and width of the input feature map.
- $D_K \times D_K$ be the spatial size of the convolution kernel (typically $3 \times 3$).
- $M$ be the number of input channels.
- $N$ be the number of output channels.

#### Standard Convolution Cost:
$$\text{Cost}_{\text{Std}} = D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F$$

#### Depthwise Separable Convolution Cost:
$$\text{Cost}_{\text{Depthwise}} = \underbrace{D_K \cdot D_K \cdot M \cdot D_F \cdot D_F}_{\text{Depthwise step}} + \underbrace{M \cdot N \cdot D_F \cdot D_F}_{\text{Pointwise step}}$$

#### Computational Reduction Ratio:
$$\text{Ratio} = \frac{D_K \cdot D_K \cdot M \cdot D_F \cdot D_F + M \cdot N \cdot D_F \cdot D_F}{D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F} = \frac{1}{N} + \frac{1}{D_K^2}$$

For a standard $3 \times 3$ kernel ($D_K = 3$):
$$\text{Ratio} = \frac{1}{N} + \frac{1}{9} \approx \frac{1}{9} \quad (\sim 8\text{x to } 9\text{x computation reduction!})$$

---

### 3. Layer-by-Layer Architecture Specification

MobileNetV1 consists of an initial standard convolution layer (stem layer) followed by 13 Depthwise Separable Convolution blocks, global average pooling, and a fully connected classification layer:

| Layer / Block Index | Type / Operation | Filter Shape / Parameters | Stride | Input Resolution | Output Channels ($N$) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **0 (Stem)** | Standard Conv 3x3 | $3 \times 3 \times 3 \times 32$ | 2 | $224 \times 224 \times 3$ | 32 |
| **1** | Depthwise Separable | 3x3 DW, 1x1 PW | 1 | $112 \times 112 \times 32$ | 64 |
| **2** | Depthwise Separable | 3x3 DW, 1x1 PW | 2 | $112 \times 112 \times 64$ | 128 |
| **3** | Depthwise Separable | 3x3 DW, 1x1 PW | 1 | $56 \times 56 \times 128$ | 128 |
| **4** | Depthwise Separable | 3x3 DW, 1x1 PW | 2 | $56 \times 56 \times 128$ | 256 |
| **5** | Depthwise Separable | 3x3 DW, 1x1 PW | 1 | $28 \times 28 \times 256$ | 256 |
| **6** | Depthwise Separable | 3x3 DW, 1x1 PW | 2 | $28 \times 28 \times 256$ | 512 |
| **7 – 11** *(5x blocks)* | Depthwise Separable | 3x3 DW, 1x1 PW | 1 | $14 \times 14 \times 512$ | 512 |
| **12** | Depthwise Separable | 3x3 DW, 1x1 PW | 2 | $14 \times 14 \times 512$ | 1024 |
| **13** | Depthwise Separable | 3x3 DW, 1x1 PW | 1 | $7 \times 7 \times 1024$ | 1024 |
| **14** | AvgPool 7x7 | AdaptiveAvgPool2d | - | $7 \times 7 \times 1024$ | $1 \times 1 \times 1024$ |
| **15** | FC / Classifier | Linear | - | $1024$ | `num_classes` |

> [!NOTE]
> Roughly **95%** of MobileNetV1's computational time and **75%** of its parameters are concentrated in the $1 \times 1$ pointwise convolutions. This hardware-friendly design translates directly into high-throughput execution via standard GEMM (General Matrix Multiply) implementations.

---

### 4. Hyperparameters: Width & Resolution Multipliers

MobileNetV1 introduces two global hyperparameters to scale model size according to resource constraints:

1. **Width Multiplier ($\alpha \in (0, 1]$)**:
   - Uniformly thins the number of channels at each layer ($M \to \alpha M$, $N \to \alpha N$).
   - Reduces computational cost and parameter count by approximately $\alpha^2$.
2. **Resolution Multiplier ($\rho \in (0, 1]$)**:
   - Scales the input image resolution (e.g. $224 \to 192, 160, 128$).
   - Reduces FLOPs by $\rho^2$ while keeping parameter count identical.

---

## 📁 Repository Structure

```
MobilenetV1/
├── __init__.py          # Package initializer exposing core classes and functions
├── model.py             # MobileNetV1 architecture and DepthwiseConvBlock definitions
├── data_prep.py         # Data preprocessing, data augmentation & DataLoader pipeline
├── train_utils.py       # Training, validation, testing loops & early stopping engine
├── MobilenetV1.ipynb    # Interactive Jupyter notebook execution
└── readme.md            # Comprehensive project documentation & API reference
```

---

## 🧩 Module Overview & API Reference

### 1. `model.py` — Network Architecture

Defines the core neural network layers based on the MobileNetV1 paper (*Howard et al., 2017*).

#### Key Components:
- **`DepthwiseConvBlock(in_channels, out_channels, stride)`**:
  - **Depthwise Layer**: $3 \times 3$ depthwise convolution (`groups=in_channels`) + BatchNorm + ReLU.
  - **Pointwise Layer**: $1 \times 1$ pointwise convolution (`kernel_size=1`) + BatchNorm + ReLU.
- **`Mobilenet(num_classes=23)`**:
  - **Stem Layer**: Standard $3 \times 3$ convolution with 32 filters, stride 2 + BatchNorm + ReLU.
  - **Depthwise Separable Layers**: 13 Depthwise Convolution blocks reducing spatial dimensions while expanding channel capacity (up to 1024 channels).
  - **Classifier**: Adaptive Average Pooling `(1, 1)` followed by a linear layer projecting to `num_classes`.

#### Module Exports:
`__all__ = ["DepthwiseConvBlock", "Mobilenet"]`

#### Example Usage:
```python
import torch
from model import Mobilenet

# Instantiate model for 10 target classes
model = Mobilenet(num_classes=10)

# Test dummy forward pass (batch_size=4, channels=3, height=224, width=224)
x = torch.randn(4, 3, 224, 224)
output = model(x)
print("Output tensor shape:", output.shape)  # torch.Size([4, 10])
```

---

### 2. `data_prep.py` — Data Pipeline & Augmentations

Handles image dataset loading via `torchvision.datasets.ImageFolder`, reproducible dataset splitting (70% train / 15% validation / 15% test), split-specific data augmentations, and `DataLoader` construction.

#### Key Classes & Functions:
- **`TransformedSubset(Subset)`**:
  A PyTorch `Subset` wrapper that dynamically applies split-specific transformations (e.g. random augmentations for train set, normalization only for test set).
- **`get_dataloaders(dataset_path, image_size, batch_size, num_workers, seed=42)`**:
  Cleans non-image system folders (such as `.config`), calculates dataset split sizes, instantiates subsets, and returns `(train_loader, val_loader, test_loader, num_classes)`.

#### Augmentation Pipeline:
- **Train Set**: `RandomResizedCrop`, `RandomHorizontalFlip`, `RandomRotation(15°)`, `ColorJitter`, `RandomPerspective`, `ToTensor`, `Normalize` (ImageNet mean & std).
- **Val/Test Set**: `Resize`, `ToTensor`, `Normalize` (ImageNet mean & std).

#### Module Exports:
`__all__ = ["TransformedSubset", "get_dataloaders"]`

#### Example Usage:
```python
from data_prep import get_dataloaders

train_loader, val_loader, test_loader, num_classes = get_dataloaders(
    dataset_path="./dataset",
    image_size=224,
    batch_size=32,
    num_workers=4,
    seed=42
)
```

---

### 3. `train_utils.py` — Training & Evaluation Engine

Provides batch iteration steps, full epoch loops, validation loss tracking, early stopping, and learning rate scheduler integration.

#### Key Functions:
- **`train_step(model, dataloader, loss_fn, optimizer, device)`**:
  Performs single training epoch and returns average train loss & accuracy.
- **`test_step(model, dataloader, loss_fn, device)`**:
  Runs evaluation step in `torch.inference_mode()` and returns average loss & accuracy.
- **`train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device, patience, scheduler)`**:
  Executes complete multi-epoch training loop. Automatically saves best model weights in memory when validation loss improves, and restores them if early stopping triggers.

#### Module Exports:
`__all__ = ["train_step", "test_step", "train"]`

---

## ⚡ Quickstart — Complete Training Pipeline

Below is a complete script demonstrating how to import and tie all modules together for training:

```python
import torch
import torch.nn as nn
from model import Mobilenet
from data_prep import get_dataloaders
from train_utils import train

# Select computing device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Prepare DataLoaders
train_loader, val_loader, test_loader, num_classes = get_dataloaders(
    dataset_path="path/to/dataset",
    image_size=224,
    batch_size=32,
    num_workers=2
)

# 2. Build Model
model = Mobilenet(num_classes=num_classes).to(device)

# 3. Define Optimizer & Loss & Scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

# 4. Train Model
history = train(
    model=model,
    train_dataloader=train_loader,
    test_dataloader=val_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=25,
    device=device,
    patience=7,
    scheduler=scheduler
)

print("Training finished! Results summary:", history)
```

---

## 📦 Package-Level Imports

The root `__init__.py` file allows importing all key classes and functions directly as a single package:

```python
from MobilenetV1 import Mobilenet, get_dataloaders, train, train_step, test_step
```

---

## ⚙ Requirements

- Python 3.8+
- `torch` >= 1.12.0
- `torchvision` >= 0.13.0
- `tqdm`

"""
Main training entry script for MobileNetV1.
Parses CLI arguments, sets up dataset loaders, instantiates MobileNetV1 model,
configures optimizer/scheduler, and runs the training pipeline.
"""

import argparse
import sys
import torch
import torch.nn as nn

from model import Mobilenet
from data_prep import get_dataloaders
from train_utils import train


def parse_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train MobileNetV1 on a custom dataset."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data",
        help="Path to dataset directory formatted for torchvision ImageFolder."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Input spatial dimension size (default: 224)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="DataLoader batch size (default: 32)."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of DataLoader worker subprocesses (default: 2)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Maximum training epochs (default: 25)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer (default: 0.001)."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Early stopping patience in epochs (default: 7)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computing device ('cuda' or 'cpu'). Auto-detected if not specified."
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Determine execution device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"=== Starting MobileNetV1 Training ===")
    print(f"Device: {device}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs} (Early Stopping Patience: {args.patience})")
    print("=====================================")

    # 1. Prepare DataLoaders
    try:
        train_loader, val_loader, test_loader, num_classes = get_dataloaders(
            dataset_path=args.dataset_path,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed
        )
    except Exception as e:
        print(f"Error loading dataset from {args.dataset_path}: {e}", file=sys.stderr)
        print("Please verify your --dataset_path contains valid class subdirectories.", file=sys.stderr)
        sys.exit(1)

    # 2. Build MobileNetV1 Model
    model = Mobilenet(num_classes=num_classes).to(device)
    print(f"Initialized MobileNetV1 model with {num_classes} target classes.")

    # 3. Setup Loss Function, Optimizer, and LR Scheduler
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3
    )

    # 4. Train Model
    history = train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=args.epochs,
        device=device,
        patience=args.patience,
        scheduler=scheduler
    )

    print("\nTraining completed successfully!")
    print(f"Final Train Accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history['test_acc'][-1]:.4f}")
    return history


if __name__ == "__main__":
    main()

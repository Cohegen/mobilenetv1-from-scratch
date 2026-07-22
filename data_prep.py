"""
Data Preparation Utility Module for Image Classification.
Provides dataset loading, augmentation pipeline, train/val/test splitting,
and DataLoader construction.
"""

import os
import shutil
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

__all__ = ["TransformedSubset", "get_dataloaders"]


class TransformedSubset(Subset):
    """
    Subset wrapper that applies a specific transformation pipeline
    to dataset elements upon retrieval.
    """
    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx: int):
        # Get original image and label from base dataset
        original_image, label = self.dataset[self.indices[idx]]

        if self.transform is not None:
            image = self.transform(original_image)
        else:
            image = original_image
        return image, label

    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]


def get_dataloaders(
    dataset_path: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Creates train, validation, and test DataLoaders from an ImageFolder directory.

    Args:
        dataset_path (str): Path to image directory formatted for ImageFolder.
        image_size (int): Target square dimension (width, height) to resize images.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of subprocesses for data loading.
        seed (int, optional): Random seed for reproducible splitting. Defaults to 42.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, int]:
            - train_loader: DataLoader for training set with data augmentation.
            - val_loader: DataLoader for validation set.
            - test_loader: DataLoader for test set.
            - num_classes: Number of unique dataset target categories.
    """
    # Base transform applied to the raw ImageFolder dataset (preserves PIL image)
    base_dataset_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
    ])

    # Augmentation and normalization transform for training set
    train_full_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    # Validation and testing transform (no augmentation, only normalization)
    val_test_full_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    # Clean up non-image system folders if present (e.g. .config)
    config_in_animals_path = os.path.join(dataset_path, '.config')
    if os.path.exists(config_in_animals_path) and os.path.isdir(config_in_animals_path):
        print(f"Removing non-image folder: {config_in_animals_path}")
        shutil.rmtree(config_in_animals_path)

    # Load full dataset using torchvision ImageFolder
    full_dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=base_dataset_transform
    )

    num_classes = len(full_dataset.classes)
    print(f"Number of classes in dataset: {num_classes}")

    # Calculate 70% train, 15% validation, 15% test splits
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset_indices, val_subset_indices, test_subset_indices = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Wrap subsets with TransformedSubset to apply split-specific transforms
    train_data_split = TransformedSubset(full_dataset, train_subset_indices.indices, transform=train_full_transform)
    val_data_split = TransformedSubset(full_dataset, val_subset_indices.indices, transform=val_test_full_transform)
    test_data_split = TransformedSubset(full_dataset, test_subset_indices.indices, transform=val_test_full_transform)

    print(f"Train set size: {len(train_data_split)}")
    print(f"Validation set size: {len(val_data_split)}")
    print(f"Test set size: {len(test_data_split)}")

    train_loader = DataLoader(
        train_data_split,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_data_split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_data_split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print("DataLoaders for train, validation, and test sets created successfully.")
    return train_loader, val_loader, test_loader, num_classes


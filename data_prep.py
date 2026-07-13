import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import os
import shutil

# Helper class to apply different transforms to a subset
class TransformedSubset(Subset):
    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
       
        
        original_image, label = self.dataset[self.indices[idx]]

        if self.transform is not None:
            image = self.transform(original_image)
        else:
            image = original_image 
        return image, label

    
    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]

def get_dataloaders(dataset_path: str, image_size: int, batch_size: int, num_workers: int, seed: int = 42):
    

    # Transforms for the base dataset before splitting 
    base_dataset_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
    ])

    # Full transform for training data
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

    # Full transform for validation and test data (no augmentation, then ToTensor and Normalize)
    val_test_full_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    # Ensure the .config folder is removed if it exists before loading with ImageFolder
    config_in_animals_path = os.path.join(dataset_path, '.config')
    if os.path.exists(config_in_animals_path) and os.path.isdir(config_in_animals_path):
        print(f"Removing non-image folder: {config_in_animals_path}")
        # Use shutil.rmtree for directories
        shutil.rmtree(config_in_animals_path)

    # Load the full dataset with the base_dataset_transform (to get PIL images after resize)
    full_dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=base_dataset_transform
    )

    # Get the number of classes from the dataset
    num_classes = len(full_dataset.classes)
    print(f"Number of classes in dataset: {num_classes}")

    # Calculate split lengths
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    # Perform random split on the full_dataset (which returns PIL images after resize)
    generator = torch.Generator().manual_seed(seed) # for reproducibility
    train_subset_indices, val_subset_indices, test_subset_indices = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Wrap subsets with TransformedSubset to apply specific transforms
    train_data_split = TransformedSubset(full_dataset, train_subset_indices.indices, transform=train_full_transform)
    val_data_split = TransformedSubset(full_dataset, val_subset_indices.indices, transform=val_test_full_transform)
    test_data_split = TransformedSubset(full_dataset, test_subset_indices.indices, transform=val_test_full_transform)

    print(f"Train set size: {len(train_data_split)}")
    print(f"Validation set size: {len(val_data_split)}")
    print(f"Test set size: {len(test_data_split)}")

    # Create DataLoaders for each split
    train_loader = DataLoader(
        train_data_split,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True # Speeds up data transfer to GPU
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

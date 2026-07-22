"""
Training and Evaluation Utility Module.
Contains training loop, test evaluation loop, and early stopping / scheduler tracking.
"""

import copy
from typing import Dict, List, Tuple, Union, Optional

import torch
import torch.nn as nn
from tqdm.auto import tqdm

__all__ = ["train_step", "test_step", "train"]


def train_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: Union[str, torch.device]
) -> Tuple[float, float]:
    """
    Performs a single training epoch step.

    Args:
        model (nn.Module): PyTorch neural network model.
        dataloader (DataLoader): DataLoader for the training dataset.
        loss_fn (nn.Module): Loss function criterion.
        optimizer (torch.optim.Optimizer): Optimization algorithm instance.
        device (Union[str, torch.device]): Computing device ('cpu' or 'cuda').

    Returns:
        Tuple[float, float]: Average train loss and train accuracy across all batches.
    """
    model.train()
    train_loss, train_acc = 0.0, 0.0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)

        # Compute loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Backward pass & optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: Union[str, torch.device]
) -> Tuple[float, float]:
    """
    Performs an evaluation step over a validation or test dataset.

    Args:
        model (nn.Module): PyTorch neural network model.
        dataloader (DataLoader): DataLoader for validation or test dataset.
        loss_fn (nn.Module): Loss function criterion.
        device (Union[str, torch.device]): Computing device ('cpu' or 'cuda').

    Returns:
        Tuple[float, float]: Average evaluation loss and accuracy across all batches.
    """
    model.eval()
    test_loss, test_acc = 0.0, 0.0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module = nn.CrossEntropyLoss(),
    epochs: int = 5,
    device: Union[str, torch.device] = 'cpu',
    patience: int = 7,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Dict[str, List[float]]:
    """
    Trains and evaluates a model over a number of epochs with support for early stopping and LR schedulers.

    Args:
        model (nn.Module): PyTorch neural network model to train.
        train_dataloader (DataLoader): DataLoader for training images.
        test_dataloader (DataLoader): DataLoader for validation/test images.
        optimizer (torch.optim.Optimizer): Optimizer instance (e.g. Adam, SGD).
        loss_fn (nn.Module, optional): Loss function. Defaults to CrossEntropyLoss().
        epochs (int, optional): Total number of training epochs. Defaults to 5.
        device (Union[str, torch.device], optional): Computing device. Defaults to 'cpu'.
        patience (int, optional): Epochs to wait before triggering early stopping on validation loss. Defaults to 7.
        scheduler (Optional[_LRScheduler], optional): Learning rate scheduler. Defaults to None.

    Returns:
        Dict[str, List[float]]: History dictionary containing 'train_loss', 'train_acc', 'test_loss', 'test_acc'.
    """
    results: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Step learning rate scheduler if provided
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()

        # Early stopping & model saving logic
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            epochs_no_improve = 0
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epochs.")
            if epochs_no_improve == patience:
                print(f"Early stopping triggered after {epoch+1} epochs. Restoring best model weights.")
                model.load_state_dict(best_model_wts)
                break

        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    return results


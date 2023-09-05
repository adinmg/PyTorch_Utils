"""
Contains functions for training and validating a PyTorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """ Trains a PyTorch model for a single epoch.

    Turns a target Pytorch model to training mode and then
    runs though all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
        model: A Pytorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").
    
        Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:
        (0.7236, 0.8726)    
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = .0, .0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    
    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def validate_step(model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  loss_fn: torch.nn.Module,
                  device: torch.device) -> Tuple[float, float]:
    """Validates a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a validation dataset.

    Args:
        model: A PyTorch model to be validated.
        dataloader: A DataLoader instance for the model to be validated on.
        loss_fn: A PyTorch loss function to calculate loss on the validation data.
        device: A target device to compute on (e.g., "cuda" or "cpu").

    Returns:
        A tuple of validation loss and validation accuracy metrics.
        In the form (validation_loss, validation_accuracy).
    """
    # Put model in eval mode
    model.eval()

    # Setup validation loss and validation accuracy values
    validation_loss, validation_acc = .0, .0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to the target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            validation_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(validation_pred_logits, y)
            validation_loss += loss.item()

            # Calculate and accumulate accuracy
            validation_pred_labels = validation_pred_logits.argmax(dim=1)
            validation_acc += ((validation_pred_labels == y).sum().item() / len(validation_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    validation_loss = validation_loss / len(dataloader)
    validation_acc = validation_acc / len(dataloader)
    return validation_loss, validation_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          validate_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    """Trains and validates a PyTorch model.

    Passes a target PyTorch model through train_step() and validate_step()
    functions for a number of epochs, training and validating the model
    in the same epoch loop.

    Calculates, prints, and stores evaluation metrics throughout.

    Args:
        model: A PyTorch model to be trained and validated.
        train_dataloader: A DataLoader instance for the model to be trained on.
        validate_dataloader: A DataLoader instance for the model to be validated on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g., "cuda" or "cpu").

    Returns:
        A dictionary of training and validation loss as well as training and
        validation accuracy metrics. Each metric has a value in a list for 
        each epoch.
        In the form: {train_loss: [...],
                    train_acc: [...],
                    valid_loss: [...],
                    valid_acc: [...]} 
    """
    # Create an empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "valid_loss": [],
               "valid_acc": []
               }

    # Loop through training and validation steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        
        validation_loss, validation_acc = validate_step(model=model,
                                                        dataloader=validate_dataloader,
                                                        loss_fn=loss_fn,
                                                        device=device)
        
        # Print out what's happening
        print(f"Epoch: {epoch+1} | "
              f"train_loss: {train_loss:.4f} | "
              f"train_acc: {train_acc:.4f} | "
              f"valid_loss: {validation_loss:.4f} | "
              f"valid_acc: {validation_acc:.4f}"
              )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["valid_loss"].append(validation_loss)
        results["valid_acc"].append(validation_acc)
    
    # Return the filled results at the end of the epochs
    return results

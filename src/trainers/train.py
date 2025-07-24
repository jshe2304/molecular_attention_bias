import os
from datetime import datetime

import numpy as np
from sklearn.metrics import r2_score

import torch
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F

@torch.no_grad()
def _sample_metrics(model, dataset, n_samples=4096, batch_size=64, device='cpu'):
    """
    Estimates the loss of the model on a dataset. 
    No support for distributed inference, only single GPU/CPU. 

    Args:
        model: The model to evaluate. 
        dataset: The dataset to evaluate on. 
        n_samples: The number of samples to use for evaluation. 
        batch_size: The batch size to use for evaluation. 
        device: The device to use for evaluation. 
    """

    dataloader = DataLoader(dataset, batch_size=batch_size)

    model.eval()
    batch_y_pred, batch_y_true = [], []
    for i, (*inputs, y_true) in enumerate(dataloader):
        if i * batch_size >= n_samples: break

        # Send data to device

        inputs = [input.to(device) for input in inputs]

        # Forward pass

        pred = model(*inputs) # (B, N_properties)

        # Store predictions and true values

        batch_y_pred.append(pred.cpu())
        batch_y_true.append(y_true.cpu())

    all_y_pred = torch.cat(batch_y_pred, dim=0)
    all_y_true = torch.cat(batch_y_true, dim=0)

    loss = F.mse_loss(all_y_pred, all_y_true, reduction='none').mean(dim=0)
    loss = dataset.unnormalize(loss).numpy()
    score = r2_score(all_y_true, all_y_pred, multioutput='raw_values')

    return loss, score

def _train_one_epoch(
    model, train_dataloader, device, 
    optimizer, scheduler=None, 
    **kwargs
    ):
    """
    Train the model for one epoch minimizing the MSE loss.

    Args:
        model: The model to train
        optimizer: The optimizer to use
        device: The device to use
        train_dataloader: The dataloader for the training data
    """

    model.train()
    for *inputs, y_true in train_dataloader:

        # Send data to device

        inputs = [input.to(device) for input in inputs]
        y_true = y_true.to(device)

        # Forward and backward passes

        optimizer.zero_grad()
        y_pred = model(*inputs)
        loss = F.mse_loss(y_pred, y_true)
        loss.backward()

        # Step optimizer and scheduler

        optimizer.step()
        if scheduler is not None: scheduler.step()

def train(
    model, device, 
    train_dataset, validation_dataset, 
    epochs, batch_size, 
    learning_rate, weight_decay,
    warmup_start_factor, warmup_epochs, plateau_factor, plateau_patience,
    output_dir, 
    ):
    """
    Train the model on a single device. 

    Args:
        model: The model to train
        device: The device to use
        train_dataset: The training dataset
        validation_dataset: The validation dataset
        learning_rate: The learning rate
        weight_decay: The weight decay
        warmup_start_factor: The start factor for the warmup phase
        warmup_epochs: The total number of epochs for the warmup phase
        plateau_factor: The factor for the plateau phase
        plateau_patience: The patience for the plateau phase
        epochs: The number of epochs to train for
        batch_size: The batch size to use
        checkpoint_period: How often to save the model (epochs)
        checkpoint_dir: The directory to save the checkpoints
        logger: Optional wandb logger
        **kwargs: Overflow arguments
    """

    # Dataloader

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
    )

    # Optimizer

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Schedulers

    warmup_steps = warmup_epochs * len(train_dataloader)
    warmup = lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_start_factor, total_iters=warmup_steps
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=plateau_factor, patience=plateau_patience
    )

    # Metrics

    metrics = []

    # Training

    for epoch in range(epochs):

        # Train

        _train_one_epoch(
            model, train_dataloader, device, 
            optimizer, scheduler=warmup
        )

        # Sample metrics

        train_loss, train_score = _sample_metrics(
            model, train_dataset, 
            batch_size=batch_size, device=device
        )
        validation_loss, validation_score = _sample_metrics(
            model, validation_dataset, 
            batch_size=batch_size, device=device
        )

        # Step scheduler

        scheduler.step(validation_loss)

        # Store metrics

        metrics.append(np.concatenate(
            (train_loss, validation_loss, train_score, validation_score)
        ))

    # Make output directory

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp += '_' + str(os.environ.get("SLURM_ARRAY_TASK_ID"))

    # Save logs

    with open(os.path.join(output_dir, f'log_{timestamp}.csv'), 'w') as f:

        # Create header

        for metric_type in ['train_loss', 'validation_loss', 'train_score', 'validation_score']:
            for label in train_dataset.y_labels:
                f.write(f'{label}_{metric_type},')
        f.write('\n')

        # Write metrics

        for row in metrics:
            f.write(','.join(list(row)) + ',\n')

    # Save model

    torch.save(model.state_dict(), os.path.join(output_dir, f'model_{timestamp}.pt'))

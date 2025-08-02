import os

import numpy as np

import torch
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .metrics import compute_metrics

def _train_one_epoch(
    model, device, 
    dataloader, 
    optimizer
    ):
    """
    Train the model for one epoch minimizing the MSE loss.

    Args:
        model: The model to train
        device: The device to use
        dataloader: The dataloader for the training data
        optimizer: The optimizer to use
    """

    model.train()
    for *inputs, y_true in dataloader:

        # Send data to device

        inputs = [input.to(device) for input in inputs]
        y_true = y_true.to(device)

        # Forward and backward passes

        optimizer.zero_grad()
        y_pred = model(*inputs)
        loss = F.mse_loss(y_pred, y_true)
        loss.backward()
        optimizer.step()

def train(
    model, device, 
    train_dataset, val_dataset, 
    epochs, batch_size, 
    learning_rate, weight_decay,
    output_dir, 
    ):
    """
    Train the model on a single device. 

    Args:
        model: The model to train
        device: The device to use
        train_dataset: The training dataset
        val_dataset: The validation dataset
        epochs: The number of epochs to train for
        batch_size: The batch size to use
        learning_rate: The learning rate
        weight_decay: The weight decay
        output_dir: Directory to write logs and checkpoints
    """

    # Make metrics labels and write log header

    metric_labels = [
        f'{split}_{label}_{metric}'
        for metric in ('mae', 'r2')
        for label in train_dataset.y_labels
        for split in ('train', 'val')
    ]

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'log.csv'), 'w') as f:
        f.write(','.join(metric_labels) + '\n')

    # Dataloader

    dataloader = DataLoader(
        train_dataset, batch_size=batch_size, 
        num_workers=4, persistent_workers=True, pin_memory=True, 
        collate_fn=train_dataset.collate
    )

    # Optimizer and schedulers

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-8
    )

    # Training

    best_state_dict, best_loss = None, float('inf')
    for epoch in range(epochs):

        # Train

        _train_one_epoch(
            model, device, 
            dataloader, 
            optimizer, 
        )

        # Sample metrics

        train_loss, train_score = compute_metrics(
            model, train_dataset, 
            batch_size=batch_size, device=device
        )

        val_loss, val_score = compute_metrics(
            model, val_dataset, 
            batch_size=batch_size, device=device
        )
        mean_val_loss = val_loss.mean()

        # Step scheduler

        scheduler.step()

        # Log metrics

        metrics = np.concatenate((
            train_loss, val_loss, train_score, val_score
        )).tolist()

        with open(os.path.join(output_dir, 'log.csv'), 'a') as f:
            f.write(','.join(str(n) for n in metrics) + '\n')

        # Update best model

        if mean_val_loss < best_loss:
            best_state_dict = model.state_dict()
            best_loss = mean_val_loss
    
    # Save best model

    torch.save(best_state_dict, os.path.join(output_dir, f'model.pt'))

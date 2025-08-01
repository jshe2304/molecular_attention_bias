import os
import time
from datetime import datetime

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
    optimizer, scheduler=None
    ):
    """
    Train the model for one epoch minimizing the MSE loss.

    Args:
        model: The model to train
        device: The device to use
        dataloader: The dataloader for the training data
        optimizer: The optimizer to use
        scheduler: Optional batch-wise learning rate scheduler
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
        if scheduler is not None: scheduler.step()

def train(
    model, device, 
    train_dataset, validation_dataset, 
    epochs, batch_size, 
    learning_rate, weight_decay,
    warmup_start_factor, warmup_epochs, plateau_factor, plateau_patience,
    output_dir, 
    logger=None, 
    ):
    """
    Train the model on a single device. 

    Args:
        model: The model to train
        device: The device to use
        train_dataset: The training dataset
        validation_dataset: The validation dataset
        epochs: The number of epochs to train for
        batch_size: The batch size to use
        learning_rate: The learning rate
        weight_decay: The weight decay
        warmup_start_factor: The start factor for the warmup phase
        warmup_epochs: The total number of epochs for the warmup phase
        plateau_factor: The factor for the plateau phase
        plateau_patience: The patience for the plateau phase
        output_dir: Directory to write logs and checkpoints
        logger: Optional wandb logger
        **kwargs: Overflow arguments
    """

    # Make metrics labels

    metric_labels = [
        f'{split}/{label}_{metric_type}'
        for metric in ('mse', 'r2')
        for label in train_dataset.y_labels
        for split in ('train', 'val')
    ]

    # Dataloader

    dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=4, 
        collate_fn=train_dataset.collate
    )

    # Optimizer and schedulers

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    warmup = lr_scheduler.LinearLR(
        optimizer, 
        start_factor=warmup_start_factor, total_iters=warmup_epochs * len(dataloader)
    )

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=plateau_factor, patience=plateau_patience
    )

    # Training

    # all_metrics = []
    for epoch in range(epochs):

        # Train

        _train_one_epoch(
            model, dataloader, device, 
            optimizer, scheduler=warmup
        )

        # Sample metrics

        train_loss, train_score = compute_metrics(
            model, train_dataset, 
            batch_size=batch_size, device=device
        )
        validation_loss, validation_score = compute_metrics(
            model, validation_dataset, 
            batch_size=batch_size, device=device
        )

        # Step scheduler

        scheduler.step(validation_loss.mean())

        # Log metrics

        metrics = np.concatenate((
            train_loss, validation_loss, train_score, validation_score
        )).tolist()

        if logger is not None:
            logger.log(dict(zip(metric_labels, metrics)))

        # all_metrics.append(this_metrics)
    
    # # Save logs

    # os.makedirs(output_dir, exist_ok=True)
    # with open(os.path.join(output_dir, f'log_{timestamp}.csv'), 'w') as f:
    #     f.write(','.join(metric_labels) + '\n')
    #     for metrics in all_metrics:
    #         f.write(','.join(str(n) for n in metrics) + '\n')

    # # Save model
    
    # model_file = os.path.join(output_dir, f'model_{timestamp}.pt')
    # torch.save(model.state_dict(), model_file)

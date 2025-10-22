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
    optimizer, 
    scheduler,
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
        scheduler.step()

def train(
    model, device, 
    train_dataset, val_dataset, test_dataset, 
    epochs, batch_size, 
    lr, weight_decay, 
    warmup_epochs, warmup_start_factor,
    output_dir, 
    ):
    """
    Train the model on a single device. 

    Args:
        model: The model to train
        device: The device to use
        train_dataset: The training dataset
        val_dataset: The validation dataset
        test_dataset: The test dataset
        epochs: The number of epochs to train for
        batch_size: The batch size to use
        lr: The learning rate
        weight_decay: The weight decay
        output_dir: Directory to write logs and checkpoints
    """

    # Make metrics labels and write log header

    metric_labels = [
        f'{split}_{label}_mae'
        for split in ('train', 'val')
        for label in train_dataset.y_labels
    ]
    metric_labels.append('lr')

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'log.csv'), 'w') as f:
        f.write(','.join(metric_labels) + '\n')

    # Dataloader

    dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, persistent_workers=True, pin_memory=True, 
        collate_fn=train_dataset.collate
    )

    # Optimizer and schedulers

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    warmup_steps = warmup_epochs * len(dataloader)
    warmup = lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_start_factor, total_iters=warmup_steps
    )

    # scheduler = lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=(epochs - warmup_epochs), eta_min=lr * 0.01
    # )
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.6, patience=6,
    )

    # Train

    best_state_dict, best_loss = None, float('inf')
    for epoch in range(epochs):

        # Pass through training data
        _train_one_epoch(
            model, device, 
            dataloader, 
            optimizer, 
            warmup,
        )

        # Compute losses
        train_loss = compute_metrics(model, train_dataset, per_atom=True, device=device)
        val_loss = compute_metrics(model, val_dataset, per_atom=True, device=device)
        mean_val_loss = val_loss.mean()

        # Step scheduler
        scheduler.step(mean_val_loss)

        # Log losses
        losses = np.concatenate((train_loss, val_loss)).tolist()
        losses.append(optimizer.param_groups[0]['lr'])
        with open(os.path.join(output_dir, 'log.csv'), 'a') as f:
            f.write(','.join(str(n) for n in losses) + '\n')

        # Update best model
        if mean_val_loss < best_loss:
            best_state_dict, best_loss = model.state_dict(), mean_val_loss

    # Evaluate on test set
    model.load_state_dict(best_state_dict)
    test_loss = compute_metrics(
        model, test_dataset, n_samples=len(test_dataset), per_atom=False, device=device,
    ).tolist()
    test_loss += compute_metrics(
        model, test_dataset, n_samples=len(test_dataset), per_atom=True, device=device,
    ).tolist()

    # Save test losses
    loss_labels = [f'test_{label}_mae' for label in test_dataset.y_labels]
    loss_labels += [f'test_{label}_mae_per_atom' for label in test_dataset.y_labels]
    with open(os.path.join(output_dir, 'test_losses.csv'), 'w') as f:
        f.write(','.join(loss_labels) + '\n')
        f.write(','.join(str(n) for n in test_loss) + '\n')

    # Save model
    torch.save(best_state_dict, os.path.join(output_dir, f'model.pt'))

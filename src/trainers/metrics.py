import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

@torch.no_grad()
def compute_metrics(model, dataset, n_samples=1024, batch_size=256, per_atom=False, device='cpu'):
    """
    Estimates the loss of the model on a dataset. 
    No support for distributed inference, only single GPU/CPU. 

    Args:
        model: The model to evaluate. 
        dataset: The dataset to evaluate on. 
        n_samples: The number of samples to use for evaluation. 
        batch_size: The batch size to use for evaluation. 
        per_atom: Whether to divide by the number of atoms in the batch. 
        device: The device to use for evaluation. 
    """

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, persistent_workers=True, pin_memory=(str(device) != 'cpu'), 
        collate_fn=dataset.collate
    )

    maes = []

    model.eval()
    for i, (*inputs, y_true) in enumerate(dataloader):
        if i * batch_size >= n_samples: break

        # Send data to device
        inputs = [input.to(device) for input in inputs]

        # Forward pass and unnormalize
        y_pred = model(*inputs).cpu()
        y_pred = dataset.unnormalize(y_pred)
        y_true = dataset.unnormalize(y_true)
        mae = F.l1_loss(y_pred, y_true, reduction='none')

        # Compute per-atom MAE if requested
        if per_atom:
            _, padding, *_ = inputs
            n_atoms = (~padding).sum(dim=1, keepdim=True).cpu() # (B, 1)
            mae = mae / n_atoms

        maes.append(mae)

    # Concatenate batches and compute mean
    return torch.cat(maes, dim=0).mean(dim=0)

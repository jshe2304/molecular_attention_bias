import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import r2_score

@torch.no_grad()
def compute_metrics(model, dataset, n_samples=2048, batch_size=64, device='cpu'):
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

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, persistent_workers=True, pin_memory=(str(device) != 'cpu'), 
        collate_fn=dataset.collate
    )

    model.eval()
    batch_y_pred, batch_y_true = [], []
    for i, (*inputs, y_true) in enumerate(dataloader):
        if i * batch_size >= n_samples: break

        # Send data to device

        inputs = [input.to(device) for input in inputs]

        # Forward pass

        pred = model(*inputs)

        # Store predictions and true values

        batch_y_pred.append(pred.cpu())
        batch_y_true.append(y_true.cpu())

    all_y_pred = dataset.unnormalize(torch.cat(batch_y_pred, dim=0))
    all_y_true = dataset.unnormalize(torch.cat(batch_y_true, dim=0))

    mae = F.l1_loss(all_y_pred, all_y_true, reduction='none').mean(dim=0)
    r2 = r2_score(all_y_true, all_y_pred, multioutput='raw_values')

    return mae.squeeze().numpy(), r2

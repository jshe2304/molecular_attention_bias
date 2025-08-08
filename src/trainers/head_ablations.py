import torch
import numpy as np

from .metrics import compute_metrics

@torch.no_grad()
def ablate_heads(model, device, dataset, n_samples=4096):
    """
    Train the model on a single device. 

    Args:
        model: The model to train
        device: The device to use
        dataset: The dataset to use
    """

    # Base model loss and score

    loss, _ = compute_metrics(model, dataset, batch_size=128, device=device, n_samples=n_samples)

    # Ablate heads

    all_exponents = []
    all_losses = []
    for l in range(len(model.transformer_blocks)):
        radial_function = model.transformer_blocks[l].radial_function
        attn = model.transformer_blocks[l].attn

        exponents = []
        losses = []
        for h in range(attn.H):

            # Ablate head

            qkv = attn.QKV.weight
            ablated_qkv = qkv.clone().T.reshape(attn.E, attn.H, 3 * attn.A)
            ablated_qkv[:, h] = 0
            ablated_qkv = ablated_qkv.reshape(attn.E, 3 * attn.H * attn.A)
            attn.QKV.weight.copy_(ablated_qkv.T)

            # Compute loss and score

            ablated_loss, _ = compute_metrics(
                model, dataset, batch_size=128, device=device, 
                n_samples=n_samples
            )

            # Restore head

            attn.QKV.weight = qkv

            # Add to logs

            exponents.append(radial_function.p[h].item())
            losses.append((ablated_loss - loss)/loss)

        all_exponents.append(exponents)
        all_losses.append(losses)

    return all_exponents, all_losses
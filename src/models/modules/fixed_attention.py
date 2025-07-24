import torch
import torch.nn as nn

class FixedAttention(nn.Module):
    def __init__(self, E, H):
        super().__init__()

        assert E % H == 0

        self.E, self.H, self.A = E, H, E // H

        self.V = nn.Linear(E, E, bias=False)
        self.out_map = nn.Linear(E, E, bias=False)

    def forward(self, embeddings, weights, causal_mask):

        B, L, E = embeddings.size() # Batch, no. Tokens, Embed dim.

        # Compute V matrix

        v = self.V(embeddings)
        v = v.reshape(B, L, self.H, self.A)
        v = v.permute(0, 2, 1, 3)

        # Compute attention pattern

        if causal_mask is not None: 
            weights.masked_fill_(causal_mask, torch.finfo(weights.dtype).min)
        weights = torch.softmax(weights, dim=-1)

        # Compute values

        values = weights @ v
        values = values.permute(0, 2, 1, 3) # (B, L, H, A)
        values = values.reshape(B, L, E) # E = H * A
        
        return self.out_map(values)

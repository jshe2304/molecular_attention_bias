import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, E, H, **kwargs):
        super().__init__()

        assert E % H == 0

        self.E, self.H = E, H
        self.scale = (E // H) ** -0.5

        self.QKV = nn.Linear(E, E * 3, bias=False)
        self.out_map = nn.Linear(E, E, bias=False)

    def forward(self, embeddings, attn_mask=None, attn_bias=None):

        B, L, E = embeddings.size() # Batch, no. Tokens, Embed dim.
        A = E // self.H # Attention dim.

        # Compute Q, K, V matrices

        qkv = self.QKV(embeddings)
        qkv = qkv.reshape(B, L, self.H, 3 * A)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        # Compute attention pattern

        attn = q @ k.transpose(-2, -1) * self.scale
        if attn_bias is not None: attn += attn_bias
        if attn_mask is not None: attn.masked_fill_(attn_mask, torch.finfo(attn.dtype).min)
        attn = torch.softmax(attn, dim=-1)

        # Compute values

        values = attn @ v
        values = values.permute(0, 2, 1, 3) # (B, L, H, A)
        values = values.reshape(B, L, E) # E = H * A
        
        return self.out_map(values)

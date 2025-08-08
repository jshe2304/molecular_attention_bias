import torch
import torch.nn as nn

class SDPAttention(nn.Module):
    def __init__(self, E, H, **kwargs):
        super().__init__()

        assert E % H == 0

        self.E, self.H, self.A = E, H, E // H
        self.scale = (E // H) ** -0.5

        self.QKV = nn.Linear(E, E * 3, bias=False)
        self.out_map = nn.Linear(E, E, bias=False)

        self.hook = {}

    def forward(self, embeddings, attn_mask=None, attn_bias=None, hook=False):

        B, L, E = embeddings.size() # Batch, no. Tokens, Embed dim.

        # Compute Q, K, V matrices

        qkv = self.QKV(embeddings)
        qkv = qkv.reshape(B, L, self.H, 3 * self.A)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        # Compute attention pattern

        attn_logits = q @ k.transpose(-2, -1) * self.scale
        #if hook: self.hook['qkt'] = attn_logits.detach().clone().cpu()
        if attn_bias is not None: attn_logits += attn_bias
        #if hook: self.hook['attn_bias'] =attn_bias.detach().clone().cpu()
        if attn_mask is not None: attn_logits.masked_fill_(attn_mask, torch.finfo(attn_logits.dtype).min)
        #if hook: self.hook['attn_logits'] = attn_logits.detach().cpu()
        attn = torch.softmax(attn_logits, dim=-1)
        #if hook: self.hook['attn'] = attn.detach().cpu()
        
        # Compute values

        values = attn @ v
        values = values.permute(0, 2, 1, 3) # (B, L, H, A)
        values = values.reshape(B, L, E) # E = H * A
        values = self.out_map(values)
        
        return values

class GraphSDPAttention(nn.Module):
    def __init__(self, E, H, **kwargs):
        super().__init__()

        assert E % H == 0

        self.E, self.H = E, H
        self.scale = (E // H) ** -0.5

        self.QKV = nn.Linear(E, E * 3, bias=False)
        self.out_map = nn.Linear(E, E, bias=False)

    def forward(self, embeddings, attn_mask=None):

        B, L, E = embeddings.size() # Batch, no. Tokens, Embed dim.
        A = E // self.H # Attention dim.

        # Compute Q, K, V matrices

        qkv = self.QKV(embeddings)
        qkv = qkv.reshape(B, L, self.H, 3 * A)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        # Compute attention

        attn = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask
        )
        attn = attn.permute(0, 2, 1, 3) # (B, L, H, A)
        attn = attn.reshape(B, L, E) # E = H * A

        return self.out_map(attn)

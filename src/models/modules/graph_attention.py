import torch
import torch.nn as nn

class MaskedSDPA(nn.Module):
    def __init__(self, E, H, **kwargs):
        super().__init__()

        assert E % H == 0

        self.E, self.H = E, H
        self.scale = (E // H) ** -0.5

        self.QKV = nn.Linear(E, E * 3, bias=False)
        self.out_map = nn.Linear(E, E, bias=False)

    def forward(self, embeddings, adj=None):

        B, L, E = embeddings.size() # Batch, no. Tokens, Embed dim.
        A = E // self.H # Attention dim.

        # Compute Q, K, V matrices

        qkv = self.QKV(embeddings)
        qkv = qkv.reshape(B, L, self.H, 3 * A)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        # Compute attention

        attn = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=~adj
        )
        attn = attn.permute(0, 2, 1, 3) # (B, L, H, A)
        attn = attn.reshape(B, L, E) # E = H * A

        return self.out_map(attn)

class GATv2(nn.Module):
    def __init__(self, E, H, **kwargs):
        super().__init__()
        assert E % H == 0

        self.E, self.H, self.A = E, H, E // H

        self.QK = nn.Linear(E, E * 2, bias=False)
        self.attn_map = nn.Linear(self.A, 1, bias=False)
        self.lrelu = nn.LeakyReLU(kwargs.get('leakage', 0.1))
        self.out_map = nn.Linear(E, E, bias=False)

    def forward(self, embeddings, adj, **kwargs):

        B, L, E = embeddings.size() # Batch, Tokens, Embed dim.

        qk = self.QK(embeddings)
        qk = qk.reshape(B, L, self.H, 2 * self.A)
        qk = qk.permute(0, 2, 1, 3)
        q, k = qk.chunk(2, dim=-1)

        attn = q.unsqueeze(2) + k.unsqueeze(3)
        attn = self.lrelu(attn)
        attn = self.attn_map(attn).squeeze(-1)
        attn.masked_fill_(~adj, torch.finfo(attn.dtype).min)
        attn = torch.softmax(attn, dim=-1)
        
        values = attn @ k
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(B, L, E)

        return self.out_map(values)
    
class GINConv(nn.Module):
    def __init__(self, E, H, **kwargs):
        super().__init__()
        assert E % H == 0
        self.mlp = nn.Sequential(
            nn.Linear(E, E * H // 2), 
            nn.ReLU(), 
            nn.Linear(E * H // 2, E)
        )

    def forward(self, embeddings, adj, **kwargs):
        adj = adj.squeeze(1).float()
        return self.mlp(adj @ embeddings + embeddings)

def get_graph_attention_operator(graph_attention_operator_type, *args, **kwargs):
    if graph_attention_operator_type == 'MaskedSDPA':
        return MaskedSDPA(*args, **kwargs)
    elif graph_attention_operator_type == 'GATv2':
        return GATv2(*args, **kwargs)
    elif graph_attention_operator_type == 'GINConv':
        return GINConv(*args, **kwargs)
    else:
        raise ValueError(f"Invalid operator type: {graph_attention_operator_type}")
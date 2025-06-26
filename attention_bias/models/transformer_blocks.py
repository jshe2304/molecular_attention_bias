import torch
import torch.nn as nn

from .graph_operators import SDPAOperator

class TransformerBlock(nn.Module):
    def __init__(self, E, H, BiasMap, dropout=0.1):
        super().__init__()

        self.bias_map = BiasMap(H)
        
        self.operator = SDPAOperator(E, H)
        self.norm_1 = nn.LayerNorm(E)
        self.mlp = nn.Sequential(
            nn.Linear(E, E * 4), 
            nn.ReLU(), 
            nn.Linear(E * 4, E)
        )
        self.norm_2 = nn.LayerNorm(E)
        self.dropout = nn.Dropout(dropout)

    def forward(self, e, d, padding_mask, attn_mask):

        B, L, E = e.shape

        # Scale bias

        attn_bias = self.bias_map(
            d.reshape(B, L, L)
        )

        # Attention block

        e0 = self.norm_1(e)
        e1 = self.operator(e0, attn_mask=attn_mask, attn_bias=attn_bias)
        e1 = self.dropout(e1)
        e2 = e1 + e0

        # MLP block

        e2 = self.norm_2(e2)
        e3 = self.mlp(e2)
        e3.masked_fill_(padding_mask, 0)
        e3 = self.dropout(e3)
        e4 = e3 + e2

        return e4

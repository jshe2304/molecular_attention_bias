import torch
import torch.nn as nn

from .graph_operators import SDPAOperator

class TransformerBlock(nn.Module):
    def __init__(self, E, H, attn_bias, dropout=0.1):
        super().__init__()

        self.attn_bias = attn_bais
        
        self.operator = SDPAOperator(E, H)
        self.norm_1 = nn.LayerNorm(E)
        self.mlp = nn.Sequential(
            nn.Linear(E, E * 4), 
            nn.ReLU(), 
            nn.Linear(E * 4, E)
        )
        self.norm_2 = nn.LayerNorm(E)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, d, padding_mask, attn_mask):

        B, L, E = x.shape

        # Scale bias

        attn_bias = self.attn_bias(
            d.reshape(B, L, L, 1)
        ).permute(0, 3, 1, 2)

        # Attention block

        x0 = self.norm_1(x)
        x1 = self.operator(x0, attn_mask=attn_mask, attn_bias=attn_bias)
        x1 = self.dropout(x1)
        x2 = x1 + x0

        # MLP block

        x2 = self.norm_2(x2)
        x3 = self.mlp(x2)
        x3.masked_fill_(padding_mask, 0)
        x3 = self.dropout(x3)
        x4 = x3 + x2

        return x4
import torch
import torch.nn as nn

from .graph_operators import WeightedMessagePassing

class TransformerBlock(nn.Module):
    def __init__(self, E, H, WeightFunction, dropout=0.1, **kwargs):
        super().__init__()

        self.weight_function = WeightFunction(H, **kwargs)
        
        self.operator = WeightedMessagePassing(E, H)
        self.norm_1 = nn.LayerNorm(E)
        self.mlp = nn.Sequential(
            nn.Linear(E, E * 4), 
            nn.ReLU(), 
            nn.Linear(E * 4, E)
        )
        self.norm_2 = nn.LayerNorm(E)
        self.dropout = nn.Dropout(dropout)

    def forward(self, e, d, causal_mask, padding_mask):

        B, L, E = e.shape

        # Process interatomic distances into weights

        weights = self.weight_function(
            d.reshape(B, L, L)
        )

        # Attention block

        e0 = self.norm_1(e)
        e1 = self.operator(e0, weights=weights, causal_mask=causal_mask)
        e1 = self.dropout(e1)
        e2 = e1 + e0

        # MLP block

        e2 = self.norm_2(e2)
        e3 = self.mlp(e2)
        e3.masked_fill_(padding_mask, 0)
        e3 = self.dropout(e3)
        e4 = e3 + e2

        return e4

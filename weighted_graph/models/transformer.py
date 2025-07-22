import torch
import torch.nn as nn

from .transformer_blocks import TransformerBlock

class Transformer(nn.Module):

    def __init__(self, n_tokens, out_features, E, H, D, WeightFunction, dropout=0.1, **kwargs):
        super().__init__()

        self.E, self.H, self.D = E, H, D

        # Embedding layer
        self.embed = nn.Embedding(n_tokens, E, padding_idx=0)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                E, H, WeightFunction, dropout=dropout, **kwargs
            ) for _ in range(D)
        ])
        
        # Out map
        self.norm = nn.LayerNorm(E)
        self.out_map = nn.Linear(E, out_features)

    def forward(self, tokens, r, padding):

        assert tokens.shape[0] == r.shape[0] == padding.shape[0]
        B, L, *_ = tokens.shape

        # Construct interatomic distances

        d = torch.norm(
            r.unsqueeze(1) - r.unsqueeze(2), 
            dim=-1
        )

        # Create causal and padding masks

        diag_causal_mask = torch.diag(torch.ones(L)).bool().to(padding.device)
        padding_causal_mask = (padding.unsqueeze(-2) | padding.unsqueeze(-1)).unsqueeze(1)
        causal_mask = padding_causal_mask | diag_causal_mask.expand_as(padding_causal_mask)

        padding_mask = padding.unsqueeze(-1).expand(B, L, self.E)

        # Forward Pass

        e = self.embed(tokens)
        
        for transformer_block in self.transformer_blocks:
            e = transformer_block(
                e=e, d=d, 
                causal_mask=causal_mask, 
                padding_mask=padding_mask, 
            )

        e = self.norm(e)
        e = e.mean(dim=1)

        return self.out_map(e)

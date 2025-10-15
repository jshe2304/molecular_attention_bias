import torch
import torch.nn as nn

from .modules.radial_functions import get_radial_function
from .modules.attention import SDPAttention

class BiasedAttentionTransformerBlock(nn.Module):
    def __init__(self, E, H, radial_function_type, dropout=0.1, **radial_kwargs):
        super().__init__()

        # Radial function

        self.radial_function = get_radial_function(
            radial_function_type, 
            H, **radial_kwargs
        )

        # Layers

        self.attn = SDPAttention(E, H)
        self.norm_1 = nn.LayerNorm(E)
        self.mlp = nn.Sequential(
            nn.Linear(E, E * 4), 
            nn.ReLU(), 
            nn.Linear(E * 4, E)
        )
        self.norm_2 = nn.LayerNorm(E)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, e, d, attn_mask, padding_mask, **kwargs):

        B, L, E = e.shape

        # Scale bias

        attn_bias = self.radial_function(
            d.reshape(B, L, L), tokens
        )

        # Attention block

        e0 = self.norm_1(e)
        e1 = self.attn(e0, attn_mask=attn_mask, attn_bias=attn_bias, **kwargs)
        e1 = self.dropout(e1)
        e2 = e1 + e0
        
        # MLP block

        e2 = self.norm_2(e2)
        e3 = self.mlp(e2)
        e3.masked_fill_(padding_mask, 0)
        e3 = self.dropout(e3)
        e4 = e3 + e2

        return e4

class BiasedAttentionTransformer(nn.Module):
    def __init__(
            self, 
            n_tokens, out_features, 
            E, H, D, 
            radial_function_type, 
            dropout=0.1, 
            **radial_kwargs
        ):
        super().__init__()

        self.E, self.H, self.D = E, H, D

        # Embedding layer
        self.embed = nn.Embedding(n_tokens, E, padding_idx=0)

        # Transformer blocks
        if type(radial_function_type) is list:
            assert len(radial_function_type) == D
            self.transformer_blocks = nn.ModuleList([
                BiasedAttentionTransformerBlock(
                    E, H, radial_function_type[i], dropout=dropout, **radial_kwargs[i]
                ) for i in range(D)
            ])
        else:
            self.transformer_blocks = nn.ModuleList([
                BiasedAttentionTransformerBlock(
                    E, H, radial_function_type, dropout=dropout, **radial_kwargs
                ) for _ in range(D)
            ])
        
        # Out map
        self.norm = nn.LayerNorm(E)
        self.out_map = nn.Linear(E, out_features)

    def forward(self, tokens, padding, r, **kwargs):

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
        attn_mask = padding_causal_mask | diag_causal_mask.expand_as(padding_causal_mask)

        padding_mask = padding.unsqueeze(-1).expand(B, L, self.E)

        # Forward Pass

        e = self.embed(tokens)
        
        for transformer_block in self.transformer_blocks:
            e = transformer_block(
                tokens=tokens, e=e, d=d, 
                padding_mask=padding_mask, 
                attn_mask=attn_mask, 
                **kwargs
            )
        e = self.norm(e)
        e = e.mean(dim=1)
        
        return self.out_map(e)

import torch
import torch.nn as nn

from .modules import graph_positional_encodings
from .modules.attention import SDPAttention

class TransformerBlock(nn.Module):
    def __init__(self, E, H, dropout=0.1):
        super().__init__()

        self.operator = SDPAttention(E, H)
        self.norm_1 = nn.LayerNorm(E)
        self.mlp = nn.Sequential(
            nn.Linear(E, E * 4), 
            nn.ReLU(), 
            nn.Linear(E * 4, E)
        )
        self.norm_2 = nn.LayerNorm(E)
        self.dropout = nn.Dropout(dropout)

    def forward(self, e, attn_mask, padding_mask):

        B, L, E = e.shape

        # Attention block

        e0 = self.norm_1(e)
        e1 = self.operator(e0, attn_mask=attn_mask)
        e1 = self.dropout(e1)
        e2 = e1 + e0

        # MLP block

        e2 = self.norm_2(e2)
        e3 = self.mlp(e2)
        e3.masked_fill_(padding_mask, 0)
        e3 = self.dropout(e3)
        e4 = e3 + e2

        return e4

class GraphPETransformer(nn.Module):
    def __init__(
            self, 
            n_tokens, out_features, 
            E, H, D, 
            positional_encoding_name, 
            dropout=0.1,
            **pe_kwargs
        ):
        super().__init__()

        self.E, self.H, self.D = E, H, D

        # Embedding layer
        self.embed = nn.Embedding(n_tokens, E, padding_idx=0)

        # Positonal Encoding
        PositionalEncoding = graph_positional_encodings.name_to_module[positional_encoding_name]
        self.positional_encoding = PositionalEncoding(E, **pe_kwargs)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                E=E, H=H, 
                dropout=dropout
            ) for _ in range(D)
        ])
        
        # Out map
        self.norm = nn.LayerNorm(E)
        self.out_map = nn.Linear(E, out_features)

    def forward(self, tokens, padding, adj):

        B, L, *_ = tokens.shape

        # Create causal and padding masks

        diag_causal_mask = torch.diag(torch.ones(L)).bool().to(padding.device)
        padding_causal_mask = (padding.unsqueeze(-2) | padding.unsqueeze(-1)).unsqueeze(1)
        causal_mask = padding_causal_mask | diag_causal_mask.expand_as(padding_causal_mask)

        padding_mask = padding.unsqueeze(-1).expand(B, L, self.E)

        # Forward Pass

        e = self.embed(tokens)

        pos_enc = self.positional_encoding(adj)
        for transformer_block in self.transformer_blocks:
            e += pos_enc
            e = transformer_block(
                e=e, adj=adj, 
                padding_mask=padding_mask, 
                attn_mask=causal_mask, 
            )

        e = self.norm(e)
        e = e.mean(dim=1)

        return self.out_map(e)
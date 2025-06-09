import torch
import torch.nn as nn

class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)

    def forward(self, x):
        return nn.functional.linear(x, self.log_weight.exp())

class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=None):
        super(MLP, self).__init__()

        if hidden_features is None:
            hidden_features = out_features * 2

        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features), 
            nn.ReLU(), 
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x):
        return self.mlp(x)

import torch
import torch.nn as nn
import torch.nn.functional as F


class Molecule3DBias(nn.Module):
    """
    Computes 3D attention bias from atomic coordinates using learned Gaussian basis functions.
    Outputs:
        - attention bias tensor: (B, H, N, N)
        - pooled edge features: (B, N, D)
        - normalized relative positions: (B, N, N, 3)
    """

    def __init__(
        self,
        num_heads: int,
        num_edges: int,
        n_layers: int,
        embed_dim: int,
        n_kernels: int,
        no_share_rpe: bool = False
    ):
        super().__init__()
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.n_kernels = n_kernels
        self.no_share_rpe = no_share_rpe

        # Per-head RPE projection: [K → H] or [K → H × L]
        rpe_heads = num_heads * n_layers if no_share_rpe else num_heads

        # Gaussian basis parameters
        self.means = nn.Parameter(torch.empty(n_kernels).uniform_(0, 3))
        self.stds = nn.Parameter(torch.empty(n_kernels).uniform_(0.1, 1.0))

        # Edge-type affine transform: γ, β per edge type
        self.mul = nn.Embedding(num_edges, 1, padding_idx=0)
        self.bias = nn.Embedding(num_edges, 1, padding_idx=0)
        nn.init.constant_(self.mul.weight, 1.0)
        nn.init.constant_(self.bias.weight, 0.0)

        # Project Gaussian basis to attention bias
        self.mlp = nn.Sequential(
            nn.Linear(n_kernels, n_kernels),
            nn.ReLU(),
            nn.Linear(n_kernels, rpe_heads)
        )

        # Project pooled edge features to model dimension if needed
        self.edge_proj = nn.Linear(n_kernels, embed_dim) if n_kernels != embed_dim else nn.Identity()

    def forward(self, batched_data):
        pos = batched_data['pos']                    # (B, N, 3)
        x = batched_data['x']                        # (B, N, D)
        edge_types = batched_data.get('node_type_edge')  # (B, N, N) or None

        B, N, _ = pos.shape

        # 1. Compute pairwise distance vectors and norms
        delta = pos.unsqueeze(1) - pos.unsqueeze(2)  # (B, N, N, 3)
        dist = delta.norm(dim=-1)                    # (B, N, N)
        normed_delta = delta / (dist.unsqueeze(-1) + 1e-5)  # (B, N, N, 3)

        # 2. Affine transform distances via edge types
        if edge_types is None:
            edge_types = torch.zeros_like(dist, dtype=torch.long)

        gamma = self.mul(edge_types).squeeze(-1)     # (B, N, N)
        beta = self.bias(edge_types).squeeze(-1)     # (B, N, N)
        transformed = gamma * dist + beta            # (B, N, N)

        # 3. Gaussian basis expansion: ψ_k(d)
        d_expanded = transformed.unsqueeze(-1)       # (B, N, N, 1)
        mu = self.means.view(1, 1, 1, -1)             # (1, 1, 1, K)
        sigma = self.stds.view(1, 1, 1, -1).abs() + 1e-2

        psi = torch.exp(-0.5 * ((d_expanded - mu) / sigma) ** 2) / (sigma * (2 * torch.pi).sqrt())  # (B, N, N, K)

        # 4. Project to attention bias per head
        attn_bias = self.mlp(psi).permute(0, 3, 1, 2)    # (B, H, N, N)

        # 6. Aggregate edge features and project
        pooled_features = psi.sum(dim=2)             # (B, N, K)
        merged_features = self.edge_proj(pooled_features)  # (B, N, D)

        return attn_bias, merged_features, normed_delta

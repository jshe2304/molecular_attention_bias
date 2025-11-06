import torch
import torch.nn as nn
import math

class Zeros(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Zeros, self).__init__()

    def forward(self, d, *args, **kwargs): 
        return torch.zeros_like(d).unsqueeze(1)

class FixedPowerLaw(nn.Module):
    def __init__(self, H, p, *args, **kwargs):
        super(FixedPowerLaw, self).__init__()
        self.p = float(p)

    def forward(self, d, *args, **kwargs):
        return self.p * (
            torch.log(d).nan_to_num(torch.finfo(d.dtype).max)
        ).unsqueeze(1)

class ExpNegativePowerLaw(nn.Module):
    def __init__(self, n_heads, *args, **kwargs):
        super(ExpNegativePowerLaw, self).__init__()
        
        self.log_p = nn.Parameter(torch.Tensor(n_heads, 1))
        nn.init.xavier_uniform_(self.log_p)

    def forward(self, d, *args, **kwargs):
        
        return -nn.functional.linear(
            torch.log(d).unsqueeze(-1).nan_to_num(torch.finfo(d.dtype).max), 
            self.log_p.exp()
        ).permute(0, 3, 1, 2)

class SoftplusNegativePowerLaw(nn.Module):
    def __init__(self, n_heads, *args, **kwargs):
        super(SoftplusNegativePowerLaw, self).__init__()
        
        self.log_p = nn.Parameter(torch.Tensor(n_heads, 1))
        nn.init.xavier_uniform_(self.log_p)

    def forward(self, d, *args, **kwargs):
        
        return -nn.functional.linear(
            torch.log(d).unsqueeze(-1).nan_to_num(torch.finfo(d.dtype).max), 
            nn.functional.softplus(self.log_p)
        ).permute(0, 3, 1, 2)

class PowerLaw(nn.Module):
    def __init__(self, n_heads, *args, **kwargs):
        super(PowerLaw, self).__init__()
        
        self.p = nn.Parameter(torch.Tensor(n_heads, 1))
        nn.init.xavier_uniform_(self.p)

    def forward(self, d, *args, **kwargs):
        return nn.functional.linear(
            torch.log(d).unsqueeze(-1).nan_to_num(torch.finfo(d.dtype).max), 
            self.p
        ).permute(0, 3, 1, 2)
    
class InitPowerLaw(nn.Module):
    def __init__(self, n_heads, *args, **kwargs):
        super(InitPowerLaw, self).__init__()
        
        self.p = nn.Parameter(torch.Tensor(n_heads, 1))
        nn.init.zeros_(self.p)

    def forward(self, d, *args, **kwargs):
        return nn.functional.linear(
            torch.log(d).unsqueeze(-1).nan_to_num(torch.finfo(d.dtype).max), 
            self.p
        ).permute(0, 3, 1, 2)
    
class NormalizedPowerLaw(nn.Module):
    def __init__(self, n_heads, *args, **kwargs):
        super(NormalizedPowerLaw, self).__init__()
        
        self.p = nn.Parameter(torch.Tensor(n_heads, 1))
        nn.init.xavier_uniform_(self.p)

    def forward(self, d, *args, **kwargs):

        d_max = (d * (d < 1e8).int()).nan_to_num(0, 0, 0).amax(dim=-1, keepdim=True)

        return nn.functional.linear(
            torch.log(d / d_max).unsqueeze(-1).nan_to_num(torch.finfo(d.dtype).max), 
            self.p
        ).permute(0, 3, 1, 2)

class MLP(nn.Module):
    def __init__(self, n_heads, k, *args, **kwargs):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, k),
            nn.ReLU(),
            nn.Linear(k, n_heads)
        )

    def forward(self, d, *args, **kwargs):
        return self.mlp(
            d.unsqueeze(-1).nan_to_num(torch.finfo(d.dtype).max)
        ).permute(0, 3, 1, 2)

class GaussianKernel(nn.Module):
    """
    Pair-type aware Gaussian kernel expansion attention bias
    """
    def __init__(self, n_heads, n_tokens, n_kernels=32, *args, **kwargs):
        super().__init__()
        
        self.n_heads = n_heads
        self.n_kernels = n_kernels

        self.mu = nn.Parameter(torch.linspace(0., 6., n_kernels))
        self.log_sigma = nn.Parameter(torch.zeros(n_kernels))

        self.mlp = nn.Sequential(
            nn.Linear(n_kernels, n_kernels, bias=True),
            nn.ReLU(),
            nn.Linear(n_kernels, n_heads, bias=True)
        )

        self.gamma_table = nn.Parameter(torch.ones(n_tokens, n_tokens))
        self.beta_table  = nn.Parameter(torch.zeros(n_tokens, n_tokens))

    def forward(self, d: torch.Tensor, tokens: torch.Tensor, *args, **kwargs):
        """
        Args:
            d: (B, N, N) interatomic distances
            tokens: (B, N) int64 atom-type indices in [0, T)
            attn_mask: optional (B, N) bool mask (True = keep / False = pad) or (B, 1, 1, N)

        Returns:
            bias_3d: (B, heads, N, N) to add to attention logits
            node_term (optional): (B, N, d_model) if enable_node_term() was called, else None
        """
        B, N, _ = d.shape
        d = d.nan_to_num(0, posinf=0, neginf=0)

        # Look up beta, gamma and scale distance
        ti = tokens[:, :, None].expand(B, N, N)
        tj = tokens[:, None, :].expand(B, N, N)
        gamma = self.gamma_table[ti, tj]
        beta  = self.beta_table[ti, tj]
        s = gamma * d + beta

        # Gaussian basis expansion ψ_k(s)
        sigma = nn.functional.softplus(self.log_sigma) + 1e-6
        x = (s.unsqueeze(-1) - self.mu) / sigma
        psi = torch.exp(-0.5 * x ** 2) / (math.sqrt(2.0 * math.pi) * sigma)

        phi = self.mlp(psi) # (B, N, N, n_heads)

        return phi.permute(0, 3, 1, 2) # (B, n_heads, N, N)

class PairedPowerLaw(nn.Module):
    """
    Pair-type aware power law expansion attention bias
    """
    def __init__(self, n_heads, n_tokens, *args, **kwargs):
        super().__init__()
        
        self.p_table = nn.Parameter(torch.ones(n_tokens, n_tokens, n_heads))

    def forward(self, d: torch.Tensor, tokens: torch.Tensor, *args, **kwargs):
        """
        Args:
            d: (B, N, N) interatomic distances
            tokens: (B, N) int64 atom-type indices in [0, T)
        """
        B, N, _ = d.shape

        # Look up p
        ti = tokens[:, :, None].expand(B, N, N)
        tj = tokens[:, None, :].expand(B, N, N)
        p = self.p_table[ti, tj]

        log_d = torch.log(d).unsqueeze(-1).nan_to_num(torch.finfo(d.dtype).max)
        
        # Power law expansion
        return (p * log_d).permute(0, 3, 1, 2)


def get_radial_function(radial_function_type: str, *args, **kwargs):
    if radial_function_type == "FixedPowerLaw": 
        return FixedPowerLaw(*args, **kwargs)
    elif radial_function_type == "ExpNegativePowerLaw": 
        return ExpNegativePowerLaw(*args, **kwargs)
    elif radial_function_type == "SoftplusNegativePowerLaw": 
        return SoftplusNegativePowerLaw(*args, **kwargs)
    elif radial_function_type == "PowerLaw": 
        return PowerLaw(*args, **kwargs)
    elif radial_function_type == "NormalizedPowerLaw": 
        return NormalizedPowerLaw(*args, **kwargs)
    elif radial_function_type == "InitPowerLaw": 
        return InitPowerLaw(*args, **kwargs)
    elif radial_function_type == "MLP": 
        return MLP(*args, **kwargs)
    elif radial_function_type == "GaussianKernel": 
        return GaussianKernel(*args, **kwargs)
    elif radial_function_type == "Zeros": 
        return Zeros()
    elif radial_function_type == "PairedPowerLaw": 
        return PairedPowerLaw(*args, **kwargs)
    else:
        raise ValueError(f"Invalid radial function type: {radial_function_type}")

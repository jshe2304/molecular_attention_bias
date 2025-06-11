import torch
import torch.nn as nn

class FixedPowerLaw1(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FixedPowerLaw1, self).__init__()

    def forward(self, d):
        return -1 * (
            torch.log(d).nan_to_num(torch.finfo(d.dtype).max)
        ).unsqueeze(1)

class FixedPowerLaw2(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FixedPowerLaw2, self).__init__()

    def forward(self, d):
        return -2 * (
            torch.log(d).nan_to_num(torch.finfo(d.dtype).max)
        ).unsqueeze(1)

class FixedPowerLaw3(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FixedPowerLaw3, self).__init__()

    def forward(self, d):
        return -3 * (
            torch.log(d).nan_to_num(torch.finfo(d.dtype).max)
        ).unsqueeze(1)

class FixedPowerLaw4(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FixedPowerLaw4, self).__init__()

    def forward(self, d):
        return -4 * (
            torch.log(d).nan_to_num(torch.finfo(d.dtype).max)
        ).unsqueeze(1)

class FixedPowerLaw5(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FixedPowerLaw5, self).__init__()

    def forward(self, d):
        return -5 * (
            torch.log(d).nan_to_num(torch.finfo(d.dtype).max)
        ).unsqueeze(1)

class LearnedNegativePowerLaw(nn.Module):
    def __init__(self, n_heads, *args, **kwargs):
        super(LearnedNegativePowerLaw, self).__init__()
        
        self.log_p = nn.Parameter(torch.Tensor(n_heads, 1))
        nn.init.xavier_uniform_(self.log_p)

    def forward(self, d):
        
        return -nn.functional.linear(
            torch.log(d).unsqueeze(-1).nan_to_num(torch.finfo(d.dtype).max), 
            self.log_p.exp()
        ).permute(0, 3, 1, 2)

class LearnedPowerLaw(nn.Module):
    def __init__(self, n_heads, *args, **kwargs):
        super(LearnedPowerLaw, self).__init__()
        
        self.p = nn.Parameter(torch.Tensor(n_heads, 1))
        nn.init.xavier_uniform_(self.p)

    def forward(self, d):
        return nn.functional.linear(torch.log(d), self.p)

class GaussianBasis(nn.Module):

    def __init__(self, n_heads, n_kernels=32, *args, **kwargs):
        super().__init__()

        # Gaussian parameters
        
        self.mu = nn.Parameter(torch.linspace(0, 4, n_kernels))
        self.log_sigma = nn.Parameter(torch.full((n_kernels,), -1.609437912))

        # MLP map to attention bias
        
        self.mlp = nn.Sequential(
            nn.Linear(n_kernels, n_kernels),
            nn.ReLU(),
            nn.Linear(n_kernels, n_heads)
        )

    def forward(self, d):

        # Gaussian Basis Expansion

        mu = self.mu.view(1, 1, 1, -1)
        sigma = self.log_sigma.exp().view(1, 1, 1, -1)
        d = d.unsqueeze(-1)

        gaussians = torch.exp(-0.5 * ((d - mu) / sigma) ** 2) / (
            sigma * 2.506628275
        )

        # Map to attention bias

        attn_bias = self.mlp(gaussians).permute(0, 3, 1, 2)  # (B, H, N, N)

        return attn_bias.nan_to_num(torch.finfo(d.dtype).max)

bias_maps = {
    "FixedPowerLaw1": FixedPowerLaw1,
    "FixedPowerLaw2": FixedPowerLaw2,
    "FixedPowerLaw3": FixedPowerLaw3,
    "FixedPowerLaw4": FixedPowerLaw4,
    "FixedPowerLaw5": FixedPowerLaw5,
    "LearnedNegativePowerLaw": LearnedNegativePowerLaw,
    "LearnedPowerLaw": LearnedPowerLaw,
    "GaussianBasis": GaussianBasis,
}

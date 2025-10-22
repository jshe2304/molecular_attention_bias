import torch.nn as nn

class OutMap(nn.Module):
    '''
    Prediction head predicts values from a (B, L, E) tensor. 
    Produces a (B, L, 3) forces tensor and a (B, L, 1) energy tensor.
    '''
    def __init__(self, E):
        super().__init__()
        self.expansion_map = nn.Sequential(
            nn.Linear(E, E), 
            nn.ReLU(), 
        )
        self.force_map = nn.Linear(E, 3)
        self.energy_map = nn.Linear(E, 1)

    def forward(self, e):

        # First mapping and then separate features for forces and energy.

        e = self.expansion_map(e) # (B, L, E) -> (B, L, 2E)
        e_forces, e_energy = e.chunk(2, dim=-1) # (B, L, 2E) -> (B, L, E), (B, L, E)

        # Apply separate maps to forces and energy.

        forces = self.force_map(e_forces) # (B, L, E) -> (B, L, 3)
        energy = self.energy_map(e_energy).sum(dim=1) # (B, L, E) -> (B, 1)

        return forces, energy
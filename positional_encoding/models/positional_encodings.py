import torch
import torch.nn as nn

class RandomWalkPE(nn.Module):
    def __init__(self, k, embed_dim):
        super(RandomWalkPE, self).__init__()
        self.k = k
        self.out_map = nn.Linear(k, embed_dim)

    def forward(self, adjs):
        """
        Compute Random Walk Positional Encoding for batched adjacency matrices.

        Args:
            adj_matrices: Batched adjacency matrices

        Returns:
            Positional encodings tensor shape: (B, N, E)
        """
        B, N, _ = adjs.shape

        # Compute degree matrix D^-1
        degrees = adjs.sum(dim=2).clamp(min=1)  # Avoid division by zero
        D_inv = torch.diag_embed(1.0 / degrees)  # Diagonal inverse degree matrix

        # Compute Random Walk matrix: RW = A * D^-1
        RW = torch.bmm(adjs.float(), D_inv)

        # Initialize positional encoding
        M = RW
        pos_enc = [torch.diagonal(M, dim1=1, dim2=2)]  # Extract diagonal for PE dimension 1

        # Compute higher powers of the Random Walk matrix
        for _ in range(self.k - 1):
            M = torch.bmm(M, RW)  # Multiply with RW
            pos_enc.append(torch.diagonal(M, dim1=1, dim2=2))  # Extract diagonal for PE dimension

        # Stack positional encodings
        pos_enc = torch.stack(pos_enc, dim=-1)  # Resulting shape: (batch_size, num_nodes, pos_enc_dim)

        return self.out_map(pos_enc)

pos_encs = {
    'RandomWalkPE': RandomWalkPE
}

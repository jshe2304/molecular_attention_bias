import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem

class RWPEEncoder(nn.Module):
    def __init__(self, embed_dim, n_steps=3):
        super().__init__()
        self.n_steps = n_steps
        self.linear = nn.Linear(n_steps, embed_dim)

    def forward(self, smiles_list):
        pe_list = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            adj = self._mol_to_adj(mol)
            pe = self._compute_rwpe(adj)
            pe_list.append(pe)

        # Pad sequences to match batch
        pe_padded = pad_sequence(pe_list, batch_first=True)  # shape: [B, N_max, k]
        return self.linear(pe_padded)  # shape: [B, N_max, pe_dim]

    def _mol_to_adj(self, mol):
        num_atoms = mol.GetNumAtoms()
        adj = torch.zeros((num_atoms, num_atoms), dtype=torch.float32)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            adj[i, j] = 1
            adj[j, i] = 1
        return adj

    def _compute_rwpe(self, adj):
        n = adj.size(0)
        deg = adj.sum(dim=1)  # [n]
        deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
        D_inv = torch.diag(deg_inv)
        RW = torch.matmul(adj, D_inv)  # Random Walk operator: A * D^{-1}

        rw_diag = []
        mat_power = torch.eye(n)
        for _ in range(self.k):
            mat_power = torch.matmul(mat_power, RW)
            rw_diag.append(torch.diagonal(mat_power))  # [n]

        return torch.stack(rw_diag, dim=1)  # [n, k]

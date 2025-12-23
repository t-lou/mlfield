import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        batch_size = z_i.shape[0]
        representations = torch.cat([z_i, z_j], dim=0)  # 2N x D

        # Similarity matrix
        sim_matrix = torch.matmul(representations, representations.T) / self.temperature

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)

        # Positive pairs: i with j
        positives = torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]).to(z_i.device)

        loss = F.cross_entropy(sim_matrix, positives)
        return

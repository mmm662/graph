from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeFeatureEncoder(nn.Module):
    def __init__(self, num_floors, num_in_dim=5, floor_emb_dim=8, out_dim=64):
        super().__init__()
        if out_dim <= floor_emb_dim:
            raise ValueError("out_dim must be > floor_emb_dim")
        self.floor_emb = nn.Embedding(num_floors, floor_emb_dim)
        self.num_proj = nn.Linear(num_in_dim, out_dim - floor_emb_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
    def forward(self, node_num_feat, floor_id):
        f = self.floor_emb(floor_id)
        n = F.relu(self.num_proj(node_num_feat))
        return F.relu(self.out_proj(torch.cat([n,f], dim=-1)))

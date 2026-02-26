from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dims, dropout=0.0):
        super().__init__()
        layers=[]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class DotAttention(nn.Module):
    def forward(self, q, K, V, mask=None):
        score = torch.einsum("bd,bld->bl", q, K)
        if mask is not None:
            score = score.masked_fill(~mask, -1e9)
        alpha = F.softmax(score, dim=-1)
        return torch.einsum("bl,bld->bd", alpha, V)

class GINEdgeLayer(nn.Module):
    def __init__(self, dim, edge_feat_dim, mlp_hidden, dropout=0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.msg_mlp = MLP([dim+edge_feat_dim, mlp_hidden, dim], dropout=dropout)
        self.self_mlp = MLP([dim, mlp_hidden, dim], dropout=dropout)
    def forward(self, H, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        msg = self.msg_mlp(torch.cat([H[src], edge_attr], dim=-1))
        agg = torch.zeros_like(H)
        agg.index_add_(0, dst, msg)
        return self.self_mlp((1.0 + self.eps) * H + agg)

class WeightedGCNLayer(nn.Module):
    """Edge-index GCN with optional scalar edge weights."""
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)

    def forward(self, H: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        # symmetric normalization: w_ij / sqrt(deg_i deg_j)
        src, dst = edge_index[0], edge_index[1]
        N = H.size(0)
        # degrees
        deg = torch.zeros(N, device=H.device)
        deg.index_add_(0, dst, edge_weight)
        deg = deg.clamp(min=1e-6)
        norm = edge_weight / torch.sqrt(deg[dst] * deg[src]).clamp(min=1e-6)
        msg = self.lin(H[src]) * norm.unsqueeze(-1)
        out = torch.zeros_like(H)
        out.index_add_(0, dst, msg)
        return out

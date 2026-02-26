from __future__ import annotations
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from graphmm.models.layers import GINEdgeLayer, DotAttention, WeightedGCNLayer
from graphmm.models.node_encoder import NodeFeatureEncoder
from graphmm.models.crf import GraphCRF
from graphmm.datasets.trajectory_provider import PADDING_ID

class GraphMMCorrector(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_floors: int,
        road_dim: int = 64,
        gin_layers: int = 2,
        gin_mlp_hidden: int = 128,
        edge_feat_dim: int = 5,
        dropout: float = 0.1,
        temperature: float = 12.0,
        node_num_dim: int = 5,
        floor_emb_dim: int = 8,
        traj_gcn_layers: int = 1,
        use_crf: bool = True,
        unreachable_penalty: float = -1e4,
        input_anchor_bias: float = 0.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.road_dim = road_dim
        self.temperature = temperature
        self.use_crf = use_crf
        self.input_anchor_bias = float(input_anchor_bias)

        self.node_encoder = NodeFeatureEncoder(num_floors, node_num_dim, floor_emb_dim, road_dim)

        self.edge_proj = nn.Sequential(
            nn.Linear(edge_feat_dim, road_dim),
            nn.ReLU(),
            nn.Linear(road_dim, road_dim),
        )
        self.gine = nn.ModuleList([
            GINEdgeLayer(road_dim, road_dim, gin_mlp_hidden, dropout) for _ in range(gin_layers)
        ])

        # trajectory correlation encoder (GCN on transition graph)
        self.traj_gcn = nn.ModuleList([WeightedGCNLayer(road_dim) for _ in range(traj_gcn_layers)])

        # Seq2Seq
        self.encoder = nn.GRU(road_dim, road_dim, batch_first=True, bidirectional=True)
        self.enc_proj = nn.Linear(road_dim*2, road_dim)

        self.decoder = nn.GRU(road_dim, road_dim, batch_first=True)
        self.attn = DotAttention()
        self.dec_out = nn.Linear(road_dim*2, road_dim)

        self.crf = GraphCRF(road_dim, unreachable_penalty=unreachable_penalty) if use_crf else None

    def compute_H_R(self, node_num_feat, floor_id, edge_index, edge_attr):
        H = self.node_encoder(node_num_feat, floor_id)
        Eemb = self.edge_proj(edge_attr)
        for layer in self.gine:
            H = F.relu(layer(H, edge_index, Eemb))
        return F.normalize(H, p=2, dim=-1)

    def enhance_with_traj_graph(self, H_R: torch.Tensor, traj_edge_index: Optional[torch.Tensor], traj_edge_weight: Optional[torch.Tensor]):
        if traj_edge_index is None or traj_edge_weight is None or len(self.traj_gcn) == 0:
            return H_R
        H = H_R
        for gcn in self.traj_gcn:
            H = F.relu(gcn(H, traj_edge_index, traj_edge_weight))
        return F.normalize(H, p=2, dim=-1)

    def hidden_similarity(self, Z, H_R):
        return torch.einsum("bld,nd->bln", Z, H_R) * self.temperature


    def _apply_input_anchor_bias(self, logits: torch.Tensor, pred_safe: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.input_anchor_bias == 0.0:
            return logits
        B, L, _ = logits.shape
        b_idx = torch.arange(B, device=logits.device).unsqueeze(1).expand(B, L)
        t_idx = torch.arange(L, device=logits.device).unsqueeze(0).expand(B, L)
        logits[b_idx[mask], t_idx[mask], pred_safe[mask]] += self.input_anchor_bias
        return logits

    def forward_unary(
        self,
        pred_seq: torch.Tensor,
        lengths: torch.Tensor,
        node_num_feat: torch.Tensor,
        floor_id: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        traj_edge_index: Optional[torch.Tensor] = None,
        traj_edge_weight: Optional[torch.Tensor] = None,
        teacher_forcing: Optional[torch.Tensor] = None,
    ):
        device = pred_seq.device
        B, L = pred_seq.shape

        H_R = self.compute_H_R(node_num_feat, floor_id, edge_index, edge_attr)
        H_R = self.enhance_with_traj_graph(H_R, traj_edge_index, traj_edge_weight)

        pred_safe = pred_seq.clone()
        pred_safe[pred_safe == PADDING_ID] = 0
        enc_inp = H_R[pred_safe]

        packed = nn.utils.rnn.pack_padded_sequence(enc_inp, lengths.cpu(), batch_first=True, enforce_sorted=False)
        enc_out_packed, _ = self.encoder(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(enc_out_packed, batch_first=True, total_length=L)
        enc_out = F.normalize(self.enc_proj(enc_out), p=2, dim=-1)

        mask = (pred_seq != PADDING_ID)
        enc_mean = (enc_out * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp(min=1).unsqueeze(-1)
        dec_h = enc_mean.unsqueeze(0)

        zero = torch.zeros(B, self.road_dim, device=device)

        if teacher_forcing is None:
            # Autoregressive decoding for inference: feed previous predicted node embedding.
            # This avoids constant decoder inputs (all zeros) that collapse outputs.
            prev_emb = zero
            logits_steps = []
            for _ in range(L):
                dec_step, dec_h = self.decoder(prev_emb.unsqueeze(1), dec_h)
                dec_step = dec_step[:, 0, :]
                ctx_step = self.attn(dec_step, enc_out, enc_out, mask=mask)
                z_step = F.normalize(self.dec_out(torch.cat([dec_step, ctx_step], dim=-1)), p=2, dim=-1)
                logit_step = torch.einsum("bd,nd->bn", z_step, H_R) * self.temperature
                logits_steps.append(logit_step.unsqueeze(1))

                pred_ids = torch.argmax(logit_step, dim=-1)
                prev_emb = H_R[pred_ids]

            logits = torch.cat(logits_steps, dim=1)
            logits = self._apply_input_anchor_bias(logits, pred_safe, mask)
            return logits, H_R

        tf = teacher_forcing.clone()
        tf[tf == PADDING_ID] = 0

        dec_inputs = []
        prev_emb = zero
        for t in range(L):
            dec_inputs.append(prev_emb.unsqueeze(1))
            prev_emb = H_R[tf[:, t]]
        dec_inp = torch.cat(dec_inputs, dim=1)
        dec_out, _ = self.decoder(dec_inp, dec_h)

        ctx = []
        for t in range(L):
            ctx.append(self.attn(dec_out[:, t, :], enc_out, enc_out, mask=mask).unsqueeze(1))
        ctx_all = torch.cat(ctx, dim=1)

        Z = F.normalize(self.dec_out(torch.cat([dec_out, ctx_all], dim=-1)), p=2, dim=-1)
        logits = self.hidden_similarity(Z, H_R)
        return logits, H_R

@torch.no_grad()
def decode_argmax(unary_logits, lengths):
    pred = torch.argmax(unary_logits, dim=-1)
    return [pred[i,:int(lengths[i].item())].tolist() for i in range(pred.size(0))]


@torch.no_grad()
def confidence_gate_sequence(
    raw_seq: List[int],
    corrected_seq: List[int],
    unary_logits: torch.Tensor,
    min_confidence: float,
    min_logit_gain: float = 0.0,
) -> List[int]:
    L = min(len(raw_seq), len(corrected_seq), int(unary_logits.size(0)))
    if L <= 0:
        return corrected_seq

    probs = torch.softmax(unary_logits[:L], dim=-1)
    conf = torch.max(probs, dim=-1).values

    out = corrected_seq[:]
    for t in range(L):
        raw_id = int(raw_seq[t])
        corr_id = int(corrected_seq[t])
        if raw_id == corr_id:
            continue

        conf_ok = (float(conf[t].item()) >= float(min_confidence))
        gain = float(unary_logits[t, corr_id].item() - unary_logits[t, raw_id].item())
        gain_ok = (gain >= float(min_logit_gain))

        if not (conf_ok and gain_ok):
            out[t] = raw_id
    return out

from __future__ import annotations
from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


def adaptive_top_r_from_unary(
    unary_logits: torch.Tensor,
    min_top_r: int,
    max_top_r: int,
) -> int:
    """Compute candidate size from model uncertainty (higher entropy => larger top_r)."""
    _, n_nodes = unary_logits.shape
    lo = max(1, min(int(min_top_r), n_nodes))
    hi = max(lo, min(int(max_top_r), n_nodes))
    if lo == hi:
        return lo

    probs = torch.softmax(unary_logits, dim=-1).clamp(min=1e-9)
    entropy = -(probs * probs.log()).sum(dim=-1)
    entropy_norm = (entropy / torch.log(torch.tensor(float(n_nodes), device=unary_logits.device))).mean()
    entropy_norm = float(torch.clamp(entropy_norm, 0.0, 1.0).item())

    return int(round(lo + (hi - lo) * entropy_norm))


class GraphCRF(nn.Module):
    """Candidate-pruned CRF with bilinear pairwise and k-hop reachability constraint."""
    def __init__(self, emb_dim: int, unreachable_penalty: float = -1e4):
        super().__init__()
        self.W = nn.Parameter(torch.empty(emb_dim, emb_dim))
        nn.init.xavier_uniform_(self.W)
        self.unreachable_penalty = unreachable_penalty

    def pairwise_scores(self, H: torch.Tensor, prev_ids: torch.Tensor, cur_ids: torch.Tensor) -> torch.Tensor:
        Hp = H[prev_ids]     # [P,d]
        Hc = H[cur_ids]      # [C,d]
        return (Hp @ self.W) @ Hc.t()  # [P,C]

    def nll_one(
        self,
        unary_logits: torch.Tensor,     # [L,N]
        gold: List[int],                # length L
        H: torch.Tensor,                # [N,d]
        allowed_prev: List[List[int]],
        top_r: int = 50,
        train_mode: bool = True,
    ) -> torch.Tensor:
        device = unary_logits.device
        L, N = unary_logits.shape
        gold_t = torch.tensor(gold, dtype=torch.long, device=device)

        cand_ids = []
        cand_maps: List[Dict[int,int]] = []
        for t in range(L):
            r = min(top_r, N)
            _, topi = torch.topk(unary_logits[t], k=r)
            cset = set(topi.tolist())
            if train_mode:
                cset.add(int(gold_t[t].item()))
            c = torch.tensor(sorted(cset), dtype=torch.long, device=device)
            cand_ids.append(c)
            cand_maps.append({int(x.item()): j for j,x in enumerate(c)})

        # forward log-space
        u0 = unary_logits[0][cand_ids[0]]
        alpha = [F.log_softmax(u0, dim=-1)]
        for t in range(1, L):
            prev = cand_ids[t-1]
            cur = cand_ids[t]
            u = unary_logits[t][cur]
            pair = self.pairwise_scores(H, prev, cur)  # [P,C]

            # reachability mask
            prev_list = prev.tolist()
            mask = torch.zeros(pair.shape, dtype=torch.bool, device=device)
            for j, cur_label in enumerate(cur.tolist()):
                allowed = set(allowed_prev[cur_label])
                for i, p_label in enumerate(prev_list):
                    if p_label in allowed:
                        mask[i,j] = True
            pair = pair + (~mask).float() * self.unreachable_penalty

            prev_alpha = alpha[-1].unsqueeze(1)  # [P,1]
            scores = prev_alpha + pair
            new_alpha = torch.logsumexp(scores, dim=0) + F.log_softmax(u, dim=-1)
            alpha.append(new_alpha)

        logZ = torch.logsumexp(alpha[-1], dim=0)

        # gold score
        score = torch.tensor(0.0, device=device)
        for t in range(L):
            gt = gold[t]
            score = score + unary_logits[t, gt]
            if t > 0:
                gprev, gcur = gold[t-1], gold[t]
                if gprev not in set(allowed_prev[gcur]):
                    score = score + self.unreachable_penalty
                else:
                    score = score + (H[gprev] @ self.W @ H[gcur])

        return logZ - score

    @torch.no_grad()
    def viterbi_one(
        self,
        unary_logits: torch.Tensor,   # [L,N]
        H: torch.Tensor,              # [N,d]
        allowed_prev: List[List[int]],
        top_r: int = 100,
    ) -> List[int]:
        device = unary_logits.device
        L, N = unary_logits.shape

        cand_ids = []
        for t in range(L):
            r = min(top_r, N)
            _, topi = torch.topk(unary_logits[t], k=r)
            c = torch.tensor(sorted(set(topi.tolist())), dtype=torch.long, device=device)
            cand_ids.append(c)

        dp = []
        back = []
        dp0 = unary_logits[0][cand_ids[0]]
        dp.append(dp0)
        back.append(torch.full_like(dp0, -1, dtype=torch.long))

        for t in range(1, L):
            prev = cand_ids[t-1]
            cur = cand_ids[t]
            u = unary_logits[t][cur]
            pair = self.pairwise_scores(H, prev, cur)

            prev_list = prev.tolist()
            mask = torch.zeros(pair.shape, dtype=torch.bool, device=device)
            for j, cur_label in enumerate(cur.tolist()):
                allowed = set(allowed_prev[cur_label])
                for i, p_label in enumerate(prev_list):
                    if p_label in allowed:
                        mask[i,j] = True
            pair = pair + (~mask).float() * self.unreachable_penalty

            prev_dp = dp[-1].unsqueeze(1) + pair
            best_val, best_idx = torch.max(prev_dp, dim=0)
            dp_t = best_val + u
            dp.append(dp_t)
            back.append(best_idx)

        last_pos = int(torch.argmax(dp[-1]).item())
        path=[]
        for t in reversed(range(L)):
            cur_cands = cand_ids[t]
            path.append(int(cur_cands[last_pos].item()))
            if t > 0:
                last_pos = int(back[t][last_pos].item())
        path.reverse()
        return path

from __future__ import annotations
from typing import List, Tuple, Dict
import torch

def build_adj_list(num_nodes: int, edge_index: torch.Tensor) -> List[List[int]]:
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    adj = [[] for _ in range(num_nodes)]
    for u,v in zip(src,dst):
        adj[u].append(v)
    # stable unique
    return [list(dict.fromkeys(a)) for a in adj]

def k_hop_neighbors(adj_list: List[List[int]], k: int) -> List[List[int]]:
    n = len(adj_list)
    out: List[List[int]] = []
    for s in range(n):
        seen = {s}
        frontier = {s}
        for _ in range(k):
            nxt = set()
            for u in frontier:
                for v in adj_list[u]:
                    if v not in seen:
                        seen.add(v)
                        nxt.add(v)
            frontier = nxt
            if not frontier:
                break
        out.append(sorted(seen))
    return out

def path_feasibility_rate(paths: List[List[int]], adj_list: List[List[int]], k_hop: int = 1) -> float:
    # fraction of transitions that are feasible within k hops
    if not paths:
        return 0.0
    allowed_prev = k_hop_neighbors(adj_list, k=k_hop)
    tot = 0
    ok = 0
    for p in paths:
        for a,b in zip(p[:-1], p[1:]):
            tot += 1
            if a in set(allowed_prev[b]):
                ok += 1
    return ok / max(tot, 1)

def token_seq_accuracy(preds: List[List[int]], golds: List[List[int]]) -> Tuple[float,float]:
    tot_tok = 0
    cor_tok = 0
    cor_seq = 0
    for p,g in zip(preds, golds):
        assert len(p) == len(g)
        c = sum(int(a==b) for a,b in zip(p,g))
        cor_tok += c
        tot_tok += len(g)
        cor_seq += int(c == len(g))
    return cor_tok / max(tot_tok,1), cor_seq / max(len(golds),1)

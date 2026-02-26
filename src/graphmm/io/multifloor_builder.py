from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import torch
from graphmm.io.mat_reader import read_mat


@dataclass
class GraphBatch:
    num_nodes: int
    edge_index: torch.Tensor  # [2,E]
    edge_attr: torch.Tensor  # [E,5]
    node_num_feat: torch.Tensor  # [N,5]
    floor_id: torch.Tensor  # [N]
    edge_is_vertical: torch.Tensor  # [E]
    img_hw: Tuple[int, int]  # (row,col)
    coord_xy: torch.Tensor


def _coord_key(x: float, y: float, eps: float) -> Tuple[int, int]:
    return (int(round(x / eps)), int(round(y / eps)))


def build_multifloor_graph_with_features(
        mat_paths: List[str],
        ppm: float = 1.0,
        directed_road: bool = True,
        add_vertical_bidirectional: bool = True,
        coord_match_eps: float = 1.0,
        device: str = "cpu",
) -> GraphBatch:
    per_floor = []
    img_hw = None
    for p in mat_paths:
        fm = read_mat(p, ppm)
        if img_hw is None:
            img_hw = (fm.row, fm.col)
        per_floor.append({"coord": fm.coord_xy, "E": fm.E})

    Ns = [pf["coord"].shape[0] for pf in per_floor]
    offsets = np.cumsum([0] + Ns[:-1]).tolist()
    N_total = int(sum(Ns))
    row, col = img_hw

    coord_xy = np.zeros((N_total, 2), dtype=np.float32)
    floor_id = np.zeros((N_total,), dtype=np.int64)
    for f, pf in enumerate(per_floor):
        off = offsets[f]
        n = pf["coord"].shape[0]
        coord_xy[off:off + n] = pf["coord"]
        floor_id[off:off + n] = f

    edges = []
    raw_attrs = []  # (t,ud,r)
    is_vert = []

    def add_edge(u: int, v: int, t: float, ud: float, r: float, vertical: bool):
        edges.append((u, v))
        raw_attrs.append((t, ud, r))
        is_vert.append(vertical)

    # intra-floor
    for f, pf in enumerate(per_floor):
        off = offsets[f]
        n = pf["coord"].shape[0]
        E = pf["E"]
        for e in E:
            u_local = int(e[0]);
            v_local = int(e[1])
            if not (1 <= u_local <= n and 1 <= v_local <= n):
                continue
            if u_local == v_local:
                continue
            u = off + (u_local - 1)
            v = off + (v_local - 1)
            t = float(e[2]) if e.shape[0] >= 3 else 0.0
            r = float(e[4]) if e.shape[0] >= 5 else 0.0
            ud = float(e[5]) if e.shape[0] >= 6 else 0.0
            add_edge(u, v, t=t, ud=ud, r=r, vertical=False)
            if not directed_road:
                add_edge(v, u, t=t, ud=-ud, r=r, vertical=False)

    # vertical by coord match
    for f in range(len(per_floor) - 1):
        off_a, off_b = offsets[f], offsets[f + 1]
        Acoord = per_floor[f]["coord"]
        Bcoord = per_floor[f + 1]["coord"]

        bucket: Dict[Tuple[int, int], List[int]] = {}
        for j in range(Bcoord.shape[0]):
            x, y = float(Bcoord[j, 0]), float(Bcoord[j, 1])
            bucket.setdefault(_coord_key(x, y, coord_match_eps), []).append(off_b + j)

        for i in range(Acoord.shape[0]):
            x, y = float(Acoord[i, 0]), float(Acoord[i, 1])
            k = _coord_key(x, y, coord_match_eps)
            if k not in bucket:
                continue
            u = off_a + i
            for v in bucket[k]:
                add_edge(u, v, t=10.0, ud=0.0, r=0.0, vertical=True)
                if add_vertical_bidirectional:
                    add_edge(v, u, t=10.0, ud=0.0, r=0.0, vertical=True)

    # dedup
    seen = set()
    ded_edges = [];
    ded_raw = [];
    ded_vert = []
    for (u, v), a, vv in zip(edges, raw_attrs, is_vert):
        if (u, v) in seen:
            continue
        seen.add((u, v))
        ded_edges.append((u, v));
        ded_raw.append(a);
        ded_vert.append(vv)
    edges, raw_attrs, is_vert = ded_edges, ded_raw, ded_vert

    src = np.array([u for u, _ in edges], dtype=np.int64)
    dst = np.array([v for _, v in edges], dtype=np.int64)
    dx = coord_xy[dst, 0] - coord_xy[src, 0]
    dy = coord_xy[dst, 1] - coord_xy[src, 1]
    length = np.sqrt(dx * dx + dy * dy).astype(np.float32)
    len_mean = float(length.mean()) if length.size else 1.0
    length_n = length / max(len_mean, 1e-6)

    t_raw = np.array([t for (t, ud, r) in raw_attrs], dtype=np.float32)
    ud_raw = np.array([ud for (t, ud, r) in raw_attrs], dtype=np.float32)
    r_raw = np.array([r for (t, ud, r) in raw_attrs], dtype=np.float32)
    nz = r_raw[r_raw > 0]
    r_mean = float(nz.mean()) if nz.size else 1.0
    r_n = r_raw / max(r_mean, 1e-6)

    is_vertical = np.array(is_vert, dtype=np.float32)
    edge_attr = np.stack([length_n, is_vertical, t_raw, ud_raw, r_n], axis=1).astype(np.float32)

    x_norm = coord_xy[:, 0] / max(col - 1, 1)
    y_norm = coord_xy[:, 1] / max(row - 1, 1)

    deg_out = np.zeros((N_total,), dtype=np.float32)
    deg_in = np.zeros((N_total,), dtype=np.float32)
    for u, v in edges:
        deg_out[u] += 1.0
        deg_in[v] += 1.0
    deg_out = np.log1p(deg_out)
    deg_in = np.log1p(deg_in)

    is_conn = np.zeros((N_total,), dtype=np.float32)
    for (u, v), vv in zip(edges, is_vert):
        if vv:
            is_conn[u] = 1.0
            is_conn[v] = 1.0

    node_num_feat = np.stack([x_norm, y_norm, deg_in, deg_out, is_conn], axis=1).astype(np.float32)

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long, device=device)
    return GraphBatch(
        num_nodes=N_total,
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32, device=device),
        node_num_feat=torch.tensor(node_num_feat, dtype=torch.float32, device=device),
        floor_id=torch.tensor(floor_id, dtype=torch.long, device=device),
        edge_is_vertical=torch.tensor(np.array(is_vert, dtype=np.bool_), device=device),
        img_hw=(row, col),
        coord_xy=torch.tensor(coord_xy, dtype=torch.float32, device=device)
    )

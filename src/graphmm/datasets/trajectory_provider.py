from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import random
import torch
from collections import Counter
from scipy.io import loadmat
import numpy as np
import os, glob

PADDING_ID = -1

@dataclass
class Sample:
    pred: List[int]
    true: List[int]

def pad_batch(seqs: List[List[int]], pad_id: int = PADDING_ID):
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    L = int(lengths.max().item()) if len(seqs) else 0
    out = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    for i,s in enumerate(seqs):
        out[i,:len(s)] = torch.tensor(s, dtype=torch.long)
    return out, lengths

def build_adj_list_from_edges(num_nodes: int, edges: List[Tuple[int,int]]):
    adj = [[] for _ in range(num_nodes)]
    for u,v in edges:
        adj[u].append(v)
    return [list(dict.fromkeys(a)) for a in adj]

def sim_samples_on_graph(
    adj_list: List[List[int]],
    num_samples: int,
    seq_len: int,
    noise_rate: float,
    prefer_intrafloor: bool,
    floor_id: torch.Tensor,
) -> List[Sample]:
    n = len(adj_list)
    valid = [i for i in range(n) if len(adj_list[i]) > 0]
    if not valid:
        raise ValueError("No valid starts in the graph.")
    floor = floor_id.cpu().tolist()

    def step(cur: int) -> int:
        neis = adj_list[cur]
        if not neis:
            return random.choice(valid)
        if not prefer_intrafloor:
            return random.choice(neis)
        same = [v for v in neis if floor[v] == floor[cur]]
        if same and random.random() < 0.85:
            return random.choice(same)
        return random.choice(neis)

    def walk(start: int, L: int) -> List[int]:
        path=[start]; cur=start
        for _ in range(L-1):
            cur = step(cur)
            path.append(cur)
        return path

    def corrupt(seq: List[int]) -> List[int]:
        out = seq[:]
        for t in range(len(out)):
            if random.random() < noise_rate:
                cur = out[t]
                neis = adj_list[cur]
                if neis and random.random() < 0.7:
                    out[t] = random.choice(neis)
                else:
                    out[t] = random.randrange(n)
        return out

    samples=[]
    for _ in range(num_samples):
        s = random.choice(valid)
        true = walk(s, seq_len)
        pred = corrupt(true)
        samples.append(Sample(pred=pred, true=true))
    return samples

def build_transition_graph(sequences: List[List[int]], directed: bool = True, min_count: int = 1):
    cnt = Counter()
    for seq in sequences:
        for a,b in zip(seq[:-1], seq[1:]):
            cnt[(a,b)] += 1
            if not directed:
                cnt[(b,a)] += 1
    edges = [(u,v,c) for (u,v),c in cnt.items() if c >= min_count and u != v]
    return edges

def _mat_keys(d):
    return [k for k in d.keys() if not k.startswith("__")]

def load_pts_coord_any(mat_path: str):
    """
    读取 mat 内的 pts_coord，支持两类情况：
    A) d["pts_coord"] 直接存在
    B) pts_coord 在某个 struct 里：d[root][0,0]["pts_coord"][0,0]...
    如果你实际字段更深，后续我可以按你的 keys 精确改，但这个版本先尽量鲁棒。
    """
    d = loadmat(mat_path)
    if "pts_coord" in d:
        pts = d["pts_coord"]
    else:
        # 尝试从第一个非 __ 的 key 里找
        roots = _mat_keys(d)
        found = None
        for r in roots:
            obj = d[r]
            # struct 常见：obj is ndarray shape (1,1) with dtype.names
            if hasattr(obj, "dtype") and obj.dtype.names and ("pts_coord" in obj.dtype.names):
                found = obj[0,0]["pts_coord"]
                break
        if found is None:
            raise KeyError(f"Cannot find pts_coord in {mat_path}. keys={roots}")
        pts = found

    pts = np.array(pts, dtype=float)
    # 期望 shape: (3, T) 或 (T,3)；统一转成 (3,T)
    if pts.shape[0] != 3 and pts.shape[1] == 3:
        pts = pts.T
    if pts.shape[0] != 3:
        raise ValueError(f"pts_coord shape unexpected: {pts.shape} in {mat_path}")

    x = pts[0, :].astype(float)
    y = pts[1, :].astype(float)
    f = pts[2, :].astype(int)
    return x, y, f

def coords_to_global_node_seq(x, y, f, gb, floor_base: int = 1, xy_mode: str = "xy"):
    """
    (x,y,floor) -> global node id sequence
    floor_base=1: 轨迹楼层为 1..5
    floor_base=0: 轨迹楼层为 0..4
    """
    coord = gb.coord_xy.detach().cpu().numpy()       # [N,2]
    floor_id = gb.floor_id.detach().cpu().numpy()    # [N]

    x_use, y_use = x, y
    if xy_mode not in {"xy", "yx", "auto"}:
        raise ValueError(f"xy_mode must be one of ['xy','yx','auto'], got: {xy_mode}")

    if xy_mode == "yx":
        x_use, y_use = y, x
    elif xy_mode == "auto":
        # Decide orientation by nearest-node distance on a small prefix.
        T = min(30, len(x))
        if T > 0:
            d_xy = []
            d_yx = []
            for xi, yi, fi in zip(x[:T], y[:T], f[:T]):
                fi0 = int(fi) - floor_base
                idx = np.where(floor_id == fi0)[0]
                if idx.size == 0:
                    continue
                sub = coord[idx]
                d2_xy = (sub[:, 0] - xi) ** 2 + (sub[:, 1] - yi) ** 2
                d2_yx = (sub[:, 0] - yi) ** 2 + (sub[:, 1] - xi) ** 2
                d_xy.append(float(np.sqrt(d2_xy.min())))
                d_yx.append(float(np.sqrt(d2_yx.min())))
            if d_xy and d_yx and (np.mean(d_yx) < np.mean(d_xy)):
                x_use, y_use = y, x

    seq = []

    for xi, yi, fi in zip(x_use, y_use, f):
        fi0 = int(fi) - floor_base
        idx = np.where(floor_id == fi0)[0]
        if idx.size == 0:
            continue
        sub = coord[idx]
        d2 = (sub[:,0] - xi)**2 + (sub[:,1] - yi)**2
        j = int(idx[int(np.argmin(d2))])  # global id
        # 去掉连续重复（处理跨层重叠点/抖动）
        if not seq or seq[-1] != j:
            seq.append(j)

    return seq

def list_all_mats(root_dir: str):
    return sorted(glob.glob(os.path.join(root_dir, "**", "*.mat"), recursive=True))

def pair_neg_gt(neg_files):
    """
    neg 文件 -> (neg_path, gt_path)
    规则：把文件名里的 _neg_ 替换为 _gt_（或 neg->gt），并在同目录/同根下找。
    """
    pairs = []
    for neg in neg_files:
        gt = neg
        if "_neg_" in gt:
            gt = gt.replace("_neg_", "_gt_")
        elif "neg" in gt:
            gt = gt.replace("neg", "gt")
        if os.path.exists(gt):
            pairs.append((neg, gt))
        else:
            # 也允许 gt 在同一个 root 下别处：按文件名查找
            fname = os.path.basename(gt)
            cand = glob.glob(os.path.join(os.path.dirname(os.path.dirname(neg)), "**", fname), recursive=True)
            if cand:
                pairs.append((neg, cand[0]))
            else:
                raise FileNotFoundError(f"GT not found for NEG: {neg} -> {gt}")
    return pairs

def load_paired_samples_from_runs(
    traj_root: str,
    gb,
    floor_base: int = 1,
    xy_mode: str = "xy",
):
    """
    从 traj_root 递归读取所有 *_neg_*.mat，并找到对应 *_gt_*.mat，返回 List[Sample(pred,true)]
    """
    all_files = list_all_mats(traj_root)
    neg_files = [p for p in all_files if ("_neg_" in os.path.basename(p) or "neg" in os.path.basename(p))]
    pairs = pair_neg_gt(neg_files)

    samples = []
    for neg_path, gt_path in pairs:
        x1,y1,f1 = load_pts_coord_any(neg_path)
        x2,y2,f2 = load_pts_coord_any(gt_path)

        pred_seq = coords_to_global_node_seq(x1, y1, f1, gb, floor_base=floor_base, xy_mode=xy_mode)
        true_seq = coords_to_global_node_seq(x2, y2, f2, gb, floor_base=floor_base, xy_mode=xy_mode)

        # 长度保护
        L = min(len(pred_seq), len(true_seq))
        if L < 3:
            continue
        pred_seq = pred_seq[:L]
        true_seq = true_seq[:L]
        samples.append(Sample(pred=pred_seq, true=true_seq))

    return samples

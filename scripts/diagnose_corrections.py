import argparse
import csv
import pickle
import glob
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graphmm.datasets.trajectory_provider import build_transition_graph, coords_to_global_node_seq, load_pts_coord_any
from graphmm.io.multifloor_builder import build_multifloor_graph_with_features
from graphmm.models.graphmm_corrector import GraphMMCorrector, confidence_gate_sequence, decode_argmax
from graphmm.utils.graph import build_adj_list, k_hop_neighbors


def _ensure_str(x, name: str) -> str:
    if isinstance(x, list):
        if not x:
            raise ValueError(f"{name} list is empty")
        return str(x[0])
    return str(x)




def _load_model_state_dict(ckpt_path: str, device: str) -> Mapping[str, Any]:
    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {ckpt_file}. Please pass a valid .pt/.pth file via --ckpt "
            f"or cfg.test.ckpt_path."
        )
    if ckpt_file.is_dir():
        raise IsADirectoryError(f"checkpoint path is a directory, expected file: {ckpt_file}")

    try:
        state = torch.load(str(ckpt_file), map_location=device)
    except (RuntimeError, pickle.UnpicklingError) as exc:
        msg = str(exc).lower()
        if "weights only load failed" in msg:
            try:
                state = torch.load(str(ckpt_file), map_location=device, weights_only=False)
            except Exception as exc2:
                raise RuntimeError(
                    "Failed to load checkpoint with both weights_only=True/False. "
                    "The file may be corrupted or not a checkpoint.\n"
                    f"  path: {ckpt_file}"
                ) from exc2
        elif "pytorchstreamreader failed reading zip archive" in msg:
            raise RuntimeError(
                "Failed to load checkpoint: file is not a valid PyTorch checkpoint archive, "
                "or is corrupted.\n"
                f"  path: {ckpt_file}\n"
                "Please verify you selected a model checkpoint file (e.g. runs/<run_name>/checkpoint.pt), "
                "not another asset such as .mat/.zip/.yaml."
            ) from exc
        else:
            raise

    if isinstance(state, Mapping):
        if "model" in state and isinstance(state["model"], Mapping):
            return state["model"]
        # allow raw state_dict checkpoints
        sample_keys = list(state.keys())
        if sample_keys and all(isinstance(k, str) for k in sample_keys):
            if any(k.startswith(("node_encoder.", "gine.", "encoder.", "decoder.", "crf.")) for k in sample_keys):
                return state

    raise ValueError(
        f"Unsupported checkpoint format at {ckpt_file}. Expected a dict with key 'model' or a raw model state_dict."
    )

def pair_neg_gt(files: Iterable[str]) -> List[Tuple[str, str]]:
    negs = [p for p in files if ("_neg_" in os.path.basename(p) or "neg" in os.path.basename(p))]
    pairs = []
    for neg in negs:
        gt = neg.replace("_neg_", "_gt_") if "_neg_" in neg else neg.replace("neg", "gt")
        if os.path.exists(gt):
            pairs.append((neg, gt))
    return pairs


def build_transition_edges(seqs: Sequence[Sequence[int]], device: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    ecount = build_transition_graph(seqs, directed=True, min_count=1)
    if not ecount:
        return None, None
    src = torch.tensor([u for (u, _, _) in ecount], dtype=torch.long, device=device)
    dst = torch.tensor([v for (_, v, _) in ecount], dtype=torch.long, device=device)
    w = torch.tensor([float(c) for (_, _, c) in ecount], dtype=torch.float32, device=device)
    w = w / w.mean().clamp(min=1e-6)
    return torch.stack([src, dst], dim=0), w


def shortest_hop_distance(adj: List[List[int]], src: int, dst: int, max_hops: int) -> int:
    if src == dst:
        return 0
    if max_hops <= 0:
        return -1
    q = deque([(src, 0)])
    seen = {src}
    while q:
        cur, d = q.popleft()
        if d >= max_hops:
            continue
        for nxt in adj[cur]:
            if nxt == dst:
                return d + 1
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, d + 1))
    return -1




def junction_proximity_mask(adj: List[List[int]], junction_nodes: set, max_hops: int) -> List[int]:
    n = len(adj)
    if not junction_nodes:
        return [0] * n
    dist = [-1] * n
    q = deque()
    for j in junction_nodes:
        if 0 <= j < n and dist[j] == -1:
            dist[j] = 0
            q.append(j)
    while q:
        cur = q.popleft()
        if dist[cur] >= max_hops:
            continue
        for nxt in adj[cur]:
            if dist[nxt] == -1:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return [int(0 <= d <= max_hops) for d in dist]

def illegal_rate(seq: Sequence[int], allowed_prev: List[set]) -> float:
    if len(seq) <= 1:
        return 0.0
    bad = 0
    for t in range(1, len(seq)):
        if seq[t - 1] not in allowed_prev[seq[t]]:
            bad += 1
    return bad / max(1, len(seq) - 1)


def summarize(rows: List[Dict], allowed_prev: List[set], print_prefix: str = "") -> None:
    n = len(rows)
    if n == 0:
        print(f"{print_prefix}[diag] no rows generated.")
        return

    raw_correct = sum(int(r["is_correct_raw"]) for r in rows)
    raw_wrong = n - raw_correct
    keep = sum(int(r["is_correct_raw"] and r["is_correct_final"]) for r in rows)
    over = sum(int(r["is_correct_raw"] and not r["is_correct_final"]) for r in rows)
    fix = sum(int((not r["is_correct_raw"]) and r["is_correct_final"]) for r in rows)

    r_keep = keep / raw_correct if raw_correct > 0 else 0.0
    r_over = over / raw_correct if raw_correct > 0 else 0.0
    r_fix = fix / raw_wrong if raw_wrong > 0 else 0.0

    argmax_correct = sum(int(r["is_correct_argmax"]) for r in rows)
    decode_correct = sum(int(r["is_correct_decode"]) for r in rows)

    reject_correct_numer = sum(int(r["is_correct_decode"] and not r["is_correct_final"]) for r in rows)
    reject_correct_denom = decode_correct
    reject_correct = reject_correct_numer / reject_correct_denom if reject_correct_denom > 0 else 0.0

    block_wrong_numer = sum(int((not r["is_correct_decode"]) and r["gate_reverted_to_raw"]) for r in rows)
    block_wrong_denom = n - decode_correct
    block_wrong = block_wrong_numer / block_wrong_denom if block_wrong_denom > 0 else 0.0

    crf_gain = sum(int(r["is_correct_decode"]) - int(r["is_correct_argmax"]) for r in rows) / n

    delta_err = [r["gain_gtx"] for r in rows if not r["is_correct_raw"]]
    delta_all = [r["gain_gtx"] for r in rows]

    print(f"{print_prefix}[diag] tokens={n} raw_correct={raw_correct} raw_wrong={raw_wrong}")
    print(f"{print_prefix}[diag] R_keep={r_keep:.4f} R_fix={r_fix:.4f} R_over={r_over:.4f}")
    print(f"{print_prefix}[diag] acc_argmax={argmax_correct / n:.4f} acc_decode={decode_correct / n:.4f} acc_final={sum(int(r['is_correct_final']) for r in rows) / n:.4f}")
    print(f"{print_prefix}[diag] R_reject_correct={reject_correct:.4f} R_block_wrong={block_wrong:.4f} crf_gain={crf_gain:.4f}")
    if delta_all:
        print(f"{print_prefix}[diag] gain_gtx mean(all)={sum(delta_all) / len(delta_all):.4f} mean(raw_wrong)={(sum(delta_err) / len(delta_err)) if delta_err else 0.0:.4f}")

    for name in ["raw", "argmax", "decode", "final"]:
        seqs = {}
        for r in rows:
            sid = r["sample_id"]
            seqs.setdefault(sid, []).append(r[f"y_{name}"])
        il = [illegal_rate(s, allowed_prev) for s in seqs.values() if len(s) > 1]
        avg = sum(il) / len(il) if il else 0.0
        print(f"{print_prefix}[diag] illegal_rate_{name}={avg:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Token-level diagnostics for GraphMM correction pipeline.")
    ap.add_argument("--config", default=str(ROOT / "configs/mall_train.yaml"))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--test_dir", default=None)
    ap.add_argument("--mat_paths", nargs="*", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--floor_base", type=int, default=None)
    ap.add_argument("--traj_xy_mode", default=None, choices=["xy", "yx", "auto"])
    ap.add_argument("--output_csv", default=str(ROOT / "runs/diagnostics/token_diagnostics.csv"))
    ap.add_argument("--max_hops", type=int, default=20, help="max hop distance for topo_dist_* metrics")
    ap.add_argument("--junction_degree", type=int, default=3, help="node degree >= this value is considered a junction")
    ap.add_argument("--junction_hops", type=int, default=1, help="distance to junction threshold in hops")
    ap.add_argument("--disable_gate", action="store_true")
    ap.add_argument("--force_argmax_decode", action="store_true", help="disable CRF decode even when available")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    mat_paths = args.mat_paths if args.mat_paths else cfg.get("data", {}).get("mat_paths", [])
    if isinstance(mat_paths, str):
        mat_paths = [mat_paths]
    if not mat_paths:
        raise ValueError("mat_paths is empty. Fill cfg.data.mat_paths or pass --mat_paths ...")

    ckpt = _ensure_str(args.ckpt or cfg.get("test", {}).get("ckpt_path", None), "ckpt_path")
    test_dir = _ensure_str(args.test_dir or cfg.get("test", {}).get("test_dir", "data/traj/test"), "test_dir")
    floor_base = args.floor_base if args.floor_base is not None else cfg.get("test", {}).get("floor_base", 1)
    traj_xy_mode = args.traj_xy_mode or cfg.get("data", {}).get("traj_xy_mode", "auto")

    min_conf = float(cfg.get("model", {}).get("min_correction_confidence", 0.0))
    min_gain = float(cfg.get("model", {}).get("min_correction_logit_gain", 0.0))
    use_crf_cfg = bool(cfg.get("model", {}).get("use_crf", False))
    crf_train_loss = str(cfg.get("train", {}).get("crf_train_loss", "ce")).lower()

    gb = build_multifloor_graph_with_features(
        mat_paths=mat_paths,
        ppm=cfg["data"]["ppm"],
        directed_road=cfg["data"]["directed_road"],
        add_vertical_bidirectional=cfg["data"]["add_vertical_bidirectional"],
        coord_match_eps=cfg["data"]["coord_match_eps"],
        device=device,
    )

    num_floors = int(gb.floor_id.max().item()) + 1
    model = GraphMMCorrector(
        num_nodes=gb.num_nodes,
        num_floors=num_floors,
        road_dim=cfg["model"]["road_dim"],
        gin_layers=cfg["model"]["gin_layers"],
        gin_mlp_hidden=cfg["model"]["gin_mlp_hidden"],
        dropout=cfg["model"]["dropout"],
        temperature=cfg["model"]["temperature"],
        floor_emb_dim=cfg["model"]["floor_emb_dim"],
        traj_gcn_layers=cfg["model"]["traj_gcn_layers"],
        use_crf=use_crf_cfg,
        unreachable_penalty=cfg["model"]["unreachable_penalty"],
        input_anchor_bias=cfg["model"].get("input_anchor_bias", 0.0),
        apply_input_anchor_bias_inference=cfg["model"].get("apply_input_anchor_bias_inference", False),
        apply_input_anchor_bias_training=cfg["model"].get("apply_input_anchor_bias_training", True),
        inference_use_input_context=cfg["model"].get("inference_use_input_context", True),
    ).to(device)

    model_state = _load_model_state_dict(ckpt, device=device)
    model.load_state_dict(model_state, strict=True)
    model.eval()

    road_adj = build_adj_list(gb.num_nodes, gb.edge_index)
    node_degree = [len(nbrs) for nbrs in road_adj]
    junction_nodes = {i for i, d in enumerate(node_degree) if d >= args.junction_degree}
    near_junction_mask = junction_proximity_mask(road_adj, junction_nodes, args.junction_hops)

    k_hop = int(cfg["train"]["k_hop"])
    allowed_prev = k_hop_neighbors(road_adj, k=k_hop)
    use_crf_decode = bool(use_crf_cfg and crf_train_loss == "crf" and not args.force_argmax_decode)

    all_files = glob.glob(os.path.join(test_dir, "**", "*.mat"), recursive=True)
    pairs = pair_neg_gt(all_files)
    if not pairs:
        raise RuntimeError(f"No neg-gt pairs found in {test_dir}")

    rows: List[Dict] = []

    with torch.no_grad():
        for sid, (neg_path, gt_path) in enumerate(pairs):
            x1, y1, f1 = load_pts_coord_any(neg_path)
            x2, y2, f2 = load_pts_coord_any(gt_path)
            raw_seq = coords_to_global_node_seq(x1, y1, f1, gb, floor_base=floor_base, xy_mode=traj_xy_mode)
            gt_seq = coords_to_global_node_seq(x2, y2, f2, gb, floor_base=floor_base, xy_mode=traj_xy_mode)
            L = min(len(raw_seq), len(gt_seq))
            if L < 2:
                continue
            raw_seq = raw_seq[:L]
            gt_seq = gt_seq[:L]

            pred = torch.tensor([raw_seq], dtype=torch.long, device=device)
            lengths = torch.tensor([L], dtype=torch.long, device=device)
            traj_edge_index, traj_edge_weight = build_transition_edges([raw_seq], device=device)

            unary_logits, h_road = model.forward_unary(
                pred,
                lengths,
                node_num_feat=gb.node_num_feat,
                floor_id=gb.floor_id,
                edge_index=gb.edge_index,
                edge_attr=gb.edge_attr,
                traj_edge_index=traj_edge_index,
                traj_edge_weight=traj_edge_weight,
                teacher_forcing=None,
            )
            unary = unary_logits[0, :L, :]

            argmax_seq = decode_argmax(unary_logits, lengths)[0]
            if use_crf_decode:
                decode_seq = model.crf.viterbi_one(
                    unary_logits=unary,
                    H=h_road,
                    allowed_prev=allowed_prev,
                    top_r=int(cfg["train"]["top_r_decode"]),
                )
            else:
                decode_seq = argmax_seq

            gated_seq = confidence_gate_sequence(
                raw_seq=raw_seq,
                corrected_seq=decode_seq,
                unary_logits=unary,
                min_confidence=min_conf,
                min_logit_gain=min_gain,
            )
            final_seq = decode_seq if args.disable_gate else gated_seq

            probs = torch.softmax(unary, dim=-1)
            conf = probs.max(dim=-1).values

            for t in range(L):
                x_t = int(raw_seq[t])
                y_gt = int(gt_seq[t])
                y_argmax = int(argmax_seq[t])
                y_decode = int(decode_seq[t])
                y_final = int(final_seq[t])

                topo_raw_gt = shortest_hop_distance(road_adj, x_t, y_gt, args.max_hops)
                topo_pred_gt = shortest_hop_distance(road_adj, y_decode, y_gt, args.max_hops)
                near_junction = int(near_junction_mask[x_t])

                row = {
                    "sample_id": sid,
                    "step": t,
                    "neg_file": os.path.basename(neg_path),
                    "gt_file": os.path.basename(gt_path),
                    "x_t": x_t,
                    "y_gt": y_gt,
                    "y_argmax": y_argmax,
                    "y_decode": y_decode,
                    "y_final": y_final,
                    "y_raw": x_t,
                    "u_gt": float(unary[t, y_gt].item()),
                    "u_x": float(unary[t, x_t].item()),
                    "u_pred": float(unary[t, y_decode].item()),
                    "gain_gtx": float((unary[t, y_gt] - unary[t, x_t]).item()),
                    "gain_predx": float((unary[t, y_decode] - unary[t, x_t]).item()),
                    "conf": float(conf[t].item()),
                    "is_correct_raw": int(x_t == y_gt),
                    "is_correct_argmax": int(y_argmax == y_gt),
                    "is_correct_decode": int(y_decode == y_gt),
                    "is_correct_final": int(y_final == y_gt),
                    "gate_reverted_to_raw": int(y_final == x_t and y_decode != x_t),
                    "topo_dist_raw_gt": topo_raw_gt,
                    "topo_dist_pred_gt": topo_pred_gt,
                    "near_junction": near_junction,
                }
                rows.append(row)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id", "step", "neg_file", "gt_file", "x_t", "y_gt", "y_argmax", "y_decode", "y_final", "y_raw",
        "u_gt", "u_x", "u_pred", "gain_gtx", "gain_predx", "conf",
        "is_correct_raw", "is_correct_argmax", "is_correct_decode", "is_correct_final", "gate_reverted_to_raw",
        "topo_dist_raw_gt", "topo_dist_pred_gt", "near_junction",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[diag] wrote token diagnostics: {output_path}")
    summarize(rows, allowed_prev)


if __name__ == "__main__":
    main()

# scripts/test.py
import argparse
import os
import sys
import glob
from pathlib import Path

import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graphmm.io.multifloor_builder import build_multifloor_graph_with_features
from graphmm.models.graphmm_corrector import GraphMMCorrector, decode_argmax
from graphmm.utils.graph import (
    build_adj_list,
    k_hop_neighbors,
    token_seq_accuracy,
    path_feasibility_rate,
)
from graphmm.datasets.trajectory_provider import load_pts_coord_any, coords_to_global_node_seq


def _ensure_str(x, name: str) -> str:
    """Allow yaml scalar or yaml list; return first element if list."""
    if isinstance(x, list):
        if len(x) == 0:
            raise ValueError(f"{name} list is empty")
        return str(x[0])
    return str(x)


def pair_neg_gt(files):
    """
    Find neg files and pair with corresponding gt by filename substitution.
    Rules:
      xxx_neg_yyy.mat <-> xxx_gt_yyy.mat
      or replace 'neg' -> 'gt'
    """
    negs = [p for p in files if ("_neg_" in os.path.basename(p) or "neg" in os.path.basename(p))]
    pairs = []
    for neg in negs:
        gt = neg.replace("_neg_", "_gt_") if "_neg_" in neg else neg.replace("neg", "gt")
        if os.path.exists(gt):
            pairs.append((neg, gt))
    return pairs


def ids_to_xyzf(seq, gb, floor_base_out: int = 1):
    """
    Convert global node ids -> (id, x, y, floor)
    floor_base_out=1 means output floor in 1..5, floor_base_out=0 means 0..4
    """
    if not hasattr(gb, "coord_xy") or gb.coord_xy is None:
        raise RuntimeError("GraphBatch has no coord_xy. Please add coord_xy to GraphBatch in multifloor_builder.py")

    coord = gb.coord_xy.detach().cpu().numpy()   # [N,2]
    floor = gb.floor_id.detach().cpu().numpy()   # [N]
    out = []
    for nid in seq:
        x, y = coord[int(nid)]
        f = int(floor[int(nid)]) + floor_base_out
        out.append((int(nid), float(x), float(y), int(f)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/mall_train.yaml"))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--test_dir", default=None)
    ap.add_argument("--mat_paths", nargs="*", default=None)
    ap.add_argument("--floor_base", type=int, default=None, help="trajectory floor base: 1 for 1..5, 0 for 0..4")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max_print", type=int, default=30, help="max points to print per sequence")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # mat paths
    mat_paths = args.mat_paths if args.mat_paths else cfg.get("data", {}).get("mat_paths", [])
    if isinstance(mat_paths, str):
        mat_paths = [mat_paths]
    if not mat_paths:
        raise ValueError("mat_paths is empty. Fill cfg.data.mat_paths or pass --mat_paths ...")

    # ckpt / test_dir / floor_base from config unless overridden
    ckpt = args.ckpt or cfg.get("test", {}).get("ckpt_path", None)
    test_dir = args.test_dir or cfg.get("test", {}).get("test_dir", "data/traj/test")
    floor_base = args.floor_base if args.floor_base is not None else cfg.get("test", {}).get("floor_base", 1)

    if ckpt is None:
        raise ValueError("ckpt_path is missing. Fill cfg.test.ckpt_path or pass --ckpt ...")

    ckpt = _ensure_str(ckpt, "ckpt_path")
    test_dir = _ensure_str(test_dir, "test_dir")

    # build graph
    gb = build_multifloor_graph_with_features(
        mat_paths=mat_paths,
        ppm=cfg["data"]["ppm"],
        directed_road=cfg["data"]["directed_road"],
        add_vertical_bidirectional=cfg["data"]["add_vertical_bidirectional"],
        coord_match_eps=cfg["data"]["coord_match_eps"],
        device=device,
    )

    if not hasattr(gb, "coord_xy") or gb.coord_xy is None:
        raise RuntimeError(
            "GraphBatch missing coord_xy. You must add coord_xy to GraphBatch in multifloor_builder.py "
            "to print (x,y,f) and to map (x,y,floor) -> node ids reliably."
        )

    # build model + load checkpoint
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
        use_crf=cfg["model"]["use_crf"],
        unreachable_penalty=cfg["model"]["unreachable_penalty"],
    ).to(device)

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    # CRF allowed_prev
    road_adj_list = build_adj_list(gb.num_nodes, gb.edge_index)
    k_hop = int(cfg["train"]["k_hop"])
    allowed_prev = k_hop_neighbors(road_adj_list, k=k_hop)

    # load test pairs
    all_files = glob.glob(os.path.join(test_dir, "**", "*.mat"), recursive=True)
    pairs = pair_neg_gt(all_files)
    if not pairs:
        raise RuntimeError(
            f"No neg-gt pairs found in {test_dir}. "
            f"Make sure you have both *_neg_*.mat and matching *_gt_*.mat."
        )

    preds, golds = [], []

    with torch.no_grad():
        for neg_path, gt_path in pairs:
            # read pts_coord
            x1, y1, f1 = load_pts_coord_any(neg_path)
            x2, y2, f2 = load_pts_coord_any(gt_path)

            # map to node sequences
            pred_seq = coords_to_global_node_seq(x1, y1, f1, gb, floor_base=floor_base)
            true_seq = coords_to_global_node_seq(x2, y2, f2, gb, floor_base=floor_base)

            L = min(len(pred_seq), len(true_seq))
            if L < 3:
                continue

            pred_seq = pred_seq[:L]
            true_seq = true_seq[:L]

            pred = torch.tensor([pred_seq], dtype=torch.long, device=device)
            lengths = torch.tensor([L], dtype=torch.long, device=device)

            unary_logits, H_R = model.forward_unary(
                pred, lengths,
                node_num_feat=gb.node_num_feat,
                floor_id=gb.floor_id,
                edge_index=gb.edge_index,
                edge_attr=gb.edge_attr,
                traj_edge_index=None,
                traj_edge_weight=None,
                teacher_forcing=pred
            )
            unary = unary_logits[0, :L, :]

            if cfg["model"]["use_crf"]:
                corrected = model.crf.viterbi_one(
                    unary_logits=unary,
                    H=H_R,
                    allowed_prev=allowed_prev,
                    top_r=int(cfg["train"]["top_r_decode"])
                )
            else:
                corrected = decode_argmax(unary_logits, lengths)[0]

            preds.append(corrected)
            golds.append(true_seq)

            # ---- print this sample ----
            maxp = int(args.max_print)
            print("\n==============================")
            print("NEG file:", os.path.basename(neg_path))
            print("GT  file:", os.path.basename(gt_path))
            print(f"len(pred_ids)={len(pred_seq)} len(corr_ids)={len(corrected)} len(gt_ids)={len(true_seq)}")

            print("\n[pred_ids]")
            print(pred_seq[:maxp], "..." if len(pred_seq) > maxp else "")

            print("\n[corrected_ids]")
            print(corrected[:maxp], "..." if len(corrected) > maxp else "")

            print("\n[gt_ids]")
            print(true_seq[:maxp], "..." if len(true_seq) > maxp else "")

            # With coordinates (print first maxp points)
            # Output floor in 1..5 if your floors are 1..5
            floor_base_out = 1 if floor_base == 1 else 0
            print("\n[pred (id,x,y,f)]")
            print(ids_to_xyzf(pred_seq[:maxp], gb, floor_base_out=floor_base_out))

            print("\n[corrected (id,x,y,f)]")
            print(ids_to_xyzf(corrected[:maxp], gb, floor_base_out=floor_base_out))

            print("\n[gt (id,x,y,f)]")
            print(ids_to_xyzf(true_seq[:maxp], gb, floor_base_out=floor_base_out))

    tok, seq = token_seq_accuracy(preds, golds)
    feas = path_feasibility_rate(preds, road_adj_list, k_hop=max(1, k_hop))
    print(f"\n[TEST] n={len(preds)} tok={tok:.3f} seq={seq:.3f} feas@k={feas:.3f}")


if __name__ == "__main__":
    main()

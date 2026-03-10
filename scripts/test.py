# scripts/read_mat_test.py
import argparse
import os
import sys
import glob
import re
import pickle
from pathlib import Path
from typing import Any, List, Mapping

import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graphmm.io.multifloor_builder import build_multifloor_graph_with_features
from graphmm.io.checkpoint import load_model_state_dict, infer_traj_gcn_layers
from graphmm.models.graphmm_corrector import GraphMMCorrector, decode_argmax, confidence_gate_sequence
from graphmm.utils.graph import (
    build_adj_list,
    k_hop_neighbors,
    token_seq_accuracy,
    path_feasibility_rate,
)
from graphmm.datasets.trajectory_provider import load_pts_coord_any, coords_to_global_node_seq, build_transition_graph


def _ensure_str(x, name: str) -> str:
    """Allow yaml scalar or yaml list; return first element if list."""
    if isinstance(x, list):
        if len(x) == 0:
            raise ValueError(f"{name} list is empty")
        return str(x[0])
    return str(x)


def _peek_file_prefix(path: Path, n: int = 256) -> bytes:
    with path.open("rb") as f:
        return f.read(n)


def _raise_if_lfs_pointer(path: Path) -> None:
    try:
        head = _peek_file_prefix(path, n=256)
    except OSError:
        return
    if b"git-lfs.github.com/spec/v1" in head:
        raise RuntimeError(
            "Checkpoint file appears to be a Git LFS pointer, not the real model weights.\n"
            f"  path: {path}\n"
            "Please install Git LFS and run: `git lfs pull` (or re-download the actual checkpoint file)."
        )


def _infer_traj_gcn_layers(state_dict: Mapping[str, Any]) -> int:
    return infer_traj_gcn_layers(state_dict)


def _load_model_state_dict(ckpt_path: str, device: str) -> Mapping[str, Any]:
    return load_model_state_dict(ckpt_path, device=device)


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
    ap.add_argument("--traj_xy_mode", default=None, choices=["xy", "yx", "auto"], help="trajectory coordinate order mapping")
    ap.add_argument("--max_print", type=int, default=30, help="max points to print per sequence")
    ap.add_argument("--disable_gate", action="store_true", help="disable confidence gate and report raw decoder output only")
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
    traj_xy_mode = args.traj_xy_mode or cfg.get("data", {}).get("traj_xy_mode", "auto")
    min_correction_confidence = float(cfg.get("model", {}).get("min_correction_confidence", 0.0))
    min_correction_logit_gain = float(cfg.get("model", {}).get("min_correction_logit_gain", 0.0))
    crf_train_loss = str(cfg.get("train", {}).get("crf_train_loss", "ce")).lower()
    use_crf_decode = bool(cfg.get("model", {}).get("use_crf", False) and crf_train_loss == "crf")

    print(
        f"[test] use_crf_cfg={int(cfg.get('model', {}).get('use_crf', False))} crf_train_loss={crf_train_loss} use_crf_decode={int(use_crf_decode)} "
        f"traj_gcn_layers={int(cfg.get('model', {}).get('traj_gcn_layers', 0))} "
        f"inference_use_input_context={int(cfg.get('model', {}).get('inference_use_input_context', True))} "
        f"input_anchor_bias={float(cfg.get('model', {}).get('input_anchor_bias', 0.0)):.4f} "
        f"apply_input_anchor_bias_inference={int(cfg.get('model', {}).get('apply_input_anchor_bias_inference', False))}"
    )

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
    model_state = _load_model_state_dict(ckpt, device=device)
    ckpt_traj_layers = _infer_traj_gcn_layers(model_state)
    cfg_traj_layers = int(cfg["model"].get("traj_gcn_layers", 0))
    if ckpt_traj_layers != cfg_traj_layers:
        print(
            f"[warn] traj_gcn_layers mismatch: cfg={cfg_traj_layers} ckpt={ckpt_traj_layers}; "
            f"using ckpt value for model construction (for traj_gcn ablation, retrain a matching checkpoint)."
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
        traj_gcn_layers=ckpt_traj_layers,
        use_crf=cfg["model"]["use_crf"],
        unreachable_penalty=cfg["model"]["unreachable_penalty"],
        input_anchor_bias=cfg["model"].get("input_anchor_bias", 0.0),
        apply_input_anchor_bias_inference=cfg["model"].get("apply_input_anchor_bias_inference", False),
        apply_input_anchor_bias_training=cfg["model"].get("apply_input_anchor_bias_training", True),
        inference_use_input_context=cfg["model"].get("inference_use_input_context", True),
        training_use_input_context=cfg["model"].get("training_use_input_context", False),
    ).to(device)

    model.load_state_dict(model_state, strict=True)
    model.eval()

    if cfg["model"]["use_crf"] and not use_crf_decode:
        print("[warn] use_crf=true but crf_train_loss!='crf'; using argmax decode because CRF pairwise may be untrained.")

    road_adj_list = build_adj_list(gb.num_nodes, gb.edge_index)
    # CRF allowed_prev
    rev_edge_index = torch.stack([gb.edge_index[1], gb.edge_index[0]], dim=0)
    road_adj_list_rev = build_adj_list(gb.num_nodes, rev_edge_index)
    k_hop = int(cfg["train"]["k_hop"])
    allowed_prev = k_hop_neighbors(road_adj_list_rev, k=k_hop)

    # load test pairs
    all_files = glob.glob(os.path.join(test_dir, "**", "*.mat"), recursive=True)
    pairs = pair_neg_gt(all_files)
    if not pairs:
        raise RuntimeError(
            f"No neg-gt pairs found in {test_dir}. "
            f"Make sure you have both *_neg_*.mat and matching *_gt_*.mat."
        )

    raw_preds, preds_ungated, preds_gated, golds = [], [], [], []

    with torch.no_grad():
        printed = 0
        for neg_path, gt_path in pairs:
            # read pts_coord
            x1, y1, f1 = load_pts_coord_any(neg_path)
            x2, y2, f2 = load_pts_coord_any(gt_path)

            # map to node sequences
            pred_seq = coords_to_global_node_seq(x1, y1, f1, gb, floor_base=floor_base, xy_mode=traj_xy_mode)
            true_seq = coords_to_global_node_seq(x2, y2, f2, gb, floor_base=floor_base, xy_mode=traj_xy_mode)

            L = min(len(pred_seq), len(true_seq))
            if L < 3:
                continue

            pred_seq = pred_seq[:L]
            true_seq = true_seq[:L]

            pred = torch.tensor([pred_seq], dtype=torch.long, device=device)
            lengths = torch.tensor([L], dtype=torch.long, device=device)

            # Build trajectory transition graph from current predicted sequence to match
            # training-time use of traj_gcn (which always consumes transition edges).
            ecount = build_transition_graph([pred_seq], directed=True, min_count=1)
            if ecount:
                src = torch.tensor([u for (u, v, c) in ecount], dtype=torch.long, device=device)
                dst = torch.tensor([v for (u, v, c) in ecount], dtype=torch.long, device=device)
                w = torch.tensor([float(c) for (u, v, c) in ecount], dtype=torch.float32, device=device)
                w = w / w.mean().clamp(min=1e-6)
                traj_edge_index = torch.stack([src, dst], dim=0)
                traj_edge_weight = w
            else:
                traj_edge_index = None
                traj_edge_weight = None

            unary_logits, H_R = model.forward_unary(
                pred, lengths,
                node_num_feat=gb.node_num_feat,
                floor_id=gb.floor_id,
                edge_index=gb.edge_index,
                edge_attr=gb.edge_attr,
                traj_edge_index=traj_edge_index,
                traj_edge_weight=traj_edge_weight,
                teacher_forcing=None
            )
            unary = unary_logits[0, :L, :]

            if use_crf_decode:
                corrected = model.crf.viterbi_one(
                    unary_logits=unary,
                    H=H_R,
                    allowed_prev=allowed_prev,
                    top_r=int(cfg["train"]["top_r_decode"])
                )
            else:
                corrected = decode_argmax(unary_logits, lengths)[0]

            corrected_gated = confidence_gate_sequence(
                raw_seq=pred_seq,
                corrected_seq=corrected,
                unary_logits=unary,
                min_confidence=min_correction_confidence,
                min_logit_gain=min_correction_logit_gain,
            )
            final_corrected = corrected if args.disable_gate else corrected_gated

            raw_preds.append(pred_seq)
            preds_ungated.append(corrected)
            preds_gated.append(corrected_gated)
            golds.append(true_seq)

            # ---- print this sample (limited by --max_print; <=0 disables sample dumps) ----
            maxp = int(args.max_print)
            if maxp > 0 and printed < maxp:
                printed += 1
                print("\n==============================")
                print("NEG file:", os.path.basename(neg_path))
                print("GT  file:", os.path.basename(gt_path))
                print(f"len(pred_ids)={len(pred_seq)} len(corr_ids)={len(final_corrected)} len(gt_ids)={len(true_seq)}")

                print("\n[pred_ids]")
                print(pred_seq[:maxp], "..." if len(pred_seq) > maxp else "")

                print("\n[corrected_ids]")
                print(final_corrected[:maxp], "..." if len(final_corrected) > maxp else "")

                print("\n[gt_ids]")
                print(true_seq[:maxp], "..." if len(true_seq) > maxp else "")

                floor_base_out = 1 if floor_base == 1 else 0
                print("\n[pred (id,x,y,f)]")
                print(ids_to_xyzf(pred_seq[:maxp], gb, floor_base_out=floor_base_out))

                print("\n[corrected (id,x,y,f)]")
                print(ids_to_xyzf(final_corrected[:maxp], gb, floor_base_out=floor_base_out))

                print("\n[gt (id,x,y,f)]")
                print(ids_to_xyzf(true_seq[:maxp], gb, floor_base_out=floor_base_out))

    raw_tok, raw_seq = token_seq_accuracy(raw_preds, golds)
    ungated_tok, ungated_seq = token_seq_accuracy(preds_ungated, golds)
    gated_tok, gated_seq = token_seq_accuracy(preds_gated, golds)

    final_preds = preds_ungated if args.disable_gate else preds_gated
    tok, seq = token_seq_accuracy(final_preds, golds)
    feas = path_feasibility_rate(final_preds, road_adj_list, k_hop=max(1, k_hop))

    total_tokens = 0
    changed_tokens = 0
    ch_raw_argmax = 0
    ch_argmax_decode = 0
    ch_decode_final = 0
    for raw, ung, gat, fin in zip(raw_preds, preds_ungated, preds_gated, final_preds):
        L = min(len(raw), len(ung), len(gat), len(fin))
        total_tokens += L
        changed_tokens += sum(int(a != b) for a, b in zip(raw[:L], fin[:L]))
        ch_raw_argmax += sum(int(a != b) for a, b in zip(raw[:L], ung[:L]))
        ch_argmax_decode += 0
        ch_decode_final += sum(int(a != b) for a, b in zip(ung[:L], fin[:L]))
    change_ratio = (changed_tokens / total_tokens) if total_tokens > 0 else 0.0
    changed_raw_to_argmax = (ch_raw_argmax / total_tokens) if total_tokens > 0 else 0.0
    changed_argmax_to_decode = (ch_argmax_decode / total_tokens) if total_tokens > 0 else 0.0
    changed_decode_to_final = (ch_decode_final / total_tokens) if total_tokens > 0 else 0.0

    print(
        f"\n[TEST] n={len(final_preds)} raw_tok={raw_tok:.3f} ungated_tok={ungated_tok:.3f} gated_tok={gated_tok:.3f} "
        f"tok={tok:.3f} raw_seq={raw_seq:.3f} ungated_seq={ungated_seq:.3f} gated_seq={gated_seq:.3f} seq={seq:.3f} "
        f"feas@k={feas:.3f} changed={change_ratio:.3f} changed_raw_to_argmax={changed_raw_to_argmax:.3f} "
        f"changed_argmax_to_decode={changed_argmax_to_decode:.3f} changed_decode_to_final={changed_decode_to_final:.3f} "
        f"gate_enabled={int(not args.disable_gate)} crf_decode={int(use_crf_decode)} "
        f"conf_gate={min_correction_confidence:.2f} gain_gate={min_correction_logit_gain:.2f}"
    )


if __name__ == "__main__":
    main()

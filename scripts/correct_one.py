import argparse, os, sys
from pathlib import Path
import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graphmm.io.multifloor_builder import build_multifloor_graph_with_features
from graphmm.models.graphmm_corrector import GraphMMCorrector, decode_argmax
from graphmm.datasets.trajectory_provider import build_transition_graph
from graphmm.utils.graph import build_adj_list, k_hop_neighbors

def parse_int_list(s: str):
    return [int(x) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/mall_train.yaml"))
    ap.add_argument("--mat_paths", nargs="*", required=True, help="5 mat files in floor order")
    ap.add_argument("--ckpt", required=True, help="e.g. runs/mall_final/checkpoint.pt")
    ap.add_argument("--pred", required=True, help="comma-separated global node ids, e.g. 10,25,26,40")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) build graph (must match training)
    gb = build_multifloor_graph_with_features(
        mat_paths=args.mat_paths,
        ppm=cfg["data"]["ppm"],
        directed_road=cfg["data"]["directed_road"],
        add_vertical_bidirectional=cfg["data"]["add_vertical_bidirectional"],
        coord_match_eps=cfg["data"]["coord_match_eps"],
        device=device,
    )

    # 2) build model + load ckpt
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

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    pred_seq = parse_int_list(args.pred)
    pred = torch.tensor([pred_seq], dtype=torch.long, device=device)
    lengths = torch.tensor([len(pred_seq)], dtype=torch.long, device=device)

    # 3) forward unary
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

    # 4) decode
    if cfg["model"]["use_crf"]:
        road_adj_list = build_adj_list(gb.num_nodes, gb.edge_index)
        allowed_prev = k_hop_neighbors(road_adj_list, k=cfg["train"]["k_hop"])
        path = model.crf.viterbi_one(
            unary_logits=unary_logits[0, :len(pred_seq), :],
            H=H_R,
            allowed_prev=allowed_prev,
            top_r=cfg["train"]["top_r_decode"]
        )
    else:
        path = decode_argmax(unary_logits, lengths)[0]

    print("pred      =", pred_seq)
    print("corrected =", path)

if __name__ == "__main__":
    main()

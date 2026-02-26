import argparse, os, random, sys
from pathlib import Path
import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graphmm.utils.seed import set_seed
from graphmm.io.multifloor_builder import build_multifloor_graph_with_features, GraphBatch
from graphmm.datasets.trajectory_provider import sim_samples_on_graph, load_paired_samples_from_runs
from graphmm.utils.graph import build_adj_list
from graphmm.models.graphmm_corrector import GraphMMCorrector
from graphmm.training import train_loop

def build_toy(device: str) -> GraphBatch:
    # minimal toy: 2 floors, 1 connector
    num_nodes = 6
    floor_id = torch.tensor([0,0,0,1,1,1], dtype=torch.long, device=device)
    node_num_feat = torch.zeros((num_nodes,5), dtype=torch.float32, device=device)
    node_num_feat[:,0] = torch.tensor([0.1,0.2,0.3,0.1,0.2,0.3], device=device)
    node_num_feat[:,1] = torch.tensor([0.1,0.2,0.3,0.1,0.2,0.3], device=device)

    edges = [(0,1),(1,2),(3,4),(4,5),(2,5),(5,2)]
    src = torch.tensor([u for u,v in edges], dtype=torch.long, device=device)
    dst = torch.tensor([v for u,v in edges], dtype=torch.long, device=device)
    edge_index = torch.stack([src,dst], dim=0)

    edge_attr = torch.zeros((len(edges),5), dtype=torch.float32, device=device)
    edge_attr[:,0] = 1.0
    edge_attr[:,1] = torch.tensor([0,0,0,0,1,1], dtype=torch.float32, device=device)
    edge_attr[:,2] = torch.tensor([1,1,1,1,10,10], dtype=torch.float32, device=device)
    edge_is_vertical = edge_attr[:,1].bool()

    for u,v in edges:
        node_num_feat[v,2] += 1
        node_num_feat[u,3] += 1
    node_num_feat[:,2] = torch.log1p(node_num_feat[:,2])
    node_num_feat[:,3] = torch.log1p(node_num_feat[:,3])
    node_num_feat[2,4] = 1.0
    node_num_feat[5,4] = 1.0

    coord_xy = torch.tensor(
        [[185, 110], [370, 220], [555, 330], [185, 110], [370, 220], [555, 330]],
        dtype=torch.float32, device=device
    )  # [N,2] 随便给个像素坐标即可，只用于最近邻映射

    return GraphBatch(num_nodes, edge_index, edge_attr, node_num_feat, floor_id, edge_is_vertical, (1100, 1850),
                      coord_xy)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT/"configs/mall_train.yaml"))
    ap.add_argument("--mat_paths", nargs="*", default=None)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.mat_paths is not None and len(args.mat_paths) > 0:
        cfg["data"]["mat_paths"] = args.mat_paths

    device = cfg["train"]["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(cfg["train"]["seed"])

    mat_paths = cfg["data"].get("mat_paths", [])
    if mat_paths and all(os.path.exists(p) for p in mat_paths):
        gb = build_multifloor_graph_with_features(
            mat_paths=mat_paths,
            ppm=cfg["data"]["ppm"],
            directed_road=cfg["data"]["directed_road"],
            add_vertical_bidirectional=cfg["data"]["add_vertical_bidirectional"],
            coord_match_eps=cfg["data"]["coord_match_eps"],
            device=device,
        )
        print(f"[graph] nodes={gb.num_nodes} edges={gb.edge_index.size(1)} vertical_edges={int(gb.edge_is_vertical.sum().item())}")
    else:
        gb = build_toy(device=device)
        print("[graph] Using TOY graph (mat files not provided or not found).")

    adj = build_adj_list(gb.num_nodes, gb.edge_index)

    samples = sim_samples_on_graph(
        adj_list=adj,
        num_samples=cfg["sim"]["num_samples"],
        seq_len=cfg["sim"]["seq_len"],
        noise_rate=cfg["sim"]["noise_rate"],
        prefer_intrafloor=cfg["sim"]["prefer_intrafloor"],
        floor_id=gb.floor_id,
    )
    random.shuffle(samples)
    k = int(len(samples) * cfg["sim"]["train_ratio"])
    # train_samples, valid_samples = samples[:k], samples[k:]
    xy_mode = cfg["data"].get("traj_xy_mode", "auto")
    train_samples = load_paired_samples_from_runs("data/traj/train", gb, floor_base=1, xy_mode=xy_mode)
    valid_samples = load_paired_samples_from_runs("data/traj/valid", gb, floor_base=1, xy_mode=xy_mode)

    print(f"[data] train={len(train_samples)} valid={len(valid_samples)}")

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
        input_anchor_bias=cfg["model"].get("input_anchor_bias", 0.0),
    )

    run_dir = ROOT / "runs" / cfg["output"]["run_name"]
    train_loop(
        model=model,
        graph_batch=gb,
        train_samples=train_samples,
        valid_samples=valid_samples,
        epochs=cfg["train"]["epochs"],
        batch_size=cfg["train"]["batch_size"],
        lr=cfg["train"]["lr"],
        device=device,
        run_dir=str(run_dir),
        k_hop=cfg["train"]["k_hop"],
        top_r_train=cfg["train"]["top_r_train"],
        top_r_decode=cfg["train"]["top_r_decode"],
        use_crf=cfg["model"]["use_crf"],
        ss_start=cfg["train"].get("ss_start", 1.0),
        ss_end=cfg["train"].get("ss_end", 1.0),
        ss_mode=cfg["train"].get("ss_mode", "linear"),
        traj_graph_source=cfg["train"].get("traj_graph_source", "mixed"),
    )

if __name__ == "__main__":
    main()

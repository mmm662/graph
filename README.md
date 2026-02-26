# GraphMM Mall — Final (Edge-aware + TrajGraph + CRF)

This is a **full runnable** project:
- Multi-floor mall map: one `.mat` per floor (keys: `E`, `v`, optional `bump`)
- Vertical connectors inferred by matching same `(x,y)` endpoints across adjacent floors
- **Node features**: x_norm, y_norm, log1p(deg_in), log1p(deg_out), is_connector + floor embedding
- **Edge features**: length_norm, is_vertical, road_type(t), ud, radius_norm
- Road graph encoder: **Edge-aware GINE**
- Trajectory correlation encoder: **Trajectory transition graph GCN** (from predicted sequences)
- Seq2Seq + attention decoder
- **Graph-CRF** with candidate pruning + k-hop reachability constraint

## pip tsinghua config
```bash
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn
python -m pip config set global.timeout 120
```

## Install
```bash
pip install -e .
pip install -r requirements.txt
```

## Run (toy fallback)
```bash
python scripts/train.py
or
python scripts/train.py --mat_paths floor1.mat floor2.mat floor3.mat floor4.mat floor5.mat
```

## Run test
```bash
python scripts/test.py --config configs/mall_train.yaml
or
python scripts/test.py --config configs/mall_train.yaml --test_dir data/traj/valid
```

## Notes
- Trajectories are simulated on the graph for now. Replace `src/graphmm/datasets/trajectory_provider.py` with your real reader later.
- Outputs:
  - prints loss + token acc + seq acc + feasibility rate
  - saves checkpoint to `runs/<run_name>/checkpoint.pt`

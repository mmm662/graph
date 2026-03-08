# GraphMM Mall — Edge-aware + TrajGraph + CRF

一个可直接运行的多楼层轨迹纠错项目，包含：
- 多楼层路网构建（每层一个 `.mat`）
- 路网编码（Edge-aware GINE）
- 轨迹转移图编码（Trajectory GCN）
- Seq2Seq + Attention 解码
- Graph-CRF 结构化解码
- 纠错门控（置信度 + logit 增益）

---

## 1. 功能概览

### 1.1 图构建
- 支持楼层图输入：`E`, `v`（可选 `bump`）
- 通过相邻楼层同 `(x,y)` 端点自动推断竖向连接边
- 可配置是否有向图、是否补双向竖向边、坐标匹配阈值

### 1.2 特征
- 节点特征：`x_norm, y_norm, log1p(deg_in), log1p(deg_out), is_connector + floor_emb`
- 边特征：`length_norm, is_vertical, road_type(t), ud, radius_norm`
- 道路语义：仅 `t=1` 默认双向；其余道路类型均按 `v1->v2` 定向；`c` 代表楼层增量（如 `+1/-1/+2`）。

### 1.3 模型
- 路网编码器：Edge-aware GINE
- 轨迹相关编码器：基于轨迹转移边的 Weighted GCN
- 解码器：GRU + DotAttention
- 可选 Graph-CRF：在拓扑约束下做 Viterbi

### 1.4 纠错门控（关键）
在推理/评估时，纠错 token 只有在以下两个条件都满足时才会被接受：
1. 置信度（softmax max prob）≥ `min_correction_confidence`
2. 改后 token 的 unary logit - 原 token unary logit ≥ `min_correction_logit_gain`

否则回退到原始输入 token，减少误改。

---

## 2. 安装

### 2.1 可选：清华源
```bash
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn
python -m pip config set global.timeout 120
```

### 2.2 安装依赖
```bash
pip install -e .
pip install -r requirements.txt
```

---

## 3. 数据与目录说明

- 楼层图：`data/mall/floor*.mat`
- 轨迹数据：
  - 训练：`data/traj/train`
  - 验证：`data/traj/valid`
  - 测试：`data/traj/test`
- 默认配置：`configs/mall_train.yaml`（当前为 CE baseline）
- 输出目录：`runs/<run_name>/`
  - `checkpoint.pt`
  - `resolved_config.yaml`（训练启动时写入的生效配置快照）

---

## 4. 配置文件与实验路线

推荐按下面配置逐步做实验：

- `configs/mall_train_ce_base.yaml`：CE baseline（无 CRF、无 anchor、traj_gcn=0）
- `configs/mall_train_ce_traj.yaml`：CE + TrajGCN（traj_gcn=1）
- `configs/mall_train_anchor.yaml`：anchor 对照实验
- `configs/mall_train_crf.yaml`：CRF 训练（`use_crf=true` + `crf_train_loss=crf`）

> `configs/mall_train.yaml` 也可直接跑，默认与 CE baseline 对齐。

---

## 5. 训练

```bash
python scripts/train.py --config configs/mall_train_ce_base.yaml
```

常见覆盖参数：
```bash
python scripts/train.py --config configs/mall_train_ce_base.yaml \
  --input_anchor_bias 0.5 \
  --traj_gcn_layers 1 \
  --temperature 4 \
  --error_token_weight 4
```

训练日志会打印关键生效项：
- `traj_graph` / `traj_scope`
- `infer_ctx` / `anchor_inf` / `anchor_train`
- `use_crf` / `crf_loss` / `crf_decode`
- `raw_tok/pred_tok/gated_tok/final_tok/final_seq`
- `pred_changed/gated_changed/changed/gate_keep`

---

## 6. 测试与推理

```bash
python scripts/test.py --config configs/mall_train_ce_base.yaml
```

常用覆盖项：
```bash
python scripts/test.py --config configs/mall_train_ce_base.yaml --test_dir data/traj/valid
python scripts/test.py --config configs/mall_train_ce_base.yaml --ckpt runs/<run_name>/checkpoint.pt
python scripts/test.py --config configs/mall_train_ce_base.yaml --disable_gate
```

`test.py` 会显式打印：
- `use_crf_cfg / crf_train_loss / use_crf_decode`
- `traj_gcn_layers`
- `inference_use_input_context`
- `input_anchor_bias / apply_input_anchor_bias_inference`

并输出分层改动率：
- `changed_raw_to_argmax`
- `changed_argmax_to_decode`
- `changed_decode_to_final`

---

## 7. 纠错诊断

```bash
python scripts/diagnose_corrections.py \
  --config configs/mall_train_ce_base.yaml \
  --ckpt runs/<run_name>/checkpoint.pt \
  --test_dir data/traj/test
```

常用参数：
- `--output_csv runs/diagnostics/token_diagnostics.csv`
- `--disable_gate`
- `--force_argmax_decode`
- `--temperature_override 4`
- `--print_raw_wrong_table`

诊断输出包括：
- `R_keep / R_fix / R_over`
- `R_reject_correct / R_block_wrong`
- `crf_gain`
- `changed_raw_to_argmax / changed_argmax_to_decode / changed_decode_to_final`
- `illegal_rate_raw/argmax/decode/final`
- `wrong_hit@5 / wrong_hit@10 / wrong_rank_gt`

---

## 8. 单条轨迹纠错（手动）

```bash
python scripts/correct_one.py \
  --config configs/mall_train_ce_base.yaml \
  --mat_paths data/mall/floor1.mat data/mall/floor2.mat data/mall/floor3.mat data/mall/floor4.mat data/mall/floor5.mat \
  --ckpt runs/<run_name>/checkpoint.pt \
  --pred 10,25,26,40
```

脚本会打印：`use_crf_cfg / crf_train_loss / use_crf_decode`，并输出纠错前后 token 序列。

---

## 9. 关键配置说明

### 9.1 解码与 CRF
- `model.use_crf`：是否构建 CRF
- `train.crf_train_loss`：`ce` 或 `crf`
  - 只有 `crf_train_loss=crf` 时才会启用 CRF decode
- `train.top_r_train / train.top_r_decode`：CRF 候选宽度
- `train.k_hop`：CRF 可达约束半径

### 9.2 输入锚定与推理上下文
- `model.input_anchor_bias`
- `model.apply_input_anchor_bias_inference`
- `model.apply_input_anchor_bias_training`
- `model.inference_use_input_context`

### 9.3 轨迹图构建一致性
- `train.traj_graph_source`: `pred | true | mixed`
- `train.traj_graph_build_scope`: `global | batch`
  - 推荐 `global`，保证 train/eval 行为一致

### 9.4 训练动态
- `ss_start / ss_end / ss_mode`：scheduled sampling
- `error_token_weight`：对原始错误位加权

---

## 10. 统一 checkpoint 加载

`test.py / diagnose_corrections.py / correct_one.py` 统一使用：
- `graphmm.io.checkpoint.load_model_state_dict`
- `graphmm.io.checkpoint.infer_traj_gcn_layers`

这样可以兼容更多 checkpoint 格式并减少脚本之间行为分叉。

---

## 11. 参考文档

- 变更记录：`CHANGELOG.md`
- 运行手册：`RUNBOOK.md`

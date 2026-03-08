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

## 2.1 可选：清华源
```bash
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn
python -m pip config set global.timeout 120
```

## 2.2 安装依赖
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
- 默认配置：`configs/mall_train.yaml`
- 输出 checkpoint：`runs/<run_name>/checkpoint.pt`

> 当前配置默认使用真实 paired 轨迹目录（`*_neg_*.mat` 与 `*_gt_*.mat`）进行训练/验证；
> toy graph 仅在楼层图缺失时回退。

---

## 4. 训练

```bash
python scripts/train.py --config configs/mall_train.yaml
```

也可覆盖楼层图路径：
```bash
python scripts/train.py --config configs/mall_train.yaml --mat_paths data/mall/floor1.mat data/mall/floor2.mat data/mall/floor3.mat data/mall/floor4.mat data/mall/floor5.mat
```

### 4.1 训练日志字段解释
每个 epoch 会打印以下核心指标：
- `raw_tok`：原始输入轨迹（不纠错）的 token accuracy（基线）
- `pred_tok`：模型解码后（未 gate）的 token accuracy
- `gated_tok`：应用纠错 gate 后的 token accuracy
- `final_tok/final_seq`：用于模型选择与保存的最终指标
- `pred_changed`：未 gate 解码相对原输入的改动率
- `gated_changed`：gate 后相对原输入的改动率
- `changed`：当前最终输出（由 `eval_apply_gate` 决定）的改动率
- `gate_keep`：gate 保留下来的改动比例（`gated_changed / pred_changed`）
- `feas@k`：路径可达性指标

### 4.2 为什么有时会看到 `raw_tok` 一直稳定
`raw_tok` 是数据基线，不会随模型变化太大；应关注 `pred_tok/gated_tok/final_tok` 的变化。

---

## 5. 测试 / 推理

```bash
python scripts/test.py --config configs/mall_train.yaml
```

常用覆盖项：
```bash
python scripts/test.py --config configs/mall_train.yaml --test_dir data/traj/valid
python scripts/test.py --config configs/mall_train.yaml --ckpt runs/<run_name>/checkpoint.pt
python scripts/test.py --config configs/mall_train.yaml --max_print 10
python scripts/test.py --config configs/mall_train.yaml --disable_gate
```



### 5.1 纠错诊断（逐时刻日志 + 核心排查指标）

当你要定位“过度纠错 / 纠错失败 / CRF 与 gate 是否起反作用”时，可运行：

```bash
python scripts/diagnose_corrections.py --config configs/mall_train.yaml --ckpt runs/<run_name>/checkpoint.pt --test_dir data/traj/test
```

常用参数：
- `--output_csv runs/diagnostics/token_diagnostics.csv`：输出逐 token 诊断表
- `--disable_gate`：仅看解码上限（不经过 gate）
- `--force_argmax_decode`：禁用 CRF，仅看 unary argmax
- `--max_hops`：拓扑距离上限

脚本会输出：
- `R_keep / R_fix / R_over`
- `R_reject_correct / R_block_wrong`（gate 是否过保守）
- `illegal_rate_raw/argmax/decode/final`（拓扑非法转移率）
- 逐时刻 CSV 字段（`x_t, y_gt, y_argmax, y_decode, y_final, u_gt, u_x, gain_gtx, conf, rank_gt, gt_in_top10, top1_margin, topo_dist_* ...`）

测试输出会同时给出：
- `raw_tok/raw_seq`（原输入基线）
- `ungated_tok/ungated_seq`（不经过 gate 的模型解码）
- `gated_tok/gated_seq`（经过 gate 后的结果）
- `tok/seq`（本次运行最终采用的结果；`--disable_gate` 时等于 ungated）
- `ungated_changed/gated_changed/changed`（未 gate、gate 后、最终改动率）
- `gate_keep`（门控保留比例）
- `conf_gate/gain_gate`（门控阈值）

---

## 6. 关键配置（`configs/mall_train.yaml`）

### 6.1 解码与纠错
- `model.use_crf`：是否启用 CRF
- `train.top_r_train / train.top_r_decode`：CRF 候选搜索宽度
- `model.min_correction_confidence`：置信度阈值
- `model.min_correction_logit_gain`：logit 增益阈值
- `model.apply_input_anchor_bias_inference`：是否在推理时给原输入 token 加偏置（建议默认 `true`，用于稳定近似复制+局部纠错任务）
- `model.apply_input_anchor_bias_training`：是否在训练 CE 分支给原输入 token 加偏置（建议 `true`，用于稳定“少改动纠错”任务）
- `model.inference_use_input_context`：推理时是否使用输入轨迹上下文解码（建议 `true`，可显著提升纠错稳定性）
- `train.eval_apply_gate`：训练期验证时是否使用 gate 后结果作为最终评分
- `train.crf_train_loss`：CRF 开启时训练损失类型，`ce`（推荐，稳定）或 `crf`（结构化 NLL）
> 重要：仅当 `crf_train_loss: crf` 时才会启用 CRF pairwise 解码；若为 `ce`，评估/测试将自动回退为 argmax 解码（避免未训练的 `crf.W` 破坏结果）。

> 说明：`input_anchor_bias` 仅在推理分支生效，不参与 teacher-forcing 训练 loss；若出现 `changed≈0` 的复制现象，优先关闭 `apply_input_anchor_bias_inference` 并将 `input_anchor_bias` 设为 0。

### 6.2 轨迹图来源
- `train.traj_graph_source` 可选：
  - `pred`：仅用输入预测轨迹构建转移图
  - `true`：仅用真值轨迹构建转移图
  - `mixed`：两者混合

> 注意：建议写成字符串（如 `"true"`），避免 YAML 将其解析为布尔值。

### 6.3 训练动态
- `ss_start / ss_end / ss_mode`：scheduled sampling
- `epochs / batch_size / lr`：训练预算
- `error_token_weight`：对原始错误位置（`pred!=true`）的 CE 加权系数，提升“纠错位”学习强度

---

## 7. 常见问题排查

### 7.1 `val_tok` 看起来“卡住”
先确认看的是哪个指标：
1. `raw_tok` 是基线，可能长期稳定
2. 看 `pred_tok` 是否在变
3. 再看 `gated_tok` 与 `changed`，判断 gate 是否过严

建议排查顺序：
- 将 `train.eval_apply_gate` 设为 `false` 观察未 gate 学习能力
- 降低 `min_correction_confidence` 或 `min_correction_logit_gain`
- 尝试 `traj_graph_source: mixed` 与 `true` 对比
- 若 `raw_tok` 与 `gated_tok` 长期相同，先看 `gate_keep`：接近 0 说明 gate 几乎把改动全回退；可继续下调门限。

### 7.2 CRF 权重加载报错（缺少 `crf.W`）
说明 checkpoint 与当前 `use_crf` 配置不一致：
- `use_crf: true` 需要用 CRF 训练得到的 checkpoint
- 或临时改回 `use_crf: false` 使用旧权重

### 7.3 `No neg-gt pairs found`
确认测试目录下存在成对文件：
- `*_neg_*.mat`
- 对应 `*_gt_*.mat`

---

## 8. 备注
- 当前工程中的轨迹读取逻辑在 `src/graphmm/datasets/trajectory_provider.py`；
  如需接入真实业务数据，可替换该模块。
- 建议固定随机种子并记录配置文件快照，便于复现实验。

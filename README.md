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
- `changed`：相对原输入被改动的 token 比例
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
```

测试输出会同时给出：
- `raw_tok/raw_seq`（原输入基线）
- `tok/seq`（最终纠错后）
- `changed`（改动率）
- `conf_gate/gain_gate`（门控阈值）

---

## 6. 关键配置（`configs/mall_train.yaml`）

### 6.1 解码与纠错
- `model.use_crf`：是否启用 CRF
- `train.top_r_train / train.top_r_decode`：CRF 候选搜索宽度
- `model.min_correction_confidence`：置信度阈值
- `model.min_correction_logit_gain`：logit 增益阈值
- `train.eval_apply_gate`：训练期验证时是否使用 gate 后结果作为最终评分
- `train.crf_train_loss`：CRF 开启时训练损失类型，`ce`（推荐，稳定）或 `crf`（结构化 NLL）

> 说明：`input_anchor_bias` 仅在推理分支生效，不参与 teacher-forcing 训练 loss，避免训练阶段被“复制输入”偏置干扰。

### 6.2 轨迹图来源
- `train.traj_graph_source` 可选：
  - `pred`：仅用输入预测轨迹构建转移图
  - `true`：仅用真值轨迹构建转移图
  - `mixed`：两者混合

> 注意：建议写成字符串（如 `"true"`），避免 YAML 将其解析为布尔值。

### 6.3 训练动态
- `ss_start / ss_end / ss_mode`：scheduled sampling
- `epochs / batch_size / lr`：训练预算

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

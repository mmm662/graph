# Runbook

## 1) CE baseline
```bash
python scripts/train.py --config configs/mall_train_ce_base.yaml
python scripts/test.py --config configs/mall_train_ce_base.yaml
python scripts/diagnose_corrections.py --config configs/mall_train_ce_base.yaml
```

## 2) CE + TrajGCN
```bash
python scripts/train.py --config configs/mall_train_ce_traj.yaml
python scripts/test.py --config configs/mall_train_ce_traj.yaml
```

## 3) Anchor ablation
```bash
python scripts/train.py --config configs/mall_train_anchor.yaml
python scripts/diagnose_corrections.py --config configs/mall_train_anchor.yaml
```

## 4) CRF training
```bash
python scripts/train.py --config configs/mall_train_crf.yaml
python scripts/test.py --config configs/mall_train_crf.yaml
```

## 5) Inference mode ablation
Override in config or CLI-compatible edited copy:
- `model.inference_use_input_context: true`
- `model.inference_use_input_context: false`

## Notes
- Every training run writes `runs/<run_name>/resolved_config.yaml`.
- `scripts/test.py`, `scripts/diagnose_corrections.py`, and `scripts/correct_one.py` share the same checkpoint-loading backend.

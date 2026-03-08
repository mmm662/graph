# Changelog

## Unreleased

### Fixed
- CRF NLL now uses a consistent raw-unary energy space in forward recursion and gold-path scoring.
- Added CRF input validation for tensor shape, gold length, and `allowed_prev` consistency.

### Changed
- Added `force_autoregressive_decode` to `GraphMMCorrector.forward_unary`.
- Scheduled sampling now explicitly samples from a fully autoregressive decode path.
- Train/eval trajectory-graph construction is now controlled by `train.traj_graph_build_scope` (`global` or `batch`) and uses one consistent policy.
- Training now saves `runs/<run_name>/resolved_config.yaml` and stores config metadata in checkpoints.
- Training logs now print effective decode/bias/CRF settings.
- Unified checkpoint loading through `graphmm.io.checkpoint.load_model_state_dict` and reused it in `test.py`, `diagnose_corrections.py`, and `correct_one.py`.
- Added explicit decode/config summary and staged changed-rate metrics in test/diagnose outputs.

### Added
- New configs for clear experiment routes:
  - `configs/mall_train_ce_base.yaml`
  - `configs/mall_train_ce_traj.yaml`
  - `configs/mall_train_crf.yaml`
  - `configs/mall_train_anchor.yaml`

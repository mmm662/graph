from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Mapping, List

import torch


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


def _format_file_signature(path: Path) -> str:
    try:
        head = _peek_file_prefix(path, n=32)
    except OSError:
        return "<unreadable>"
    return f"hex={head.hex()} ascii={''.join(chr(b) if 32 <= b <= 126 else '.' for b in head)}"


def _try_legacy_torch_load(path: Path, device: str):
    with path.open("rb") as f:
        return torch.serialization._legacy_load(
            f,
            map_location=device,
            pickle_module=pickle,
            encoding="latin1",
        )


def load_model_state_dict(ckpt_path: str, device: str) -> Mapping[str, Any]:
    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_file}")
    if ckpt_file.is_dir():
        raise IsADirectoryError(f"checkpoint path is a directory: {ckpt_file}")

    _raise_if_lfs_pointer(ckpt_file)
    size_bytes = ckpt_file.stat().st_size
    load_errors: List[str] = []

    for loader_name, loader in [
        ("torch.load", lambda: torch.load(str(ckpt_file), map_location=device)),
        ("torch.load(weights_only=False)", lambda: torch.load(str(ckpt_file), map_location=device, weights_only=False)),
        ("torch.jit.load", lambda: torch.jit.load(str(ckpt_file), map_location=device).state_dict()),
        ("torch.serialization._legacy_load", lambda: _try_legacy_torch_load(ckpt_file, device=device)),
    ]:
        try:
            state = loader()
            if isinstance(state, Mapping):
                if "model" in state and isinstance(state["model"], Mapping):
                    return state["model"]
                if "state_dict" in state and isinstance(state["state_dict"], Mapping):
                    return state["state_dict"]
                sample_keys = list(state.keys())
                if sample_keys and all(isinstance(k, str) for k in sample_keys):
                    if any(k.startswith(("node_encoder.", "gine.", "encoder.", "decoder.", "crf.")) for k in sample_keys):
                        return state
        except Exception as exc:
            load_errors.append(f"{loader_name} failed: {type(exc).__name__}: {exc}")

    _raise_if_lfs_pointer(ckpt_file)
    summary = "\n  - ".join(load_errors[-4:])
    raise RuntimeError(
        "Failed to load checkpoint.\n"
        f"  path: {ckpt_file}\n"
        f"  size_bytes: {size_bytes}\n"
        f"  file_signature: {_format_file_signature(ckpt_file)}\n"
        "  attempted loaders:\n"
        f"  - {summary}"
    )


def infer_traj_gcn_layers(state_dict: Mapping[str, Any]) -> int:
    max_idx = -1
    pat = re.compile(r"^traj_gcn\.(\d+)\.")
    for key in state_dict.keys():
        m = pat.match(str(key))
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1

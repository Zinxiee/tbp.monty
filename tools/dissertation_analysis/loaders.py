# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Thin wrappers over load_stats() that survive missing files."""

from __future__ import annotations

import warnings

import pandas as pd
import torch

from tbp.monty.frameworks.utils.logging_utils import load_stats as _load_stats


def load_stats(
    run_path: Path,
    *,
    load_train: bool = True,
    load_eval: bool = True,
    load_detailed: bool = False,
    load_models: bool = False,
    pretrained_dict: Path | None = None,
):
    """Load stats files if run_path exists, otherwise return empty outputs.

    Returns:
        Tuple of train, eval, detailed and model outputs.
    """
    if not run_path.exists():
        warnings.warn(f"missing {run_path}", stacklevel=2)
        return None, None, None, None
    return _load_stats(
        run_path,
        load_train=load_train,
        load_eval=load_eval,
        load_detailed=load_detailed,
        load_models=load_models,
        pretrained_dict=pretrained_dict,
    )


def load_csv(run_path: Path, kind: str) -> pd.DataFrame | None:
    """Load <kind>_stats.csv if present.

    Args:
        kind: "train" or "eval".

    Returns:
        Loaded dataframe or None.
    """
    csv_path = run_path / f"{kind}_stats.csv"
    if not csv_path.exists():
        warnings.warn(f"missing {csv_path}", stacklevel=2)
        return None
    return pd.read_csv(csv_path)


def load_eval_or_train(run_path: Path) -> pd.DataFrame | None:
    """Prefer eval_stats.csv; fall back to train_stats.csv.

    Returns:
        Loaded dataframe or None.
    """
    df = load_csv(run_path, "eval")
    if df is not None:
        return df
    return load_csv(run_path, "train")


def load_model(run_path: Path, epoch: int = 0) -> dict | None:
    """Load <run_path>/<epoch>/model.pt if present.

    Returns:
        Raw torch state dict with ``lm_dict`` key, or None.
    """
    model_path = run_path / str(epoch) / "model.pt"
    if not model_path.exists():
        warnings.warn(f"missing {model_path}", stacklevel=2)
        return None
    try:
        return torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location="cpu")


def object_label_from_suffix(suffix: str, fallback: str = "ALL") -> str:
    """Parse the suffix of a dissertation run dir into an object label.

    Examples:
        "" → "ALL"
        "O1_mug" → "O1_mug"
        "rot1" → "rot1"  (Exp 2 — caller maps to ORI1)
    Returns:
        Parsed label or fallback.
    """
    return suffix or fallback

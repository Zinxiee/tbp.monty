# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Aggregate raw stats CSVs into summary tables and Markdown output."""

from __future__ import annotations

import numpy as np
import pandas as pd


def filter_lm_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Stats CSVs contain one row per (episode, LM). Keep LM_0 rows only.

    The unnamed first column holds the LM identifier (e.g. 'LM_0').

    Returns:
        Dataframe filtered to LM_0 rows.
    """
    if df.empty:
        return df
    first_col = df.columns[0]
    mask = df[first_col].astype(str).str.startswith("LM")
    if not mask.any():
        return df
    return df[df[first_col] == "LM_0"].reset_index(drop=True)


def summarise_episodes(df: pd.DataFrame) -> dict[str, float | int]:
    """Aggregate one block of episodes into report summary columns.

    Returns:
        Mapping with accuracy, confusion, steps, rotation error and timing.
    """
    if df is None or df.empty:
        return {
            "Correct (%)": np.nan,
            "Confused (%)": np.nan,
            "No Match (%)": np.nan,
            "Num Match Steps": np.nan,
            "Rotation Error (degrees)": np.nan,
            "Episode Run Time (s)": np.nan,
            "Num Episodes": 0,
        }
    n = len(df)
    perf = df["primary_performance"].astype(str)
    correct = perf.isin(["correct", "correct_mlh"]).mean() * 100
    confused = perf.isin(["confused", "confused_mlh"]).mean() * 100
    no_match = perf.isin(["no_match", "no_match_pose"]).mean() * 100

    rot_err_deg = pd.to_numeric(df.get("rotation_error"), errors="coerce").dropna()
    mean_rot_err = float(rot_err_deg.mean()) if not rot_err_deg.empty else np.nan

    return {
        "Correct (%)": round(float(correct), 1),
        "Confused (%)": round(float(confused), 1),
        "No Match (%)": round(float(no_match), 1),
        "Num Match Steps": round(
            float(pd.to_numeric(df["monty_matching_steps"], errors="coerce").mean()),
            1,
        ),
        "Rotation Error (degrees)": (
            round(mean_rot_err, 1) if not np.isnan(mean_rot_err) else np.nan
        ),
        "Episode Run Time (s)": round(
            float(pd.to_numeric(df["time"], errors="coerce").mean()), 2
        ),
        "Num Episodes": int(n),
    }


def summarise_surface_graphs(df: pd.DataFrame) -> dict[str, float | int]:
    """Summaries for surface unsupervised runs.

    Returns:
        Mapping with training steps and episode time.
    """
    if df is None or df.empty:
        return {
            "Num Training Steps": np.nan,
            "Episode Time (s)": np.nan,
        }
    first = df.iloc[0]
    num_steps = pd.to_numeric(first.get("num_steps"), errors="coerce")
    episode_time = pd.to_numeric(first.get("time"), errors="coerce")
    return {
        "Num Training Steps": int(num_steps) if pd.notna(num_steps) else np.nan,
        "Episode Time (s)": round(float(episode_time), 2)
        if pd.notna(episode_time)
        else np.nan,
    }


def _format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        if value.is_integer():
            return f"{int(value)}"
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def to_markdown(df: pd.DataFrame, title: str | None = None) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table.

    Hand-rolled to avoid the `tabulate` optional dependency.

    Returns:
        Markdown table string.
    """
    out = []
    if title:
        out.append(f"### {title}\n")
    if df.empty:
        out.append("_(no data)_\n")
        return "\n".join(out)
    headers = [str(c) for c in df.columns]
    rows = [[_format_cell(v) for v in row] for row in df.itertuples(index=False)]
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    return "\n".join(out)


def write_md(path: Path, sections: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sections) + "\n")

# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Exp 6 — Distant similar object discrimination."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tools.dissertation_analysis import discovery, figures, loaders, tables
from tools.dissertation_analysis.experiments import ExperimentReport


OBJECT_IDS = [
    "tbp_mug",
    "sw_mug",
    "tea_tin",
    "hexagons",
    "mc_fox",
    "cap",
    "washbag",
]


def _capture_to_object_id(label: str) -> str:
    """Map capture_00* graph labels to semantic object ids when available."""
    match = re.fullmatch(r"capture_(\d+)", label)
    if match is None:
        return label
    idx = int(match.group(1)) - 1
    if 0 <= idx < len(OBJECT_IDS):
        return OBJECT_IDS[idx]
    return label


def _object_to_capture_id(label: str) -> str:
    """Map semantic object id back to capture_00* id when known."""
    if label in OBJECT_IDS:
        return f"capture_{OBJECT_IDS.index(label) + 1:03d}"
    return label


def _render_confusion_markdown_table(matrix: pd.DataFrame) -> str:
    """Render confusion matrix as explicit markdown table."""
    columns = [str(col) for col in matrix.columns]
    index = [str(row) for row in matrix.index]
    header = ["Predicted →<br>Actual ↓"] + [
        f"`{col}` ({_object_to_capture_id(col)})" for col in columns
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]

    for row in index:
        values = []
        for col in columns:
            value = int(matrix.loc[row, col])
            values.append(f"**{value}**" if row == col else str(value))
        lines.append("| " + " | ".join([f"`{row}`"] + values) + " |")

    return "\n".join(lines)


def _load_distant_df(results_dir: Path) -> pd.DataFrame | None:
    run = discovery.find_run(results_dir, "exp6_distant_similar_eval")
    if run is None:
        return None
    df = loaders.load_csv(run, "eval")
    if df is None:
        df = loaders.load_csv(run, "train")
    if df is None:
        return None
    return tables.filter_lm_rows(df)


def _pair_confusion(df: pd.DataFrame) -> pd.DataFrame:
    """Build the confusion matrix over the pair of objects actually present.

    Object labels in the trained model are capture indices (e.g. capture_001),
    not semantic tags, so the matrix axes are derived from the data rather
    than hard-coded.

    Returns:
        Square confusion matrix indexed by target and predicted graph id.
    """
    target = df["primary_target_object"].astype(str).map(_capture_to_object_id)
    predicted = df["most_likely_object"].astype(str).map(_capture_to_object_id)
    pair = sorted(set(target) | (set(predicted) & set(target)))
    matrix = pd.crosstab(target, predicted)
    return matrix.reindex(index=pair, columns=pair, fill_value=0)


def run(results_dir: Path, output_dir: Path) -> ExperimentReport:
    out = output_dir / "exp6"
    out.mkdir(parents=True, exist_ok=True)

    _ = discovery.find_run(results_dir, "exp6_surface_similar_eval")
    distant = _load_distant_df(results_dir)

    if distant is None or distant.empty:
        return ExperimentReport(
            name="exp6",
            relative_dir="exp6",
            title="Distant Agent — Exp 6 Similar Object Discrimination",
            missing=True,
            missing_reason="no exp6_distant_similar_eval run found.",
        )

    matrix = _pair_confusion(distant)
    matrix.to_csv(out / "confusion_pairs.csv")

    sections = [
        "# Experiment 6 — Similar Object Discrimination",
        "### Distant confusion counts",
        _render_confusion_markdown_table(matrix),
    ]

    figures.heatmap(
        matrix.astype(float),
        out_path=out / "confusion_pairs.png",
        title="similar pair confusion",
        cbar_label="Episodes per target/predicted cell",
        fmt=".0f",
        cmap="Blues",
    )
    sections.append("![](confusion_pairs.png)")

    tables.write_md(out / "confusion_pairs.md", sections)
    tables.write_md(out / "summary.md", sections)
    return ExperimentReport(
        name="exp6",
        relative_dir="exp6",
        title="Distant Agent — Exp 6 Similar Object Discrimination",
        sections=sections,
        figures=["confusion_pairs.png"],
    )

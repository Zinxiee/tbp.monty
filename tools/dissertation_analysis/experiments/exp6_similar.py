# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Exp 6 — Distant similar object discrimination."""

from __future__ import annotations

import pandas as pd

from tools.dissertation_analysis import discovery, figures, loaders, tables
from tools.dissertation_analysis.experiments import ExperimentReport

PAIR = ["tbp_mug", "tea_tin"]


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
    target = df["primary_target_object"].astype(str)
    predicted = df["most_likely_object"].astype(str)
    matrix = pd.crosstab(target, predicted)
    return matrix.reindex(index=PAIR, columns=PAIR, fill_value=0)


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
        tables.to_markdown(
            matrix.reset_index().rename(columns={"index": "Target \\ Predicted"}),
            title="Distant confusion counts",
        ),
    ]

    figures.heatmap(
        matrix.astype(float),
        out_path=out / "confusion_pairs.png",
        title="Exp 6 — similar pair confusion",
        cbar_label="Episode count",
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

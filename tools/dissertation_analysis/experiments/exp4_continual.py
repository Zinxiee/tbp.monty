# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Exp 4 — Distant continual learning."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from tools.dissertation_analysis import discovery, figures, loaders, tables
from tools.dissertation_analysis.experiments import ExperimentReport

EXP4_SEQUENCE = [
    (1, "O1", "learn"),
    (2, "O2", "learn"),
    (3, "O3", "learn"),
    (4, "O4", "learn"),
    (5, "O5", "learn"),
    (6, "O1", "recall"),
    (7, "O3", "recall"),
    (8, "O5", "recall"),
    (9, "O2", "recall"),
    (10, "O4", "recall"),
]


def _load_df(results_dir: Path) -> pd.DataFrame | None:
    run = discovery.find_run(results_dir, "exp4_distant_continual")
    if run is None:
        return None
    df = loaders.load_csv(run, "train")
    if df is None:
        df = loaders.load_csv(run, "eval")
    if df is None:
        return None
    return tables.filter_lm_rows(df)


def _build_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ep_num, obj, kind in EXP4_SEQUENCE:
        if ep_num - 1 < len(df):
            row = df.iloc[ep_num - 1]
            perf = str(row.get("primary_performance", ""))
            hit = perf in ("correct", "correct_mlh")
            rows.append(
                {
                    "Episode": ep_num,
                    "Object": obj,
                    "Episode Type": kind,
                    "Recall": "hit"
                    if kind == "recall" and hit
                    else ("miss" if kind == "recall" else ""),
                    "Mean Objects Per Graph": row.get("mean_objects_per_graph"),
                    "Mean Graphs Per Object": row.get("mean_graphs_per_object"),
                    "Time (s)": row.get("time"),
                }
            )
        else:
            rows.append(
                {
                    "Episode": ep_num,
                    "Object": obj,
                    "Episode Type": kind,
                    "Recall": "",
                    "Mean Objects Per Graph": None,
                    "Mean Graphs Per Object": None,
                    "Time (s)": None,
                }
            )
    return pd.DataFrame(rows)


def _graph_growth_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(
        df["Episode"],
        pd.to_numeric(df["Mean Objects Per Graph"], errors="coerce"),
        marker="o",
        label="Mean objects per graph",
    )
    ax.plot(
        df["Episode"],
        pd.to_numeric(df["Mean Graphs Per Object"], errors="coerce"),
        marker="o",
        label="Mean graphs per object",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Value")
    ax.set_title("Exp 4 — graph growth over episodes")
    ax.grid(visible=True, alpha=0.3)
    ax.legend()
    figures.save_figure(fig, out_path)


def run(results_dir: Path, output_dir: Path) -> ExperimentReport:
    out = output_dir / "exp4"
    out.mkdir(parents=True, exist_ok=True)

    df = _load_df(results_dir)
    if df is None or df.empty:
        return ExperimentReport(
            name="exp4",
            relative_dir="exp4",
            title="Distant Agent — Exp 4 Continual Learning",
            missing=True,
            missing_reason="no exp4_distant_continual run found.",
        )

    combined = _build_table(df)
    combined.to_csv(out / "continual_summary.csv", index=False)
    combined.to_csv(out / "summary.csv", index=False)

    sections = [
        "# Experiment 4 — Continual Learning",
        tables.to_markdown(combined, title="Episode-by-episode continual learning"),
    ]

    figures_rel: list[str] = []
    recall = combined[combined["Episode Type"] == "recall"]
    if not recall.empty:
        figures.recall_strip(
            episodes=recall["Episode"].tolist(),
            objects=recall["Object"].tolist(),
            correct=[r == "hit" for r in recall["Recall"].tolist()],
            out_path=out / "recall_timeline.png",
            title="Exp 4 — recall hit/miss",
        )
        figures_rel.append("recall_timeline.png")

    _graph_growth_plot(combined, out / "graph_growth.png")
    figures_rel.append("graph_growth.png")

    sections.append("\n".join(f"![]({rel})" for rel in figures_rel))
    tables.write_md(out / "continual_summary.md", sections)
    tables.write_md(out / "summary.md", sections)
    return ExperimentReport(
        name="exp4",
        relative_dir="exp4",
        title="Distant Agent — Exp 4 Continual Learning",
        sections=sections,
        figures=figures_rel,
    )

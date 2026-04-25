# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Exp 2 — Rotation invariance for distant agent."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from tools.dissertation_analysis import discovery, figures, loaders, tables
from tools.dissertation_analysis.experiments import ExperimentReport

ORIENTATION_ORDER = ["ORI0", "ORI1", "ORI2", "ORI3", "ORI4"]


def _suffix_to_ori(suffix: str) -> str:
    digits = "".join(c for c in suffix if c.isdigit())
    if digits == "":
        return suffix or "ORI?"
    return f"ORI{digits}"


def _load_frame(run_path: Path) -> pd.DataFrame | None:
    df = loaders.load_csv(run_path, "eval")
    if df is None:
        df = loaders.load_csv(run_path, "train")
    if df is None:
        return None
    return tables.filter_lm_rows(df)


def _collect(results_dir: Path) -> pd.DataFrame:
    rows = []
    baseline = discovery.find_run(results_dir, "exp1_distant_eval")
    if baseline is not None:
        df = _load_frame(baseline)
        if df is not None:
            rows.append({"Orientation": "ORI0", **tables.summarise_episodes(df)})

    for run in discovery.find_runs(results_dir, "exp2_distant_eval_rot"):
        df = _load_frame(run.path)
        if df is None:
            continue
        rows.append(
            {"Orientation": _suffix_to_ori(run.suffix), **tables.summarise_episodes(df)}
        )

    combined = pd.DataFrame(rows)
    if combined.empty:
        return combined
    combined["Orientation"] = pd.Categorical(
        combined["Orientation"], categories=ORIENTATION_ORDER, ordered=True
    )
    return combined.sort_values("Orientation").reset_index(drop=True)


def _line_plot(
    df: pd.DataFrame, *, y: str, ylabel: str, title: str, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    plot_df = df.dropna(subset=[y]).copy()
    ax.plot(
        plot_df["Orientation"].astype(str),
        pd.to_numeric(plot_df[y], errors="coerce"),
        marker="o",
        color="tab:blue",
    )
    ax.set_xlabel("Orientation")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(visible=True, alpha=0.3)
    figures.save_figure(fig, out_path)


def run(results_dir: Path, output_dir: Path) -> ExperimentReport:
    out = output_dir / "exp2"
    out.mkdir(parents=True, exist_ok=True)

    combined = _collect(results_dir)
    if combined.empty:
        return ExperimentReport(
            name="exp2",
            relative_dir="exp2",
            title="Distant Agent — Exp 2 Rotation Invariance",
            missing=True,
            missing_reason="no distant rotation runs found.",
        )

    combined.to_csv(out / "accuracy_vs_orientation.csv", index=False)

    sections = [
        "# Experiment 2 — Rotation Invariance",
        tables.to_markdown(
            combined, title="Distant accuracy and rotation error by orientation"
        ),
    ]

    figures_rel: list[str] = []
    _line_plot(
        combined,
        y="Correct (%)",
        ylabel="Correct (%)",
        title="Exp 2 — Accuracy vs orientation",
        out_path=out / "accuracy_vs_orientation.png",
    )
    figures_rel.append("accuracy_vs_orientation.png")

    _line_plot(
        combined,
        y="Rotation Error (degrees)",
        ylabel="Rotation error (degrees)",
        title="Exp 2 — Rotation error vs orientation",
        out_path=out / "rotation_error_vs_orientation.png",
    )
    figures_rel.append("rotation_error_vs_orientation.png")

    sections.append("\n".join(f"![]({rel})" for rel in figures_rel))
    tables.write_md(out / "accuracy_vs_orientation.md", sections)
    tables.write_md(out / "summary.md", sections)

    return ExperimentReport(
        name="exp2",
        relative_dir="exp2",
        title="Distant Agent — Exp 2 Rotation Invariance",
        sections=sections,
        figures=figures_rel,
    )

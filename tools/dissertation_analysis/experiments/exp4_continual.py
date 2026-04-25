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


def _parse_result_label(value: object) -> str:
    """Extract the assigned graph id from the `result` column.

    The column is heterogeneous: a bare graph id (e.g. `new_object0`),
    an outcome tag (e.g. `unknown_object_not_matched_(TN)`), or a stringified
    Python list (e.g. `['new_object1']`).

    Returns:
        The underlying graph id when it can be identified, else the original
        string.
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return text
        first = inner.split(",", 1)[0].strip().strip("'\"")
        return first or text
    return text


def _build_table(df: pd.DataFrame) -> pd.DataFrame:
    """Walk every episode in `df`, label epoch and hit/miss from the data.

    The first occurrence of each `primary_target_object` is treated as the
    learn episode for that object; subsequent occurrences are recall episodes.
    For learn episodes the assigned graph is taken from `result`, since the
    `most_likely_object` field still points at the best pre-existing match
    even when a new graph is seeded. Recall episodes count as a hit when
    `most_likely_object` matches the learn-episode assignment.

    Returns:
        DataFrame with one row per episode and columns Episode, Epoch, Object,
        Episode Type, Recall, Predicted, mean_objects/graphs metrics, Time.
    """
    rows = []
    first_assignment: dict[str, str] = {}
    seen_objects: list[str] = []
    epoch = 1
    episodes_per_epoch: int | None = None
    episode_in_epoch = 0

    for ep_num, (_, row) in enumerate(df.iterrows(), start=1):
        target = str(row.get("primary_target_object", ""))
        predicted = str(row.get("most_likely_object", ""))
        assigned = _parse_result_label(row.get("result"))

        if target not in first_assignment:
            first_assignment[target] = assigned
            seen_objects.append(target)
            kind = "learn"
            recall = ""
        else:
            kind = "recall"
            expected = first_assignment[target]
            recall = "hit" if predicted == expected else "miss"

        epoch_started = (
            episodes_per_epoch is None
            and len(seen_objects) > 1
            and target == seen_objects[0]
        )
        if epoch_started:
            episodes_per_epoch = ep_num - 1
        if episodes_per_epoch is not None:
            episode_in_epoch += 1
            if episode_in_epoch > episodes_per_epoch:
                epoch += 1
                episode_in_epoch = 1
        else:
            episode_in_epoch = ep_num

        rows.append(
            {
                "Episode": ep_num,
                "Epoch": epoch,
                "Object": target,
                "Episode Type": kind,
                "Recall": recall,
                "Predicted": predicted,
                "Mean Objects Per Graph": row.get("mean_objects_per_graph"),
                "Mean Graphs Per Object": row.get("mean_graphs_per_object"),
                "Time (s)": row.get("time"),
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

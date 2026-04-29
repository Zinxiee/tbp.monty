# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Exp 1 — Baseline feasibility for distant agent."""

from __future__ import annotations

import pandas as pd

from tbp.monty.frameworks.utils.plot_utils_dev import plot_graph
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


def _resolve_eval_run(results_dir):
    run = discovery.find_run(results_dir, "exp1_distant_eval")
    if run is not None:
        return run
    return discovery.find_run(results_dir, "exp1_distant_train")


def _load_eval_frame(results_dir):
    run = _resolve_eval_run(results_dir)
    if run is None:
        return None, None
    df = loaders.load_csv(run, "eval")
    if df is None:
        df = loaders.load_csv(run, "train")
    if df is None:
        return None, None
    return tables.filter_lm_rows(df), run


def _per_object_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for object_name, object_df in df.groupby("primary_target_object", sort=True):
        rows.append({"Object": object_name, **tables.summarise_episodes(object_df)})
    return pd.DataFrame(rows)


def _save_learned_graphs(
    results_dir,
    out_dir,
    *,
    rotation: float = -70,
    pitch: float | None = None,
    yaw: float = 0,
    roll: float = 0,
    axis_labels: tuple[str, str, str] = ("y", "x", "z"),
) -> list[str]:
    model_run = discovery.find_run(results_dir, "exp1_distant_eval")
    if model_run is None:
        model_run = discovery.find_run(results_dir, "exp1_distant_train")
    if model_run is None:
        return []

    state = loaders.load_model(model_run, epoch=0)
    if state is None:
        return []

    lm_dict = state.get("lm_dict", {})
    lm_state = lm_dict.get(0) if isinstance(lm_dict, dict) else None
    if lm_state is None and lm_dict:
        lm_state = next(iter(lm_dict.values()))
    if lm_state is None:
        return []

    graph_memory = lm_state.get("graph_memory", {})
    figure_paths: list[str] = []
    for object_name in sorted(graph_memory):
        object_id = object_name
        if object_name.startswith("capture_"):
            capture_idx = object_name.rsplit("_", maxsplit=1)[-1]
            if capture_idx.isdigit():
                zero_based_idx = int(capture_idx) - 1
                if 0 <= zero_based_idx < len(OBJECT_IDS):
                    object_id = OBJECT_IDS[zero_based_idx]

        entry = graph_memory[object_name]
        graph = entry.get("patch") if isinstance(entry, dict) else entry
        if graph is None:
            continue
        fig = plot_graph(
            graph,
            rotation=rotation,
            show_nodes=True,
            show_edges=False,
            show_axticks=False,
        )
        if fig.axes:
            ax = fig.axes[0]
            elev = rotation if pitch is None else pitch
            try:
                ax.view_init(elev=elev, azim=180 + yaw, roll=roll)
            except TypeError:
                ax.view_init(elev=elev, azim=180 + yaw)
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])
            ax.set_zlabel(axis_labels[2])
            # Keep title outside 3D axes to avoid overlap with projected labels.
            ax.set_title("")
            fig.suptitle(object_id, y=0.98)
            fig.subplots_adjust(top=0.9)
        figure_path = out_dir / "learned_graphs" / f"{object_name}.png"
        figures.save_figure(fig, figure_path)
        figure_paths.append(f"learned_graphs/{object_name}.png")
    return figure_paths


def run(results_dir, output_dir) -> ExperimentReport:
    out = output_dir / "exp1"
    out.mkdir(parents=True, exist_ok=True)

    distant_df, distant_run = _load_eval_frame(results_dir)
    if distant_df is None or distant_run is None:
        return ExperimentReport(
            name="exp1",
            relative_dir="exp1",
            title="Distant Agent: Exp 1 Baseline",
            missing=True,
            missing_reason="no exp1_distant_eval run found.",
        )

    per_object = _per_object_summary(distant_df)
    overall = pd.DataFrame([tables.summarise_episodes(distant_df)])
    per_object.to_csv(out / "per_object.csv", index=False)
    overall.to_csv(out / "summary.csv", index=False)

    figures_rel: list[str] = []
    if not per_object.empty:
        per_object_for_plot = per_object.copy()
        per_object_for_plot["Object"] = OBJECT_IDS[: len(per_object_for_plot)]
        figures.grouped_bar(
            per_object_for_plot.assign(Agent="distant"),
            x="Object",
            y="Correct (%)",
            hue="Agent",
            out_path=out / "accuracy_per_object.png",
            ylabel="Correct (%)",
            title="Recognition accuracy by object",
            ylim=(0, 100),
            color="green",
        )
        figures_rel.append("accuracy_per_object.png")

    correct_mask = distant_df["primary_performance"].astype(str).isin(
        ["correct", "correct_mlh"]
    )
    rot_deg = pd.to_numeric(
        distant_df.loc[correct_mask, "rotation_error"], errors="coerce"
    )
    rot_df = pd.DataFrame(
        {
            "Agent": ["distant"] * len(rot_deg),
            "rot_err_deg": rot_deg.to_numpy(),
        }
    ).dropna()
    if not rot_df.empty:
        figures.histogram_per_agent(
            rot_df,
            column="rot_err_deg",
            hue="Agent",
            out_path=out / "rotation_error_hist.png",
            xlabel="Rotation error (degrees)",
            title="Rotation error on correct episodes",
        )
        figures_rel.append("rotation_error_hist.png")

    figures_rel.extend(
        _save_learned_graphs(
            results_dir,
            out,
            pitch=-70,
            yaw=-10,
            roll=12,
            axis_labels=("y", "x", "z"),
        )
    )

    summary_md = [
        "# Experiment 1 — Baseline Feasibility",
        tables.to_markdown(overall, title="Overall distant-agent performance"),
        tables.to_markdown(per_object, title="Per-object distant performance"),
        "See [per_object.md](per_object.md).",
    ]
    if figures_rel:
        summary_md.append("\n".join(f"![]({rel})" for rel in figures_rel))

    per_object_md = [
        "# Experiment 1 — Per-object distant performance",
        tables.to_markdown(per_object, title="Per-object distant performance"),
    ]
    tables.write_md(out / "summary.md", summary_md)
    tables.write_md(out / "per_object.md", per_object_md)

    return ExperimentReport(
        name="exp1",
        relative_dir="exp1",
        title="Distant Agent: Exp 1 Baseline",
        sections=summary_md,
        figures=figures_rel,
    )

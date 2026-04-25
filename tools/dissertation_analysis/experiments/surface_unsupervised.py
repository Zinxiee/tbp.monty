# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Surface-agent unsupervised training-feasibility analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tbp.monty.frameworks.utils.plot_utils_dev import plot_graph

from tools.dissertation_analysis import discovery, figures, loaders, tables
from tools.dissertation_analysis.experiments import ExperimentReport


def _to_numpy(array) -> np.ndarray:
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _graph_from_state(state: dict) -> object | None:
    lm_dict = state.get("lm_dict", {})
    lm_state = lm_dict.get(0) if isinstance(lm_dict, dict) else None
    if lm_state is None and lm_dict:
        lm_state = next(iter(lm_dict.values()))
    if lm_state is None:
        return None
    graph_memory = lm_state.get("graph_memory", {})
    if not graph_memory:
        return None
    first_key = sorted(graph_memory)[0]
    entry = graph_memory[first_key]
    return entry.get("patch") if isinstance(entry, dict) else entry


def _graph_stats(graph) -> dict[str, float | int]:
    pos = _to_numpy(graph.pos)
    point_count = int(pos.shape[0])
    bbox = pos.max(axis=0) - pos.min(axis=0)
    bbox_mm = bbox * 1000.0
    bbox_volume = float(np.prod(bbox_mm))

    if point_count < 2:
        mean_nn = 0.0
    else:
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf)
        mean_nn = float(np.min(dist, axis=1).mean() * 1000.0)

    return {
        "Point Count": point_count,
        "BBox X (mm)": round(float(bbox_mm[0]), 2),
        "BBox Y (mm)": round(float(bbox_mm[1]), 2),
        "BBox Z (mm)": round(float(bbox_mm[2]), 2),
        "BBox Volume (mm^3)": round(bbox_volume, 2),
        "Mean Inter-point Distance (mm)": round(mean_nn, 2),
    }


def _run_tag(path: Path) -> str:
    return path.name.replace("real_world_lite6_maixsense_unsupervised_", "")


def _parse_tag(tag: str) -> tuple[str, int]:
    for prefix in ("washbag", "CBOX", "MCFOX", "CAP"):
        if tag.startswith(prefix):
            suffix = tag[len(prefix) :]
            return prefix, int(suffix) if suffix.isdigit() else 1
    return tag, 1


def _save_graph_figure(graph, out_path: Path, title: str) -> None:
    fig = plot_graph(graph)
    if fig.axes:
        fig.axes[0].set_title(title)
    figures.save_figure(fig, out_path)


def _repeatability_plot(df: pd.DataFrame, out_path: Path) -> None:
    objects = ["washbag", "CBOX"]
    fig, axes = plt.subplots(len(objects), 1, figsize=(7.2, 6.5), sharex=False)
    if len(objects) == 1:
        axes = [axes]
    for ax, object_name in zip(axes, objects):
        subset = df[df["Object"] == object_name].sort_values("Repeat")
        if subset.empty:
            ax.set_axis_off()
            continue
        x = np.arange(len(subset))
        width = 0.35
        ax.bar(
            x - width / 2,
            subset["Point Count"],
            width=width,
            color="#1f77b4",
            label="Point count",
            edgecolor="black",
        )
        ax.set_ylabel("Point count")
        ax.set_title(object_name)
        ax.set_xticks(x)
        ax.set_xticklabels(subset["Run Tag"], rotation=0)
        ax2 = ax.twinx()
        ax2.plot(
            x + width / 2,
            subset["BBox Volume (mm^3)"],
            color="#d62728",
            marker="o",
            label="BBox volume",
        )
        ax2.set_ylabel("BBox volume (mm^3)")
        ax.grid(visible=True, axis="y", alpha=0.25)
    fig.tight_layout()
    figures.save_figure(fig, out_path)


def run(results_dir: Path, output_dir: Path) -> ExperimentReport:
    out = output_dir / "surface_unsupervised"
    out.mkdir(parents=True, exist_ok=True)

    grouped = discovery.find_surface_unsupervised_runs(results_dir)
    runs = [path for paths in grouped.values() for path in paths]
    if not runs:
        return ExperimentReport(
            name="surface_unsupervised",
            relative_dir="surface_unsupervised",
            title="Surface Agent — Training Feasibility",
            missing=True,
            missing_reason="no surface unsupervised runs found.",
        )

    summary_rows = []
    repeat_rows = []
    figure_paths: list[str] = []

    for object_name in ("washbag", "CBOX", "MCFOX", "CAP"):
        for run_path in grouped.get(object_name, []):
            tag = _run_tag(run_path)
            base, repeat = _parse_tag(tag)
            df = loaders.load_csv(run_path, "train")
            if df is None or df.empty:
                continue
            df = tables.filter_lm_rows(df)
            state = loaders.load_model(run_path, epoch=0)
            if state is None:
                continue
            graph = _graph_from_state(state)
            if graph is None:
                continue

            stats = _graph_stats(graph)
            summary_rows.append(
                {
                    "Run Tag": tag,
                    "Object": base,
                    "Repeat": repeat,
                    **tables.summarise_surface_graphs(df),
                    **stats,
                    "Result": str(
                        df.iloc[0].get("result", "unknown_object_not_matched_(TN)")
                    ),
                }
            )
            repeat_rows.append(
                {
                    "Run Tag": tag,
                    "Object": base,
                    "Repeat": repeat,
                    "Point Count": stats["Point Count"],
                    "BBox X (mm)": stats["BBox X (mm)"],
                    "BBox Y (mm)": stats["BBox Y (mm)"],
                    "BBox Z (mm)": stats["BBox Z (mm)"],
                    "BBox Volume (mm^3)": stats["BBox Volume (mm^3)"],
                    "Mean Inter-point Distance (mm)": stats[
                        "Mean Inter-point Distance (mm)"
                    ],
                    "Time (s)": round(float(df.iloc[0].get("time", np.nan)), 2),
                }
            )

            _save_graph_figure(
                graph,
                out / "learned_graphs" / f"{tag}.png",
                title=f"{base} / {tag}",
            )
            figure_paths.append(f"learned_graphs/{tag}.png")

    summary = pd.DataFrame(summary_rows).sort_values(["Object", "Repeat"])
    repeatability = pd.DataFrame(repeat_rows).sort_values(["Object", "Repeat"])
    summary.to_csv(out / "summary.csv", index=False)
    repeatability.to_csv(out / "repeatability.csv", index=False)

    summary_md = [
        "# Surface Agent — Training Feasibility",
        tables.to_markdown(summary, title="Per-run graph summary"),
        "## Learned graphs",
        "\n".join(f"![]({rel})" for rel in figure_paths),
    ]
    tables.write_md(out / "summary.md", summary_md)

    repeat_md = ["# Surface Agent — Limitations: Sensor Noise & Repeatability"]
    for object_name in ("washbag", "CBOX"):
        subset = repeatability[repeatability["Object"] == object_name].copy()
        if subset.empty:
            continue
        subset = subset[
            [
                "Run Tag",
                "Point Count",
                "BBox X (mm)",
                "BBox Y (mm)",
                "BBox Z (mm)",
                "BBox Volume (mm^3)",
                "Mean Inter-point Distance (mm)",
                "Time (s)",
            ]
        ]
        delta = {
            "Run Tag": "Δ max-min",
            "Point Count": int(
                subset["Point Count"].max() - subset["Point Count"].min()
            ),
            "BBox X (mm)": round(
                float(subset["BBox X (mm)"].max() - subset["BBox X (mm)"].min()), 2
            ),
            "BBox Y (mm)": round(
                float(subset["BBox Y (mm)"].max() - subset["BBox Y (mm)"].min()), 2
            ),
            "BBox Z (mm)": round(
                float(subset["BBox Z (mm)"].max() - subset["BBox Z (mm)"].min()), 2
            ),
            "BBox Volume (mm^3)": round(
                float(
                    subset["BBox Volume (mm^3)"].max()
                    - subset["BBox Volume (mm^3)"].min()
                ),
                2,
            ),
            "Mean Inter-point Distance (mm)": round(
                float(
                    subset["Mean Inter-point Distance (mm)"].max()
                    - subset["Mean Inter-point Distance (mm)"].min()
                ),
                2,
            ),
            "Time (s)": round(
                float(subset["Time (s)"].max() - subset["Time (s)"].min()), 2
            ),
        }
        repeat_md.extend(
            [
                f"## {object_name}",
                tables.to_markdown(subset, title=f"{object_name} repeat runs"),
                tables.to_markdown(pd.DataFrame([delta]), title=f"{object_name} delta"),
            ]
        )
    tables.write_md(out / "repeatability.md", repeat_md)

    _repeatability_plot(repeatability, out / "repeatability.png")

    return ExperimentReport(
        name="surface_unsupervised",
        relative_dir="surface_unsupervised",
        title="Surface Agent — Training Feasibility and Limitations",
        sections=summary_md + repeat_md,
        figures=[*figure_paths, "repeatability.png"],
    )

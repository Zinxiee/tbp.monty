# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Exp 2 — Rotation invariance broken down by object."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _bootstrap_repo_path() -> Path:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[3]
    for path in (repo_root / "src", repo_root):
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
    return repo_root


REPO_ROOT = _bootstrap_repo_path()

from tools.dissertation_analysis import discovery, figures, loaders, tables
from tools.dissertation_analysis.experiments import ExperimentReport


ORIENTATION_ORDER = ["ORI0", "ORI1", "ORI2", "ORI3", "ORI4"]
OBJECT_LABELS = {
    "capture_001": "tbp_mug",
    "capture_002": "sw_mug",
    "capture_003": "tea_tin",
    "capture_004": "hexagons",
    "capture_005": "mc_fox",
    "capture_006": "cap",
    "capture_007": "washbag",
}
OBJECT_ORDER = [OBJECT_LABELS[key] for key in OBJECT_LABELS]


def _suffix_to_ori(suffix: str) -> str:
    digits = "".join(c for c in suffix if c.isdigit())
    if digits == "":
        return suffix or "ORI?"
    return f"ORI{digits}"


def _object_label(object_id: str) -> str:
    return OBJECT_LABELS.get(object_id, object_id)


def _load_frame(run_path: Path) -> pd.DataFrame | None:
    df = loaders.load_csv(run_path, "eval")
    if df is None:
        df = loaders.load_csv(run_path, "train")
    if df is None:
        return None
    return tables.filter_lm_rows(df)


def _prepare_frame(df: pd.DataFrame, *, orientation: str) -> pd.DataFrame:
    frame = df.copy()
    frame["Object ID"] = frame["primary_target_object"].astype(str)
    frame["Object"] = frame["Object ID"].map(_object_label)
    frame["Orientation"] = orientation
    return frame


def _summarise_by_object_and_orientation(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (object_id, object_name, orientation), group in df.groupby(
        ["Object ID", "Object", "Orientation"], sort=False
    ):
        performance = group["primary_performance"].astype(str)
        correct = performance.isin(["correct", "correct_mlh"]).sum()
        n = int(len(group))
        accuracy = correct / n * 100 if n else np.nan
        rows.append(
            {
                "Object": object_name,
                "Object ID": object_id,
                "Orientation": orientation,
                "Correct episodes": int(correct),
                "Num episodes": n,
                "Accuracy (%)": round(float(accuracy), 1) if n else np.nan,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    summary["Object"] = pd.Categorical(
        summary["Object"], categories=OBJECT_ORDER, ordered=True
    )
    summary["Orientation"] = pd.Categorical(
        summary["Orientation"], categories=ORIENTATION_ORDER, ordered=True
    )
    return summary.sort_values(["Object", "Orientation"]).reset_index(drop=True)


def _plot_object(summary: pd.DataFrame, *, object_name: str, out_path: Path) -> None:
    object_df = summary[summary["Object"] == object_name].copy()
    object_df = object_df.set_index("Orientation").reindex(ORIENTATION_ORDER)

    x = np.arange(len(ORIENTATION_ORDER))
    y = pd.to_numeric(object_df["Accuracy (%)"], errors="coerce").to_numpy(dtype=float)
    mask = ~np.isnan(y)

    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.plot(x[mask], y[mask], marker="o", linewidth=1.8, color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(ORIENTATION_ORDER)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Orientation")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(object_name)
    ax.grid(visible=True, alpha=0.3)
    ax.text(
        0.02,
        0.04,
        f"n={int(object_df['Num episodes'].sum(skipna=True))}",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
    )
    figures.save_figure(fig, out_path)


def _collect(results_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    baseline = discovery.find_run(results_dir, "exp1_distant_eval")
    if baseline is None:
        baseline = discovery.find_run(results_dir, "exp1_distant_train")
    if baseline is not None:
        df = _load_frame(baseline)
        if df is not None:
            frames.append(_prepare_frame(df, orientation="ORI0"))

    for run in discovery.find_runs(results_dir, "exp2_distant_eval_rot"):
        df = _load_frame(run.path)
        if df is None:
            continue
        frames.append(_prepare_frame(df, orientation=_suffix_to_ori(run.suffix)))

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    return _summarise_by_object_and_orientation(combined)


def run(results_dir: Path, output_dir: Path) -> ExperimentReport:
    out = output_dir / "exp2_individual_objects"
    out.mkdir(parents=True, exist_ok=True)

    summary = _collect(results_dir)
    if summary.empty:
        return ExperimentReport(
            name="exp2_individual_objects",
            relative_dir="exp2_individual_objects",
            title="Distant Agent — Exp 2 Rotation Invariance by Object",
            missing=True,
            missing_reason="no baseline or rotation runs found.",
        )

    summary.to_csv(out / "accuracy_vs_orientation_by_object.csv", index=False)

    sections = [
        "# Experiment 2 — Rotation Invariance by Object",
        "Accuracy is episode-level accuracy from `primary_performance` in {correct, correct_mlh}. Object IDs use the same capture-to-object mapping as exp1.",
        tables.to_markdown(summary, title="Per-object accuracy by orientation"),
    ]

    figures_rel: list[str] = []
    for object_name in OBJECT_ORDER:
        object_df = summary[summary["Object"] == object_name]
        if object_df.empty:
            continue
        figure_rel = f"by_object/{object_name}.png"
        _plot_object(summary, object_name=object_name, out_path=out / figure_rel)
        figures_rel.append(figure_rel)
        sections.append(f"### {object_name}\n\n![]({figure_rel})")

    tables.write_md(out / "accuracy_vs_orientation_by_object.md", sections)

    return ExperimentReport(
        name="exp2_individual_objects",
        relative_dir="exp2_individual_objects",
        title="Distant Agent — Exp 2 Rotation Invariance by Object",
        sections=sections,
        figures=figures_rel,
        summary_path="accuracy_vs_orientation_by_object.md",
    )


def _write_figures_only(results_dir: Path, output_dir: Path) -> int:
    out = output_dir / "exp2_individual_objects"
    out.mkdir(parents=True, exist_ok=True)

    summary = _collect(results_dir)
    if summary.empty:
        return 0

    figure_count = 0
    for object_name in OBJECT_ORDER:
        object_df = summary[summary["Object"] == object_name]
        if object_df.empty:
            continue
        figure_rel = f"by_object/{object_name}.png"
        _plot_object(summary, object_name=object_name, out_path=out / figure_rel)
        figure_count += 1
    return figure_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "experiment_results",
        help="Directory containing exp1/exp2 run folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "dissertation" / "analysis",
        help="Where to write plots.",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Write only object-specific graphs, skip markdown and CSV outputs.",
    )
    args = parser.parse_args()

    if args.figures_only:
        figure_count = _write_figures_only(args.results_dir, args.output_dir)
        print(f"wrote {figure_count} figure(s)")
        return

    report = run(args.results_dir, args.output_dir)
    print(f"wrote {len(report.figures)} figure(s)")


if __name__ == "__main__":
    main()
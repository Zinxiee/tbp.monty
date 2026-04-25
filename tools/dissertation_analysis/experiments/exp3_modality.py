# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Exp 3 — Modality discussion using distant and surface outputs."""

from __future__ import annotations

import pandas as pd

from tools.dissertation_analysis import figures, tables
from tools.dissertation_analysis.experiments import ExperimentReport


def _surface_note(surface_dir: Path) -> str:
    summary_csv = surface_dir / "summary.csv"
    repeatability_csv = surface_dir / "repeatability.csv"
    if not summary_csv.exists() or not repeatability_csv.exists():
        return (
            "Surface summary unavailable. Run `python -m tools.dissertation_analysis "
            "--experiments surface_unsupervised` first."
        )

    repeatability = pd.read_csv(repeatability_csv)
    notes = []
    for object_name in ["washbag", "CBOX", "MCFOX", "CAP"]:
        rows = repeatability[repeatability["Object"] == object_name]
        if rows.empty:
            continue
        volume_spread = (
            rows["BBox Volume (mm^3)"].max() - rows["BBox Volume (mm^3)"].min()
        )
        point_spread = rows["Point Count"].max() - rows["Point Count"].min()
        notes.append(
            f"{object_name}: bbox-volume spread {volume_spread:.1f} mm^3, "
            f"point-count spread {point_spread:.0f}."
        )

    if not notes:
        return "Surface runs loaded, but repeatability summary was empty."
    return " ".join(notes)


def run(results_dir: Path, output_dir: Path) -> ExperimentReport:  # noqa: ARG001
    out = output_dir / "exp3"
    out.mkdir(parents=True, exist_ok=True)
    exp1_csv = output_dir / "exp1" / "per_object.csv"
    surface_dir = output_dir / "surface_unsupervised"

    if not exp1_csv.exists() or not (surface_dir / "summary.csv").exists():
        return ExperimentReport(
            name="exp3",
            relative_dir="exp3",
            title="Modality Discussion (Exp 3)",
            missing=True,
            missing_reason="requires exp1 and surface_unsupervised outputs.",
        )

    df = pd.read_csv(exp1_csv)
    table_df = df[
        ["Object", "Correct (%)", "Num Match Steps", "Episode Run Time (s)"]
    ].copy()
    table_df = table_df.sort_values("Object").reset_index(drop=True)
    table_df.to_csv(out / "modality_comparison.csv", index=False)

    note = _surface_note(surface_dir)
    sections = [
        "# Experiment 3 — Modality Discussion",
        tables.to_markdown(table_df, title="Distant per-object performance"),
        "## Surface note",
        note,
    ]

    figures_rel: list[str] = []
    if not table_df.empty:
        figures.scatter_labeled(
            table_df,
            x="Num Match Steps",
            y="Correct (%)",
            label="Object",
            out_path=out / "complementarity.png",
            title="Exp 3 — Distant accuracy vs matching steps",
            xlabel="Average LM steps",
            ylabel="Correct (%)",
        )
        figures_rel.append("complementarity.png")
        sections.append(f"![]({figures_rel[-1]})")

    tables.write_md(out / "modality_comparison.md", sections)
    tables.write_md(out / "summary.md", sections)
    return ExperimentReport(
        name="exp3",
        relative_dir="exp3",
        title="Modality Discussion (Exp 3)",
        sections=sections,
        figures=figures_rel,
    )

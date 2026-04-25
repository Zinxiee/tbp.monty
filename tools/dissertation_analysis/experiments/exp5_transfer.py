# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Exp 5 — Discussion-only cross-agent transfer section."""

from __future__ import annotations

from tools.dissertation_analysis import tables
from tools.dissertation_analysis.experiments import ExperimentReport


def run(results_dir: Path, output_dir: Path) -> ExperimentReport:  # noqa: ARG001
    out = output_dir / "exp5"
    out.mkdir(parents=True, exist_ok=True)

    matrix_md = [
        "# Experiment 5 — Cross-Agent Transfer",
        "This section is discussion only. No transfer runs were executed.",
        "",
        "| Train \\ Eval | Surface Agent | Distant Agent |",
        "| --- | --- | --- |",
        "| Surface Agent | not executed | not executed |",
        "| Distant Agent | not executed | see Exp 1c baseline |",
    ]

    tables.write_md(out / "discussion.md", matrix_md)
    tables.write_md(out / "summary.md", matrix_md)
    return ExperimentReport(
        name="exp5",
        relative_dir="exp5",
        title="Cross-agent Transfer (Exp 5) — Discussion Only",
        sections=matrix_md,
        figures=[],
    )

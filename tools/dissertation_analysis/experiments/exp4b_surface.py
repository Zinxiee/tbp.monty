# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Exp 4b — Surface continual-learning stub."""

from __future__ import annotations

from tools.dissertation_analysis.experiments import ExperimentReport


def run(results_dir: Path, output_dir: Path) -> ExperimentReport:  # noqa: ARG001
    out = output_dir / "exp4b"
    out.mkdir(parents=True, exist_ok=True)
    return ExperimentReport(
        name="exp4b_surface",
        relative_dir="exp4b",
        title="Surface Agent — Exp 4b Continual Learning",
        missing=True,
        missing_reason="surface continual-learning eval not executed.",
    )

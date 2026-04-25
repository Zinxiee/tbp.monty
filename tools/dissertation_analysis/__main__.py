# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Run dissertation analysis end-to-end.

Usage:
    python -m tools.dissertation_analysis
    python -m tools.dissertation_analysis --experiments exp1,exp2
    python -m tools.dissertation_analysis --results-dir /path/to/runs \
        --output-dir /path/to/analysis
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path


def _bootstrap_repo_path() -> Path:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    for path in (repo_root / "src", repo_root):
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
    return repo_root


REPO_ROOT = _bootstrap_repo_path()

from tools.dissertation_analysis import report  # noqa: E402
from tools.dissertation_analysis.experiments import (  # noqa: E402
    exp1_baseline,
    exp2_rotation,
    exp3_modality,
    exp4_continual,
    exp5_transfer,
    exp6_similar,
    surface_unsupervised,
)

EXPERIMENT_RUNNERS = {
    "exp1": exp1_baseline.run,
    "exp2": exp2_rotation.run,
    "exp3": exp3_modality.run,
    "exp4": exp4_continual.run,
    "exp5": exp5_transfer.run,
    "exp6": exp6_similar.run,
    "surface_unsupervised": surface_unsupervised.run,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "experiment_results",
        help=(
            "Directory containing per-run output folders "
            "(default: experiment_results/)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "dissertation" / "analysis",
        help="Where to write tables, figures, and index.md.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="all",
        help=(
            "Comma-separated subset of {exp1,exp2,exp3,exp4,exp5,exp6,"
            "surface_unsupervised} or 'all'."
        ),
    )
    args = parser.parse_args()

    if args.experiments == "all":
        names = list(EXPERIMENT_RUNNERS)
    else:
        names = [n.strip() for n in args.experiments.split(",") if n.strip()]
        invalid = [n for n in names if n not in EXPERIMENT_RUNNERS]
        if invalid:
            parser.error(f"unknown experiments: {invalid}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"results_dir: {args.results_dir}")
    print(f"output_dir : {args.output_dir}")

    reports = []
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        for name in names:
            print(f"\n--- {name} ---")
            try:
                rep = EXPERIMENT_RUNNERS[name](args.results_dir, args.output_dir)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"{name} failed: {exc}", stacklevel=2)
                continue
            reports.append(rep)
            if rep.missing:
                print(f"  missing: {rep.missing_reason}")
            else:
                print(f"  wrote {len(rep.figures)} figure(s)")

    index_path = report.build_index(reports, args.output_dir)
    print(f"\nindex: {index_path}")


if __name__ == "__main__":
    main()

# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Locate dissertation experiment run directories under experiment_results/.

Convention: each dissertation YAML sets ``logging.run_name`` to a canonical
string (e.g. ``exp1_distant_eval``). Run directories may add a suffix, such as
``exp2_distant_eval_rot1`` or ``real_world_lite6_maixsense_unsupervised_CBOX2``.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass


@dataclass(frozen=True)
class RunDir:
    path: Path
    run_name: str
    suffix: str  # everything after the canonical run_name (may be "")


CANONICAL_RUN_NAMES = {
    "exp1_distant_eval": "exp1_distant_eval",
    "exp1_distant_train": "exp1_distant_train",
    "exp2_distant_eval_rot": "exp2_distant_eval_rot",
    "exp4_distant_continual": "exp4_distant_continual",
    "exp4b_surface_continual": "exp4b_surface_continual",
    "exp5_distant_to_surface_eval": "exp5_distant_to_surface_eval",
    "exp5_surface_to_distant_eval": "exp5_surface_to_distant_eval",
    "exp6_distant_similar_eval": "exp6_distant_similar_eval",
    "exp6_surface_similar_eval": "exp6_surface_similar_eval",
}


def find_runs(results_dir: Path, run_name: str) -> list[RunDir]:
    """Return every dir under results_dir whose name starts with run_name.

    Sorted by suffix to get deterministic ordering (rot1 < rot2 < rot3 < rot4).
    Emits a warning and returns [] if nothing matches.
    """
    matches: list[RunDir] = []
    if not results_dir.exists():
        warnings.warn(f"results_dir not found: {results_dir}", stacklevel=2)
        return matches

    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name == run_name or child.name.startswith(run_name + "_"):
            suffix = child.name[len(run_name) :].lstrip("_")
            matches.append(RunDir(path=child, run_name=run_name, suffix=suffix))
        elif child.name.startswith(run_name) and child.name != run_name:
            # Handle names like "exp2_surface_eval_rot1" matching prefix.
            suffix = child.name[len(run_name) :]
            matches.append(RunDir(path=child, run_name=run_name, suffix=suffix))

    if not matches:
        warnings.warn(f"no runs found for {run_name} under {results_dir}", stacklevel=2)
    return matches


def find_one_run(results_dir: Path, run_name: str) -> RunDir | None:
    """Return the first run dir matching run_name, or None."""
    runs = find_runs(results_dir, run_name)
    return runs[0] if runs else None


def find_run(results_dir: Path, run_name: str) -> Path | None:
    """Return first matching run path, or None."""
    run = find_one_run(results_dir, run_name)
    return run.path if run is not None else None


_SURFACE_TAG_PREFIXES = ("washbag", "CBOX", "MCFOX", "CAP")


def _parse_surface_tag(tag: str) -> tuple[str, int]:
    for prefix in _SURFACE_TAG_PREFIXES:
        if tag.startswith(prefix):
            suffix = tag[len(prefix) :]
            if suffix.isdigit():
                return prefix, int(suffix)
            return prefix, 1
    return tag, 1


def find_surface_unsupervised_runs(results_dir: Path) -> dict[str, list[Path]]:
    """Return surface training runs grouped by object tag."""
    grouped: dict[str, list[Path]] = {key: [] for key in _SURFACE_TAG_PREFIXES}
    if not results_dir.exists():
        warnings.warn(f"results_dir not found: {results_dir}", stacklevel=2)
        return grouped

    pattern = re.compile(r"^real_world_lite6_maixsense_unsupervised_(.+)$")
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if not match:
            continue
        tag = match.group(1)
        base, _repeat = _parse_surface_tag(tag)
        grouped.setdefault(base, []).append(child)

    for key, value in grouped.items():
        grouped[key] = sorted(value)
    return grouped

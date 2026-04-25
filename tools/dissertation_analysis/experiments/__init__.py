# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExperimentReport:
    """Per-experiment artefact bundle written to <output_dir>/exp<N>/."""

    name: str
    relative_dir: str  # e.g. "exp1"
    title: str
    sections: list[str] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)  # relative paths for index.md
    missing: bool = False
    missing_reason: str = ""

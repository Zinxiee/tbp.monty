# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Exp 4 — Distant continual learning."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from tools.dissertation_analysis import discovery, figures, loaders, tables
from tools.dissertation_analysis.experiments import ExperimentReport


CAPTURE_LABELS = [
    "tbp_mug",
    "sw_mug",
    "tea_tin",
    "hexagons",
    "mc_fox",
    "cap",
    "washbag",
]


def _display_object_label(value: object) -> str:
    """Map capture IDs to dissertation object names when possible."""
    text = str(value).strip()
    match = re.fullmatch(r"capture_0*([1-9]\d*)", text)
    if match is None:
        return text
    idx = int(match.group(1))
    if 1 <= idx <= len(CAPTURE_LABELS):
        return CAPTURE_LABELS[idx - 1]
    return text


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


def _classify(tfnp: str, is_first: bool) -> tuple[str, str, str]:
    """Return (Episode Type, Recall Outcome, New Graph?) from TFNP + first-seen.

    The TFNP column is the run logger's ground-truth tag:
    - `unknown_object_not_matched_(TN)` — no existing graph matched, a new
      graph was seeded for this target. Only meaningful on first occurrence.
    - `target_in_possible_matches_(TP)` — the matched graph genuinely belongs
      to this target. On first occurrence this means the freshly-seeded graph
      was immediately recognised; on later occurrences it is a recall hit.
    - `unknown_object_in_possible_matches_(FP)` — the system matched against
      an existing graph that does NOT belong to this target. On first
      occurrence this means the target was collapsed onto an earlier object's
      graph (no new graph seeded); on later occurrences it is a recall miss.
    """
    tag = str(tfnp)
    if "TN" in tag:
        return ("learn", "learn", "Yes")
    if "TP" in tag:
        if is_first:
            return ("learn", "learn", "Yes")
        return ("recall", "hit", "")
    if is_first:
        return ("learn", "miss", "No")
    return ("recall", "miss", "")


def _build_table(df: pd.DataFrame) -> pd.DataFrame:
    """Walk every episode in `df`, label epoch and outcome from TFNP.

    Returns:
        DataFrame with one row per episode and columns Episode, Epoch, Object,
        Episode Type, Recall, Recall Outcome, New Graph?, Predicted, TFNP,
        mean_objects/graphs metrics, Time.
    """
    rows = []
    seen_objects: list[str] = []
    epoch = 1
    episodes_per_epoch: int | None = None
    episode_in_epoch = 0

    for ep_num, (_, row) in enumerate(df.iterrows(), start=1):
        target = _display_object_label(row.get("primary_target_object", ""))
        predicted = str(row.get("most_likely_object", ""))
        tfnp = str(row.get("TFNP", ""))
        is_first = target not in seen_objects
        if is_first:
            seen_objects.append(target)

        kind, outcome, new_graph = _classify(tfnp, is_first)
        recall = "" if kind == "learn" else outcome

        epoch_started = (
            episodes_per_epoch is None
            and len(seen_objects) > 1
            and target == seen_objects[0]
            and not is_first
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
                "Recall Outcome": outcome,
                "New Graph?": new_graph,
                "Predicted": predicted,
                "TFNP": tfnp,
                "Mean Objects Per Graph": row.get("mean_objects_per_graph"),
                "Mean Graphs Per Object": row.get("mean_graphs_per_object"),
                "Time (s)": row.get("time"),
            }
        )
    return pd.DataFrame(rows)


def _short_tfnp(value: object) -> str:
    """Compact TFNP tag for the markdown table (TN/TP/FP/FN)."""
    text = str(value)
    for tag in ("TN", "TP", "FP", "FN"):
        if f"({tag})" in text:
            return tag
    return text


def _summary_paragraph(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    new_graphs = (df["New Graph?"] == "Yes").sum()
    learn_eps = (df["Episode Type"] == "learn").sum()
    recall_eps = (df["Episode Type"] == "recall").sum()
    hits = ((df["Episode Type"] == "recall") & (df["Recall Outcome"] == "hit")).sum()
    fp_first = (
        (df["Episode Type"] == "learn") & (df["New Graph?"] == "No")
    ).sum()
    accuracy = (hits / recall_eps * 100) if recall_eps else 0.0
    return (
        "### Summary\n\n"
        f"- New graphs seeded: **{int(new_graphs)} / {int(learn_eps)}** "
        "first-occurrence episodes "
        f"({int(fp_first)} first-occurrence targets were collapsed onto an "
        "existing graph rather than seeded as new).\n"
        f"- Recall accuracy (epochs ≥2): **{int(hits)} / {int(recall_eps)} "
        f"= {accuracy:.0f}%**.\n"
    )


def run(results_dir: Path, output_dir: Path) -> ExperimentReport:
    out = output_dir / "exp4"
    out.mkdir(parents=True, exist_ok=True)

    df = _load_df(results_dir)
    if df is None or df.empty:
        return ExperimentReport(
            name="Continual Learning",
            relative_dir="exp4",
            title="Distant Agent — Exp 4 Continual Learning",
            missing=True,
            missing_reason="no exp4_distant_continual run found.",
        )

    combined = _build_table(df)
    combined.to_csv(out / "continual_summary.csv", index=False)

    display = combined[
        [
            "Episode",
            "Epoch",
            "Object",
            "Episode Type",
            "Recall Outcome",
            "New Graph?",
            "Predicted",
            "TFNP",
            "Time (s)",
        ]
    ].copy()
    display["TFNP"] = display["TFNP"].map(_short_tfnp)

    sections = [
        "# Experiment 4 — Continual Learning",
        tables.to_markdown(display, title="Episode-by-episode continual learning"),
        _summary_paragraph(combined),
    ]

    figures_rel: list[str] = []
    if not combined.empty:
        outcomes = combined["Recall Outcome"].fillna("learn").astype(str).tolist()
        figures.recall_strip(
            episodes=combined["Episode"].tolist(),
            epochs=combined["Epoch"].tolist(),
            objects=combined["Object"].tolist(),
            correct=[r == "hit" for r in outcomes],
            statuses=outcomes,
            out_path=out / "recall_timeline.png",
            title="Continual Learning: learning and recall timeline",
        )
        figures_rel.append("recall_timeline.png")

    sections.append("\n".join(f"![]({rel})" for rel in figures_rel))
    tables.write_md(out / "continual_summary.md", sections)
    tables.write_md(out / "summary.md", sections)
    combined.to_csv(out / "summary.csv", index=False)
    return ExperimentReport(
        name="exp4",
        relative_dir="exp4",
        title="Distant Agent — Exp 4 Continual Learning",
        sections=sections,
        figures=figures_rel,
    )

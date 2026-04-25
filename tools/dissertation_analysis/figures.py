# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Matplotlib helpers. Save-only; no plt.show()."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AGENT_COLOURS = {"surface": "#1f77b4", "distant": "#d62728"}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_figure(fig, out_path: Path) -> None:
    _save(fig, out_path)


def grouped_bar(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str,
    out_path: Path,
    ylabel: str,
    title: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Grouped bar chart: one cluster per x value, bars coloured by hue."""
    pivot = data.pivot_table(index=x, columns=hue, values=y, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(pivot)), 4))
    pivot.plot(
        kind="bar",
        ax=ax,
        color=[AGENT_COLOURS.get(c) for c in pivot.columns],
        edgecolor="black",
        width=0.8,
    )
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(title=hue)
    plt.xticks(rotation=30, ha="right")
    _save(fig, out_path)


def line_per_agent(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str,
    out_path: Path,
    ylabel: str,
    title: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for agent, group in data.groupby(hue):
        group_sorted = group.sort_values(x)
        ax.plot(
            group_sorted[x],
            group_sorted[y],
            marker="o",
            label=agent,
            color=AGENT_COLOURS.get(agent),
        )
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(title=hue)
    ax.grid(visible=True, alpha=0.3)
    _save(fig, out_path)


def histogram_per_agent(
    data: pd.DataFrame,
    *,
    column: str,
    hue: str,
    out_path: Path,
    xlabel: str,
    title: str,
    bins: int = 20,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for agent, group in data.groupby(hue):
        ax.hist(
            group[column].dropna(),
            bins=bins,
            alpha=0.55,
            label=agent,
            color=AGENT_COLOURS.get(agent),
            edgecolor="black",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend(title=hue)
    _save(fig, out_path)


def scatter_complementarity(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    label: str,
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df[x], df[y], s=80, color="tab:purple", edgecolor="black")
    for _, row in df.iterrows():
        ax.annotate(
            row[label],
            (row[x], row[y]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    lims = [0, 100]
    ax.plot(lims, lims, "--", color="grey", alpha=0.6, label="parity")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"{x} accuracy (%)")
    ax.set_ylabel(f"{y} accuracy (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(visible=True, alpha=0.3)
    _save(fig, out_path)


def scatter_labeled(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    label: str,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(df[x], df[y], s=70, color="tab:purple", edgecolor="black")
    for _, row in df.iterrows():
        ax.annotate(
            row[label],
            (row[x], row[y]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(visible=True, alpha=0.3)
    _save(fig, out_path)


def heatmap(
    matrix: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    cbar_label: str,
    fmt: str = ".1f",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(
        figsize=(1.2 + 1.0 * len(matrix.columns), 1.0 + 0.8 * len(matrix.index))
    )
    matrix_values = matrix.to_numpy()
    im = ax.imshow(matrix_values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            v = matrix_values[i, j]
            text = "" if pd.isna(v) else format(v, fmt)
            ax.text(j, i, text, ha="center", va="center", color="white", fontsize=10)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    _save(fig, out_path)


def recall_strip(
    episodes: list[int],
    objects: list[str],
    correct: list[bool],
    *,
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(episodes)), 2.5))
    colours = ["tab:green" if c else "tab:red" for c in correct]
    ax.bar(range(len(episodes)), [1] * len(episodes), color=colours, edgecolor="black")
    ax.set_xticks(range(len(episodes)))
    ax.set_xticklabels([f"E{e}\n{o}" for e, o in zip(episodes, objects)], fontsize=9)
    ax.set_yticks([])
    ax.set_title(title)
    _save(fig, out_path)


def graph_3d(pos: np.ndarray, *, out_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=pos[:, 2], s=8, cmap="viridis")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title)
    _save(fig, out_path)

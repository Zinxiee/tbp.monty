import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _bootstrap_repo_path() -> None:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    for path in (repo_root / "src", repo_root):
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


_bootstrap_repo_path()

from tbp.monty.frameworks.models.object_model import GraphObjectModel


def load_graph(exp_dir: str, epoch: int, object_name: str) -> GraphObjectModel:
    model_path = Path(exp_dir) / str(epoch) / "model.pt"
    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        state_dict = torch.load(model_path, map_location="cpu")
    return state_dict["lm_dict"][0]["graph_memory"][object_name]["patch"]


def _to_numpy(array) -> np.ndarray:
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def get_trace_positions(graph: GraphObjectModel) -> np.ndarray:
    pos = _to_numpy(graph.pos)
    x = _to_numpy(graph.x)

    if hasattr(graph, "feature_mapping") and "node_ids" in graph.feature_mapping:
        start_idx, end_idx = graph.feature_mapping["node_ids"]
        node_ids = x[:, start_idx:end_idx].reshape(-1)
        ordered_idx = np.argsort(node_ids)
    else:
        ordered_idx = np.arange(pos.shape[0])

    return pos[ordered_idx]


def main() -> None:
    exp_dir = os.path.expanduser(
        # "~/tbp/results/monty/projects/monty_runs/real_world_lite6_maixsense_unsupervised"
        # "~/tbp/results/monty/projects/monty_runs/zed_continual_manual_train"
        "experiment_results/real_world_lite6_maixsense_unsupervised_MCFOX"
    )
    n_epochs = 1
    obj_name = "new_object0"  # Generated object ID

    graphs = [load_graph(exp_dir, epoch, obj_name) for epoch in range(n_epochs)]

    fig = plt.figure(figsize=(8, 3))
    for epoch in range(n_epochs):
        ax = fig.add_subplot(1, n_epochs, epoch + 1, projection="3d")

        pos = _to_numpy(graphs[epoch].pos)
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=pos[:, 2], s=10)

        trace_pos = get_trace_positions(graphs[epoch])
        trace_x = trace_pos[:, 0]
        trace_y = trace_pos[:, 1]
        trace_z = trace_pos[:, 2]

        ax.plot(trace_x, trace_y, trace_z, color="tab:blue", linewidth=1.5)
        ax.scatter(trace_x[0], trace_y[0], trace_z[0], c="tab:green", s=50)
        ax.scatter(trace_x[-1], trace_y[-1], trace_z[-1], c="tab:red", s=50)
        ax.text(trace_x[0], trace_y[0], trace_z[0], " Start", color="tab:green")
        ax.text(trace_x[-1], trace_y[-1], trace_z[-1], " End", color="tab:red")

        ax.set_xlabel("x [m] (Monty +X = right)")
        ax.set_ylabel("y [m] (Monty +Y = up)")
        ax.set_zlabel("z [m] (Monty +Z = backward)")
        ax.set_aspect("equal")
        ax.set_title(f"epoch {epoch} — Monty world frame (right-up-backward)")
    fig.tight_layout()
    plt.show()

    g = _to_numpy(load_graph(exp_dir, 0, "new_object0").pos)
    print("nodes:", len(g))
    print("Y range:", g[:, 1].min(), g[:, 1].max())
    print("XZ extent:", np.ptp(g[:, 0]), np.ptp(g[:, 2]))
    print("spread / extent ratio:", g.std(0))


if __name__ == "__main__":
    main()

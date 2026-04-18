# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as rot

from ufactory_api.robot_interface import RobotInterface


def make_transform(translation_xyz: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = np.asarray(translation_xyz, dtype=float)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation_matrix = transform[:3, :3]
    translation_xyz = transform[:3, 3]

    transform_inverse = np.eye(4, dtype=float)
    transform_inverse[:3, :3] = rotation_matrix.T
    transform_inverse[:3, 3] = -rotation_matrix.T @ translation_xyz
    return transform_inverse


def quaternion_wxyz_to_rotation_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion_wxyz
    return rot.from_quat([x, y, z, w]).as_matrix()


def set_equal_axes(ax: plt.Axes, all_points_xyz: np.ndarray, margin_m: float = 0.03) -> None:
    if all_points_xyz.size == 0:
        return

    min_xyz = np.min(all_points_xyz, axis=0)
    max_xyz = np.max(all_points_xyz, axis=0)
    center_xyz = (min_xyz + max_xyz) / 2.0

    ranges = max_xyz - min_xyz
    half_extent = max(0.08, float(np.max(ranges) / 2.0) + margin_m)

    ax.set_xlim(center_xyz[0] - half_extent, center_xyz[0] + half_extent)
    ax.set_ylim(center_xyz[1] - half_extent, center_xyz[1] + half_extent)
    ax.set_zlim(center_xyz[2] - half_extent, center_xyz[2] + half_extent)


def draw_frame(
    ax: plt.Axes,
    transform_ref_frame: np.ndarray,
    label: str,
    axis_length_m: float,
    origin_color: str,
    axis_colors: tuple[str, str, str] = ("r", "g", "b"),
) -> None:
    origin_xyz = transform_ref_frame[:3, 3]
    rotation_matrix = transform_ref_frame[:3, :3]

    x_axis = rotation_matrix[:, 0]
    y_axis = rotation_matrix[:, 1]
    z_axis = rotation_matrix[:, 2]

    ax.scatter(
        origin_xyz[0],
        origin_xyz[1],
        origin_xyz[2],
        color=origin_color,
        s=30,
        label=label,
    )

    ax.quiver(
        origin_xyz[0],
        origin_xyz[1],
        origin_xyz[2],
        x_axis[0],
        x_axis[1],
        x_axis[2],
        length=axis_length_m,
        normalize=True,
        color=axis_colors[0],
        linewidth=1.6,
    )
    ax.quiver(
        origin_xyz[0],
        origin_xyz[1],
        origin_xyz[2],
        y_axis[0],
        y_axis[1],
        y_axis[2],
        length=axis_length_m,
        normalize=True,
        color=axis_colors[1],
        linewidth=1.6,
    )
    ax.quiver(
        origin_xyz[0],
        origin_xyz[1],
        origin_xyz[2],
        z_axis[0],
        z_axis[1],
        z_axis[2],
        length=axis_length_m,
        normalize=True,
        color=axis_colors[2],
        linewidth=1.6,
    )

    ax.text(
        origin_xyz[0],
        origin_xyz[1],
        origin_xyz[2],
        f" {label}",
        color=origin_color,
        fontsize=9,
    )


def configure_axes(ax: plt.Axes, title: str, reference_name: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(f"X in {reference_name} [m]")
    ax.set_ylabel(f"Y in {reference_name} [m]")
    ax.set_zlabel(f"Z in {reference_name} [m]")
    ax.grid(True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Base/World/Link6/Sensor frames in robot base and world references."
    )
    parser.add_argument("--ip", default="192.168.1.159", help="Robot IP address.")
    parser.add_argument(
        "--axis-length-m",
        type=float,
        default=0.04,
        help="Length of plotted frame axes in meters.",
    )
    parser.add_argument(
        "--listener-warmup-s",
        type=float,
        default=0.12,
        help="Wait time before reading first robot state.",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Skip robot connection and use a sample home pose for visualization.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Optional path to save figure (e.g., frames.png).",
    )
    return parser.parse_args()


def build_base_link6_transform_from_robot(
    ip_address: str,
    listener_warmup_s: float,
) -> np.ndarray:
    robot = RobotInterface(ip_address)
    robot.start_listening()

    try:
        time.sleep(listener_warmup_s)
        state = robot.get_sense_state()
        end_effector = state.get("end_effector", [])
        if len(end_effector) < 6:
            raise RuntimeError(
                "Robot state missing end_effector pose; expected [x,y,z,roll,pitch,yaw]."
            )

        x_mm, y_mm, z_mm, roll_rad, pitch_rad, yaw_rad = end_effector[:6]
        translation_m = np.array([x_mm, y_mm, z_mm], dtype=float) / 1000.0
        rotation_matrix = rot.from_euler(
            "xyz", [roll_rad, pitch_rad, yaw_rad], degrees=False
        ).as_matrix()
        return make_transform(translation_m, rotation_matrix)
    finally:
        robot.stop_listening()


def build_sample_base_link6_transform() -> np.ndarray:
    # Example: home_pose_mm_deg [220, 9, 210, 180, 0, 0]
    translation_m = np.array([0.220, 0.009, 0.210], dtype=float)
    rotation_matrix = rot.from_euler("xyz", [180.0, 0.0, 0.0], degrees=True).as_matrix()
    return make_transform(translation_m, rotation_matrix)


def main() -> None:
    args = parse_args()

    if args.sample_only:
        base_link6_transform = build_sample_base_link6_transform()
    else:
        base_link6_transform = build_base_link6_transform_from_robot(
            ip_address=args.ip,
            listener_warmup_s=args.listener_warmup_s,
        )

    # From your YAML config.
    world_to_robot_translation_m = np.array([0.0, 0.0, 0.0], dtype=float)
    world_to_robot_rotation_wxyz = np.array([0.5, 0.5, -0.5, -0.5], dtype=float)

    link6_to_sensor_translation_m = np.array([0.060, 0.0, 0.0135], dtype=float)
    link6_to_sensor_rotation_wxyz = np.array([0.0, 0.0, 1.0, 0.0], dtype=float)

    base_world_transform = make_transform(
        world_to_robot_translation_m,
        quaternion_wxyz_to_rotation_matrix(world_to_robot_rotation_wxyz),
    )
    world_base_transform = invert_transform(base_world_transform)

    link6_sensor_transform = make_transform(
        link6_to_sensor_translation_m,
        quaternion_wxyz_to_rotation_matrix(link6_to_sensor_rotation_wxyz),
    )

    base_base_transform = np.eye(4, dtype=float)
    base_sensor_transform = base_link6_transform @ link6_sensor_transform

    world_world_transform = np.eye(4, dtype=float)
    world_link6_transform = world_base_transform @ base_link6_transform
    world_sensor_transform = world_base_transform @ base_sensor_transform

    # Frame origins for autoscaling.
    base_points = np.vstack(
        [
            base_base_transform[:3, 3],
            base_world_transform[:3, 3],
            base_link6_transform[:3, 3],
            base_sensor_transform[:3, 3],
        ]
    )
    world_points = np.vstack(
        [
            world_world_transform[:3, 3],
            world_base_transform[:3, 3],
            world_link6_transform[:3, 3],
            world_sensor_transform[:3, 3],
        ]
    )

    figure = plt.figure(figsize=(14, 6))

    ax_base = figure.add_subplot(1, 2, 1, projection="3d")
    configure_axes(ax_base, "Frames expressed in Robot Base", "Base")
    draw_frame(
        ax_base,
        base_world_transform,
        "WORLD",
        args.axis_length_m,
        "c",
        axis_colors=("#ff00003d", "#3bff0a37", "#4400ff44")
    )
    draw_frame(
        ax_base,
        base_base_transform,
        "BASE",
        args.axis_length_m,
        "k",
    )
    draw_frame(ax_base, base_link6_transform, "LINK6", args.axis_length_m, "m")
    draw_frame(ax_base, base_sensor_transform, "SENSOR", args.axis_length_m, "orange")
    set_equal_axes(ax_base, base_points)
    ax_base.legend(loc="upper left")

    ax_world = figure.add_subplot(1, 2, 2, projection="3d")
    configure_axes(ax_world, "Frames expressed in World / Monty", "World")
    draw_frame(
        ax_world,
        world_base_transform,
        "BASE",
        args.axis_length_m,
        "c",
        axis_colors=("#ff00003d", "#3bff0a37", "#4400ff44"),
    )
    draw_frame(
        ax_world,
        world_world_transform,
        "WORLD",
        args.axis_length_m,
        "k",
    )
    draw_frame(ax_world, world_link6_transform, "LINK6", args.axis_length_m, "m")
    draw_frame(ax_world, world_sensor_transform, "SENSOR", args.axis_length_m, "orange")
    set_equal_axes(ax_world, world_points)
    ax_world.legend(loc="upper left")

    if args.sample_only:
        mode_text = "SAMPLE MODE"
    else:
        mode_text = f"LIVE MODE | Robot IP: {args.ip}"
    figure.suptitle(mode_text, fontsize=12)

    plt.tight_layout()

    if args.save_path:
        figure.savefig(args.save_path, dpi=160, bbox_inches="tight")
        print(f"Saved figure to: {args.save_path}")

    print("T_world_sensor translation [m]:", np.round(world_sensor_transform[:3, 3], 6))
    plt.show()


if __name__ == "__main__":
    main()

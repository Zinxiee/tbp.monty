# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Real-world surface policy for the Lite6 + Maixsense A010 setup.

Provides ``RealWorldSurfacePolicy``, a subclass of ``SurfacePolicy`` that
adapts the simulated surface-agent loop to the constraints of the physical
Lite6 + Maixsense A010 rig.

Behavioral overrides layered on top of the parent policy:

1. ``_touch_sensor_id`` returns ``"patch"`` instead of ``"view_finder"``.

2. ``_touch_object`` reads forward distance from the semantically-filtered
   point cloud instead of the raw center-pixel depth.  Falls back to a
   distance-constrained arc search instead of the parent's 0.48 m-radius
   random search.

3. ``_orient_horizontal`` / ``_orient_vertical`` clamp the computed rotation
   angle to ``_MAX_ORIENT_DEG`` **before** computing compensating distances,
   and use the semantically-filtered median depth instead of the percept's
   ``mean_depth`` (which includes background pixels and inflates distances
   2-4x on real hardware).

4. ``_move_forward`` uses the same semantic-filtered depth as ``_touch_object``
   instead of the percept's ``min_depth`` feature, capped to
   ``_MAX_FORWARD_STEP_M``.

The default ``min_object_coverage`` is also lowered (0.1 -> 0.05).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import quaternion as qt

from tbp.monty.cmp import Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.actions import (
    MoveForward,
    OrientHorizontal,
    OrientVertical,
)
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_policies import (
    MotorPolicyResult,
    SurfacePolicy,
)
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState
from tbp.monty.frameworks.sensors import SensorID

logger = logging.getLogger(__name__)

# Maximum orient rotation per step (degrees). Keeps compensating translation
# distances sane: tan(15) * 0.22m = 6cm vs tan(35) * 0.22m = 15cm.
_MAX_ORIENT_DEG = 15.0

# Suppress orient corrections smaller than this. Small noise-driven angles
# produce compensating translations that accumulate into drift across cycles. (Disabled because I don't think this fixes anything)
_ORIENT_DEADBAND_DEG = 8.0

# Forward-step limits.  Even valid filtered depth can overshoot on real
# hardware when the sensor axis is slightly tilted.
_MAX_FORWARD_STEP_M = 0.05
_FALLBACK_FORWARD_STEP_M = 0.01

# Touch-object search: cap the arc radius so search steps stay small.
# Parent uses desired_object_distance * 4 = 0.48 m → 0.287 m steps.
_MAX_SEARCH_RADIUS_M = 0.10
_SEARCH_ROTATION_DEG = 15.0
_SEARCH_RANDOM_RANGE_DEG = 90.0


class RealWorldSurfacePolicy(SurfacePolicy):
    """SurfacePolicy variant for real-world setups with no view-finder sensor.

    Args:
        patch_sensor_id: Sensor ID for ``_touch_object``.
        min_object_coverage: Coverage threshold for the touch-object loop.
        **kwargs: Forwarded to ``SurfacePolicy.__init__``.
    """

    def __init__(
        self,
        *args: Any,
        patch_sensor_id: str = "patch",
        min_object_coverage: float = 0.01,
        disable_orient_compensation_translation: bool = False,
        enable_orient_decomposition_logging: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            min_object_coverage=min_object_coverage,
            **kwargs,
        )
        self._patch_sensor_id = SensorID(patch_sensor_id)
        self._last_observations: Observations | None = None
        self._disable_orient_compensation_translation = bool(
            disable_orient_compensation_translation
        )
        self._effective_max_orient_deg = _MAX_ORIENT_DEG
        self._orient_axis: str = "horizontal"
        self._enable_orient_decomposition_logging = bool(
            enable_orient_decomposition_logging
        )

    def _touch_sensor_id(self) -> SensorID:
        """Use the real sensor patch instead of the non-existent view-finder.

        Returns:
            The configured patch sensor ID.
        """
        return self._patch_sensor_id

    # ------------------------------------------------------------------
    # Exploration cycle overrides
    # ------------------------------------------------------------------

    def __call__(self, ctx, observations, state, percept, goal):
        """Stash observations, delegate, then feed LM whenever object is in view.

        Parent marks all cycle actions except OrientVertical as ``motor_only``
        and makes every ``_touch_object`` step motor-only too.  On real
        hardware this starves the LM: with noisy data the sensor frequently
        drops below ``min_object_coverage`` and all subsequent steps go
        through ``_touch_object``, so no observation ever reaches the LM and
        ``exploratory_steps`` never advances.  Here we rewrite the result
        to send data whenever the filtered depth is available — i.e., when
        the sensor actually sees the object.

        Returns:
            MotorPolicyResult, possibly with ``motor_only_step`` rewritten.
        """
        self._last_observations = observations
        result = super().__call__(ctx, observations, state, percept, goal)
        if result.motor_only_step and self._sensor_sees_object():
            result = MotorPolicyResult(
                actions=result.actions,
                motor_only_step=False,
                telemetry=result.telemetry,
                goal_pose=result.goal_pose,
            )
        return result

    def _sensor_sees_object(self) -> bool:
        """True when filtered (semantic) depth is available from last obs.

        Returns:
            ``True`` when the last observation yields a valid filtered depth.
        """
        return self._filtered_forward_depth_from_stashed_obs() is not None

    def _orient_horizontal(
        self, state: MotorSystemState, percept: Message
    ) -> OrientHorizontal:
        """Orient horizontally with dead-banded clamped angle + object depth.

        Returns:
            OrientHorizontal action.
        """
        self._orient_axis = "horizontal"
        raw_angle = self.orienting_angle_from_normal(
            orienting="horizontal",
            state=state,
            percept=percept,
        )
        rotation_degrees = self._shape_orient_angle(raw_angle)
        left_distance, forward_distance = self._compensating_distances(
            rotation_degrees, percept
        )
        self._log_orient_diagnostics(
            "horizontal",
            raw_angle=raw_angle,
            rotation_degrees=rotation_degrees,
            lateral=left_distance,
            forward=forward_distance,
            state=state,
            percept=percept,
        )
        return OrientHorizontal(
            agent_id=self.agent_id,
            rotation_degrees=rotation_degrees,
            left_distance=left_distance,
            forward_distance=forward_distance,
        )

    def _orient_vertical(
        self, state: MotorSystemState, percept: Message
    ) -> OrientVertical:
        """Orient vertically with coverage-aware throttling.

        When ``object_coverage`` is below 0.5 the sensor is likely near the
        table boundary — skip the correction entirely.  Between 0.5 and 0.8,
        scale the maximum orient angle proportionally to avoid large
        downward displacements.

        Returns:
            OrientVertical action.
        """
        coverage = percept.get_feature_by_name("object_coverage")
        if coverage is not None and coverage < 0.5:
            logger.info("orient_vertical: SKIPPED (coverage=%.2f < 0.5)", coverage)
            return OrientVertical(
                agent_id=self.agent_id,
                rotation_degrees=0.0,
                down_distance=0.0,
                forward_distance=0.0,
            )

        if coverage is not None and coverage < 0.8:
            self._effective_max_orient_deg = _MAX_ORIENT_DEG * coverage
        else:
            self._effective_max_orient_deg = _MAX_ORIENT_DEG

        self._orient_axis = "vertical"
        raw_angle = self.orienting_angle_from_normal(
            orienting="vertical",
            state=state,
            percept=percept,
        )
        rotation_degrees = self._shape_orient_angle(raw_angle)
        down_distance, forward_distance = self._compensating_distances(
            rotation_degrees, percept
        )
        self._log_orient_diagnostics(
            "vertical",
            raw_angle=raw_angle,
            rotation_degrees=rotation_degrees,
            lateral=down_distance,
            forward=forward_distance,
            state=state,
            percept=percept,
        )
        # Reset to default for subsequent horizontal orients.
        self._effective_max_orient_deg = _MAX_ORIENT_DEG
        return OrientVertical(
            agent_id=self.agent_id,
            rotation_degrees=rotation_degrees,
            down_distance=down_distance,
            forward_distance=forward_distance,
        )

    def tangential_direction(
        self,
        ctx: RuntimeContext,
        state: MotorSystemState,
        percept: Message,
    ) -> tuple[float, float, float]:
        """Bias tangential movement away from table when near boundary.

        When ``object_coverage`` drops below a threshold, the sensor is
        likely near the table edge (y_min rejections).  Forcing the
        world-Y component of the tangential direction positive steers the
        agent back toward the upper part of the object.

        Returns:
            Direction vector in world frame.
        """
        direction = super().tangential_direction(ctx, state, percept)
        coverage = percept.get_feature_by_name("object_coverage")
        if coverage is not None and coverage < 0.8:
            dx, dy, dz = direction
            if dy < 0:
                dy = -dy  # flip to upward
            norm = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            if norm > 0:
                direction = (dx / norm, dy / norm, dz / norm)
            logger.debug(
                "tangential_direction: coverage=%.2f < 0.8, biased upward direction=%s",
                coverage,
                direction,
            )
        return direction

    def _shape_orient_angle(self, raw_angle: float) -> float:
        """Apply dead-band then clamp to effective max orient angle.

        The effective cap is normally ``_MAX_ORIENT_DEG`` but may be reduced
        by ``_orient_vertical`` when coverage is low (table-boundary
        proximity).

        Returns:
            The shaped rotation angle in degrees.
        """
        angle = float(raw_angle)
        if abs(angle) < _ORIENT_DEADBAND_DEG:
            return 0.0
        cap = self._effective_max_orient_deg
        return float(np.clip(angle, -cap, cap))

    def _log_orient_diagnostics(
        self,
        axis: str,
        *,
        raw_angle: float,
        rotation_degrees: float,
        lateral: float,
        forward: float,
        state: MotorSystemState,
        percept: Message,
    ) -> None:
        """Log raw normal, computed angle, depth, and compensating distances."""
        try:
            normal = np.asarray(percept.get_surface_normal(), dtype=float)
            normal_str = np.round(normal, 4).tolist()
        except Exception:  # noqa: BLE001
            normal_str = "unavailable"

        if self._enable_orient_decomposition_logging:
            decomposition = self._orient_decomposition(axis, state, percept)
            logger.info(
                "orient_%s_decomposition: branch=%s rotated_normal=%s x=%.6f y=%.6f "
                "z=%.6f raw_from_components=%.3f",
                axis,
                decomposition["branch"],
                decomposition["rotated_normal"],
                decomposition["x"],
                decomposition["y"],
                decomposition["z"],
                decomposition["raw_from_components"],
            )

        depth = self._filtered_forward_depth_from_stashed_obs()
        logger.info(
            "orient_%s: raw=%.3f° shaped=%.3f° lateral=%.4fm fwd=%.4fm "
            "depth=%s normal=%s translation_disabled=%s",
            axis,
            float(raw_angle),
            float(rotation_degrees),
            float(lateral),
            float(forward),
            f"{depth:.4f}m" if depth is not None else "None",
            normal_str,
            self._disable_orient_compensation_translation,
        )

    def _orient_decomposition(
        self,
        axis: str,
        state: MotorSystemState,
        percept: Message,
    ) -> dict[str, Any]:
        """Return intermediate values used by orient angle computation."""
        original_surface_normal = np.asarray(percept.get_surface_normal(), dtype=float)
        inverse_quaternion_rotation = self.get_inverse_agent_rot(state)
        rotated_surface_normal = np.asarray(
            qt.rotate_vectors(inverse_quaternion_rotation, original_surface_normal),
            dtype=float,
        )
        x, y, z = rotated_surface_normal

        if axis == "horizontal":
            if z != 0:
                raw_from_components = -np.degrees(np.arctan(x / z))
                branch = "atan_x_over_z"
            else:
                raw_from_components = -np.sign(x) * 90.0
                branch = "z_zero_sign_x"
        else:
            if z != 0:
                raw_from_components = -np.degrees(np.arctan(y / z))
                branch = "atan_y_over_z"
            else:
                raw_from_components = -np.sign(y) * 90.0
                branch = "z_zero_sign_y"

        return {
            "rotated_normal": np.round(rotated_surface_normal, 6).tolist(),
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "branch": branch,
            "raw_from_components": float(raw_from_components),
        }

    def _move_forward(self, percept: Message) -> MoveForward:  # noqa: ARG002
        """Move forward using semantic-filtered depth, capped for safety.

        Returns:
            MoveForward action.
        """
        filtered = self._filtered_forward_depth_from_stashed_obs()
        if filtered is not None:
            raw_distance = filtered - self.desired_object_distance
            distance = min(raw_distance, _MAX_FORWARD_STEP_M)
            logger.info(
                "move_forward_distance: filtered_depth=%.6fm desired=%.6fm "
                "raw=%.6fm capped=%.6fm cap=%.6fm",
                float(filtered),
                float(self.desired_object_distance),
                float(raw_distance),
                float(distance),
                float(_MAX_FORWARD_STEP_M),
            )
        else:
            distance = _FALLBACK_FORWARD_STEP_M
            logger.info(
                "move_forward_distance: filtered_depth=None fallback_step=%.6fm",
                float(distance),
            )
        return MoveForward(agent_id=self.agent_id, distance=distance)

    # ------------------------------------------------------------------
    # _touch_object override
    # ------------------------------------------------------------------

    def _touch_object(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        view_sensor_id: SensorID,
        state: MotorSystemState,
    ) -> MoveForward | OrientHorizontal | OrientVertical:
        """Touch-object override using semantically-filtered depth.

        Returns:
            A ``MoveForward`` if filtered depth is available, otherwise
            the parent's search action.
        """
        filtered_depth = self._filtered_forward_depth(observations, view_sensor_id)

        if filtered_depth is not None and filtered_depth < 1.0:
            sensor_state_z = state[self.agent_id].sensors[view_sensor_id].position[2]
            raw_distance = (
                filtered_depth - self.desired_object_distance - sensor_state_z
            )
            no_sensor_z_distance = filtered_depth - self.desired_object_distance
            distance = filtered_depth - self.desired_object_distance - sensor_state_z
            distance = min(distance, _MAX_FORWARD_STEP_M)
            logger.info(
                "touch_object_distance: filtered_depth=%.6fm desired=%.6fm "
                "sensor_state_z=%.6fm raw=%.6fm raw_no_sensor_z=%.6fm "
                "capped=%.6fm cap=%.6fm",
                float(filtered_depth),
                float(self.desired_object_distance),
                float(sensor_state_z),
                float(raw_distance),
                float(no_sensor_z_distance),
                float(distance),
                float(_MAX_FORWARD_STEP_M),
            )
            self.attempting_to_find_object = False
            return MoveForward(agent_id=self.agent_id, distance=distance)

        return self._constrained_search(ctx)

    def _constrained_search(
        self, ctx: RuntimeContext
    ) -> OrientHorizontal | OrientVertical:
        """Search for the object with distance-constrained arc movements.

        Replaces the parent's 0.48 m-radius random search with a tighter
        arc (``_MAX_SEARCH_RADIUS_M``) and smaller rotation steps so the
        sensor stays near the last known object position.

        When search begins (``touch_search_amount`` near zero) the sensor
        has typically just drifted into the table rejection zone.  The
        first few search steps move the sensor **upward** to escape the
        table boundary before starting the normal arc search.

        Returns:
            An orient action that arcs the sensor around a nearby point.
        """
        self.attempting_to_find_object = True

        # --- Escape-upward phase ---
        # The first few search steps tilt the sensor upward (negative
        # rotation = tilt up) and translate upward (negative
        # down_distance = move up) to leave the table zone.
        escape_steps = 3
        escape_deg = -10.0  # negative = tilt up
        escape_dist_m = -0.02  # negative = move up (opposite of down)

        if self.touch_search_amount < abs(escape_deg) * escape_steps:
            step_idx = (
                int(self.touch_search_amount / abs(escape_deg))
                if escape_deg != 0
                else 0
            )
            if step_idx < escape_steps:
                self.touch_search_amount += abs(escape_deg)
                logger.info(
                    "constrained_search: escape-upward step %d/%d",
                    step_idx + 1,
                    escape_steps,
                )
                return OrientVertical(
                    agent_id=self.agent_id,
                    rotation_degrees=escape_deg,
                    down_distance=escape_dist_m,
                    forward_distance=0.0,
                )

        # --- Normal arc search ---
        radius = min(
            self.desired_object_distance * 4,
            _MAX_SEARCH_RADIUS_M,
        )
        rotation_degrees = _SEARCH_ROTATION_DEG

        if self.touch_search_amount >= 720:
            orientation = "vertical" if ctx.rng.uniform() < 0.5 else "horizontal"
            rotation_degrees = ctx.rng.uniform(
                -_SEARCH_RANDOM_RANGE_DEG, _SEARCH_RANDOM_RANGE_DEG
            )
        elif self.touch_search_amount >= 360:
            orientation = "vertical"
        else:
            orientation = "horizontal"

        lateral = np.tan(np.radians(rotation_degrees)) * radius
        cos_r = np.cos(np.radians(rotation_degrees))
        forward = radius * (1.0 - cos_r) / cos_r

        self.touch_search_amount += rotation_degrees

        if orientation == "vertical":
            return OrientVertical(
                agent_id=self.agent_id,
                rotation_degrees=rotation_degrees,
                down_distance=lateral,
                forward_distance=forward,
            )
        return OrientHorizontal(
            agent_id=self.agent_id,
            rotation_degrees=rotation_degrees,
            left_distance=lateral,
            forward_distance=forward,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compensating_distances(
        self,
        rotation_degrees: float,
        percept: Message,  # noqa: ARG002
    ) -> tuple[float, float]:
        """Lateral and forward distances for an orient step.

        Uses filtered object-only depth instead of ``percept.mean_depth``
        which includes background pixels.  Falls back to
        ``desired_object_distance`` when filtered depth is unavailable
        (the percept ``mean_depth`` includes background and inflates
        distances 2-4x on real hardware).

        For **vertical** orient steps, positive ``lateral`` means
        "move down" (toward the table).  When the resulting down-distance
        would push the sensor toward the table, it is suppressed to
        prevent cumulative downward drift.

        Returns:
            (lateral_distance, forward_distance) in metres.
        """
        depth = self._filtered_forward_depth_from_stashed_obs()
        if depth is None:
            depth = self.desired_object_distance
        if self._disable_orient_compensation_translation:
            return 0.0, 0.0
        rotation_radians = np.radians(rotation_degrees)
        cos_r = np.cos(rotation_radians)
        lateral = np.tan(rotation_radians) * depth
        forward = depth * (1.0 - cos_r) / cos_r

        # For vertical orient: positive lateral = down_distance toward table.
        # Suppress to prevent cumulative downward drift.
        if self._orient_axis == "vertical" and lateral > 0:
            logger.debug(
                "compensating_distances: suppressed downward lateral "
                "%.4fm for vertical orient",
                lateral,
            )
            lateral = 0.0

        return lateral, forward

    def _filtered_forward_depth_from_stashed_obs(self) -> float | None:
        """Convenience wrapper using stashed observations.

        Returns:
            Filtered forward depth or ``None``.
        """
        if self._last_observations is None:
            return None
        return self._filtered_forward_depth(
            self._last_observations, self._patch_sensor_id
        )

    def _filtered_forward_depth(
        self,
        observations: Observations,
        view_sensor_id: SensorID,
    ) -> float | None:
        """Median forward distance (m) over in-bounds sensor-frame points.

        Returns:
            Median positive forward depth, or ``None`` when unavailable.
        """
        sensor_obs = observations[self.agent_id][view_sensor_id]
        sensor_xyz4 = sensor_obs.get("sensor_frame_data")
        if sensor_xyz4 is None:
            return None

        arr = np.asarray(sensor_xyz4)
        if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] == 0:
            return None

        in_bounds = arr[:, 3] > 0
        if not np.any(in_bounds):
            return None

        forward_depths = -arr[in_bounds, 2]
        finite = np.isfinite(forward_depths) & (forward_depths > 0)
        if not np.any(finite):
            return None

        return float(np.median(forward_depths[finite]))

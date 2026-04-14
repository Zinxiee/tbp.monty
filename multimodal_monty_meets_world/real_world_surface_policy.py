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

from tbp.monty.cmp import Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.actions import (
    MoveForward,
    OrientHorizontal,
    OrientVertical,
)
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_policies import SurfacePolicy
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState
from tbp.monty.frameworks.sensors import SensorID

logger = logging.getLogger(__name__)

# Maximum orient rotation per step (degrees). Keeps compensating translation
# distances sane: tan(15) * 0.22m = 6cm vs tan(35) * 0.22m = 15cm.
_MAX_ORIENT_DEG = 15.0

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
        min_object_coverage: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            min_object_coverage=min_object_coverage,
            **kwargs,
        )
        self._patch_sensor_id = SensorID(patch_sensor_id)
        self._last_observations: Observations | None = None

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
        """Stash observations for ``_move_forward``, then delegate.

        Returns:
            MotorPolicyResult from the parent ``__call__``.
        """
        self._last_observations = observations
        return super().__call__(ctx, observations, state, percept, goal)

    def _orient_horizontal(
        self, state: MotorSystemState, percept: Message
    ) -> OrientHorizontal:
        """Orient horizontally with clamped angle and object-only depth.

        Uses filtered median depth (object pixels only) instead of the
        percept's ``mean_depth`` which includes background pixels and
        inflates compensating distances 2-4x on real hardware.

        Returns:
            OrientHorizontal action.
        """
        rotation_degrees = self.orienting_angle_from_normal(
            orienting="horizontal",
            state=state,
            percept=percept,
        )
        rotation_degrees = float(
            np.clip(rotation_degrees, -_MAX_ORIENT_DEG, _MAX_ORIENT_DEG)
        )
        left_distance, forward_distance = self._compensating_distances(
            rotation_degrees, percept
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
        """Orient vertically with clamped angle and object-only depth.

        Returns:
            OrientVertical action.
        """
        rotation_degrees = self.orienting_angle_from_normal(
            orienting="vertical",
            state=state,
            percept=percept,
        )
        rotation_degrees = float(
            np.clip(rotation_degrees, -_MAX_ORIENT_DEG, _MAX_ORIENT_DEG)
        )
        down_distance, forward_distance = self._compensating_distances(
            rotation_degrees, percept
        )
        return OrientVertical(
            agent_id=self.agent_id,
            rotation_degrees=rotation_degrees,
            down_distance=down_distance,
            forward_distance=forward_distance,
        )

    def _move_forward(self, percept: Message) -> MoveForward:  # noqa: ARG002
        """Move forward using semantic-filtered depth, capped for safety.

        Returns:
            MoveForward action.
        """
        filtered = self._filtered_forward_depth_from_stashed_obs()
        if filtered is not None:
            distance = min(
                filtered - self.desired_object_distance,
                _MAX_FORWARD_STEP_M,
            )
        else:
            distance = _FALLBACK_FORWARD_STEP_M
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
        filtered_depth = self._filtered_forward_depth(
            observations, view_sensor_id
        )

        if filtered_depth is not None and filtered_depth < 1.0:
            distance = (
                filtered_depth
                - self.desired_object_distance
                - state[self.agent_id]
                .sensors[view_sensor_id]
                .position[2]
            )
            distance = min(distance, _MAX_FORWARD_STEP_M)
            self.attempting_to_find_object = False
            return MoveForward(
                agent_id=self.agent_id, distance=distance
            )

        return self._constrained_search(ctx)

    def _constrained_search(
        self, ctx: RuntimeContext
    ) -> OrientHorizontal | OrientVertical:
        """Search for the object with distance-constrained arc movements.

        Replaces the parent's 0.48 m-radius random search with a tighter
        arc (``_MAX_SEARCH_RADIUS_M``) and smaller rotation steps so the
        sensor stays near the last known object position.

        Returns:
            An orient action that arcs the sensor around a nearby point.
        """
        self.attempting_to_find_object = True

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
        self, rotation_degrees: float, percept: Message
    ) -> tuple[float, float]:
        """Lateral and forward distances for an orient step.

        Uses filtered object-only depth instead of ``percept.mean_depth``
        which includes background pixels.  Falls back to the percept value
        when filtered depth is unavailable.

        Returns:
            (lateral_distance, forward_distance) in metres.
        """
        depth = self._filtered_forward_depth_from_stashed_obs()
        if depth is None:
            depth = percept.get_feature_by_name("mean_depth")
        rotation_radians = np.radians(rotation_degrees)
        cos_r = np.cos(rotation_radians)
        lateral = np.tan(rotation_radians) * depth
        forward = depth * (1.0 - cos_r) / cos_r
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


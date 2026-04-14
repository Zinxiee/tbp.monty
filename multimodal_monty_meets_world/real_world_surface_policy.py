"""Real-world surface policy for the Lite6 + Maixsense A010 setup.

Provides ``RealWorldSurfacePolicy``, a subclass of ``SurfacePolicy`` that
adapts the simulated surface-agent loop to the constraints of the physical
Lite6 + Maixsense A010 rig.

Behavioral overrides layered on top of the parent policy:

1. ``_touch_sensor_id`` returns ``"patch"`` instead of ``"view_finder"``.

2. ``_touch_object`` reads forward distance from the semantically-filtered
   point cloud instead of the raw center-pixel depth.

3. ``_orient_horizontal`` / ``_orient_vertical`` clamp the computed rotation
   angle to ``_MAX_ORIENT_DEG`` **before** computing compensating distances.
   The parent computes distances from the raw angle and lets the env clip the
   rotation separately, creating a magnitude/direction mismatch that causes
   oversized translation steps in the real world.

4. ``_move_forward`` uses the same semantic-filtered depth as ``_touch_object``
   instead of the percept's ``min_depth`` feature, which can be noisy on a
   real ToF sensor.

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
        """Orient horizontally with clamped rotation angle.

        Clamps the rotation to ±_MAX_ORIENT_DEG *before* computing the
        compensating lateral/forward distances so the distances match
        the actual rotation that will be applied.

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
        left_distance, forward_distance = self.horizontal_distances(
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
        """Orient vertically with clamped rotation angle.

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
        down_distance, forward_distance = self.vertical_distances(
            rotation_degrees, percept
        )
        return OrientVertical(
            agent_id=self.agent_id,
            rotation_degrees=rotation_degrees,
            down_distance=down_distance,
            forward_distance=forward_distance,
        )

    def _move_forward(self, percept: Message) -> MoveForward:
        """Move forward using semantic-filtered depth when available.

        Falls back to the parent's ``min_depth`` approach if no filtered
        observations are available.

        Returns:
            MoveForward action.
        """
        filtered = self._filtered_forward_depth_from_stashed_obs()
        if filtered is not None:
            distance = filtered - self.desired_object_distance
        else:
            distance = (
                percept.get_feature_by_name("min_depth")
                - self.desired_object_distance
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
            self.attempting_to_find_object = False
            return MoveForward(
                agent_id=self.agent_id, distance=distance
            )

        spoofed = self._observations_with_spoofed_center_depth(
            observations, view_sensor_id, value=10.0
        )
        return super()._touch_object(
            ctx, spoofed, view_sensor_id, state
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    def _observations_with_spoofed_center_depth(
        self,
        observations: Observations,
        view_sensor_id: SensorID,
        value: float,
    ) -> Observations:
        """Return a copy with the center depth pixel set to ``value``."""
        agent_obs = dict(observations[self.agent_id])
        sensor_obs = dict(agent_obs[view_sensor_id])
        depth = np.array(sensor_obs["depth"], copy=True)
        h, w = depth.shape[0], depth.shape[1]
        depth[h // 2, w // 2] = value
        sensor_obs["depth"] = depth
        agent_obs[view_sensor_id] = sensor_obs
        spoofed = dict(observations)
        spoofed[self.agent_id] = agent_obs
        return spoofed

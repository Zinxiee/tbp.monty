"""Real-world surface policy for the Lite6 + Maixsense A010 setup.

Provides ``RealWorldSurfacePolicy``, a thin subclass of ``SurfacePolicy`` that
adapts the simulated surface-agent loop to the constraints of the physical
Lite6 + Maixsense A010 rig.

Two behavioral overrides are layered on top of the parent policy:

1. ``_touch_sensor_id`` returns the ``"patch"`` sensor instead of the simulator
   ``"view_finder"``.  The real-world environment exposes only one sensor.

2. ``_touch_object`` reads the forward distance to the object from the
   semantically-filtered point cloud (``sensor_frame_data`` produced by the
   Maixsense observation adapter) instead of the raw center-pixel depth used
   by the parent.  Reading raw center-pixel depth is unsafe on a real ToF
   sensor: specular returns and partial off-object framing routinely produce
   depths that look in-range but actually correspond to the table or far
   background, causing the surface agent to drive itself into the work
   surface.  The semantic mask has already filtered those out, so the median
   in-bounds depth is the right thing to act on.

The default ``min_object_coverage`` is also lowered (parent default 0.1 →
0.05) to keep a single noisy frame from flipping the policy into the
``attempting_to_find_object`` recovery state, which is what triggers the
``_touch_object`` loop in the first place.
"""

from __future__ import annotations

from typing import Any

import numpy as np

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


class RealWorldSurfacePolicy(SurfacePolicy):
    """SurfacePolicy variant for real-world setups that have no view-finder sensor.

    Args:
        patch_sensor_id: ID of the sensor to use for object-finding in
            ``_touch_object``.  Defaults to ``"patch"``, matching the sensor ID
            used in the Lite6 + Maixsense A010 experiment configs.
        min_object_coverage: Coverage threshold below which the policy enters
            the touch-object recovery loop.  Defaults to ``0.05`` (parent
            default is ``0.1``); the lower value reduces false-positive
            transitions caused by single noisy ToF frames.
        **kwargs: Forwarded to ``SurfacePolicy.__init__``.
    """

    def __init__(
        self,
        *args: Any,
        patch_sensor_id: str = "patch",
        min_object_coverage: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, min_object_coverage=min_object_coverage, **kwargs)
        self._patch_sensor_id = SensorID(patch_sensor_id)

    def _touch_sensor_id(self) -> SensorID:
        """Use the real sensor patch instead of the non-existent view-finder.

        Returns:
            The configured patch sensor ID.
        """
        return self._patch_sensor_id

    def _touch_object(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        view_sensor_id: SensorID,
        state: MotorSystemState,
    ) -> MoveForward | OrientHorizontal | OrientVertical:
        """Touch-object override that uses semantically-filtered depth.

        Replaces the parent's raw center-pixel depth read with the median
        forward distance over the in-bounds (semantic-mask > 0) points from
        the Maixsense adapter's ``sensor_frame_data``.  When at least one
        in-bounds point exists, returns a ``MoveForward`` toward that depth.
        Otherwise, falls back to the parent search behavior so the agent
        rotates to look for the object instead of acting on a stale or
        unreliable raw center reading.

        Args:
            ctx: Runtime context.
            observations: Environment observations.
            view_sensor_id: Sensor ID to query (the patch sensor).
            state: Current motor system state.

        Returns:
            A ``MoveForward`` if a filtered depth is available, otherwise the
            search action chosen by the parent ``_touch_object``.
        """
        filtered_depth = self._filtered_forward_depth(observations, view_sensor_id)

        if filtered_depth is not None and filtered_depth < 1.0:
            distance = (
                filtered_depth
                - self.desired_object_distance
                - state[self.agent_id].sensors[view_sensor_id].position[2]
            )
            self.attempting_to_find_object = False
            return MoveForward(agent_id=self.agent_id, distance=distance)

        # No usable in-bounds points: hand off to the parent's search loop.
        # Spoof the center-pixel depth so the parent's `< 1.0` check fails and
        # it falls into the rotational search branch instead of acting on the
        # raw value.
        spoofed = self._observations_with_spoofed_center_depth(
            observations, view_sensor_id, value=10.0
        )
        return super()._touch_object(ctx, spoofed, view_sensor_id, state)

    def _filtered_forward_depth(
        self,
        observations: Observations,
        view_sensor_id: SensorID,
    ) -> float | None:
        """Median forward distance (m) over in-bounds sensor-frame points.

        The Maixsense adapter publishes ``sensor_frame_data`` as an Nx4 array
        of (x, y, z, semantic_flag) where ``z = -depth`` (Monty
        right-up-backward sensor convention).

        Returns:
            The median positive forward depth across in-bounds points, or
            ``None`` when the observation lacks ``sensor_frame_data`` or no
            points pass the mask.
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
        """Return a copy of ``observations`` with the center depth set to ``value``.

        Used to delegate to the parent ``_touch_object`` search branch without
        mutating the original observation dict.  Only the depth array of the
        targeted sensor is copied; other entries are shared by reference.
        """
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

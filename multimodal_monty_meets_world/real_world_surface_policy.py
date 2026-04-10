"""Real-world surface policy for the Lite6 + Maixsense A010 setup.

Provides ``RealWorldSurfacePolicy``, a thin subclass of ``SurfacePolicy`` that
replaces the hardcoded ``"view_finder"`` sensor used by ``_touch_object`` with the
``"patch"`` sensor that is actually present in the real-world environment.

In simulation, Monty's surface agent has a dedicated view-finder sensor that is used
to locate the object before the surface-following loop begins.  The real-world Lite6
environment exposes only a single ``patch`` sensor (the Maixsense A010).  This
subclass overrides the sensor-ID hook so that ``_touch_object`` reads depth from that
same sensor instead of crashing with a missing ``"view_finder"`` key.
"""

from __future__ import annotations

from tbp.monty.frameworks.models.motor_policies import SurfacePolicy
from tbp.monty.frameworks.sensors import SensorID


class RealWorldSurfacePolicy(SurfacePolicy):
    """SurfacePolicy variant for real-world setups that have no view-finder sensor.

    Args:
        patch_sensor_id: ID of the sensor to use for object-finding in
            ``_touch_object``.  Defaults to ``"patch"``, matching the sensor ID
            used in the Lite6 + Maixsense A010 experiment configs.
        **kwargs: Forwarded to ``SurfacePolicy.__init__``.
    """

    def __init__(self, *args, patch_sensor_id: str = "patch", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._patch_sensor_id = SensorID(patch_sensor_id)

    def _touch_sensor_id(self) -> SensorID:
        """Use the real sensor patch instead of the non-existent view-finder."""
        return self._patch_sensor_id

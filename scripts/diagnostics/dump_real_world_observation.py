#!/usr/bin/env python
# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT license that can be found in the
# LICENSE file or at https://opensource.org/licenses/MIT.
"""Capture one observation at the home pose and dump every pipeline stage.

Constructs the same hardware-backed environment used by the
``real_world/lite6_maixsense_unsupervised`` experiment, resets to the home pose,
captures a single observation with ``MONTY_ADAPTER_DEBUG_DUMP=1`` set, then
exits cleanly.

Usage:
    python scripts/diagnostics/dump_real_world_observation.py

Output:
    A ``.npz`` per call in ``~/monty_diag/`` and a single summary log line per
    call (see ``MaixsenseMontyObservationAdapter._dump_pipeline_stage``).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from hydra.utils import instantiate
from omegaconf import OmegaConf

from tbp.monty.frameworks.environments.real_world_environment import (
    RealWorldLite6A010Environment,
)

ENV_YAML_PATH = (
    REPO_ROOT
    / "src"
    / "tbp"
    / "monty"
    / "conf"
    / "environment"
    / "real_world_lite6_maixsense.yaml"
)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("dump_real_world_observation")

    if not ENV_YAML_PATH.exists():
        log.error("Environment YAML not found at %s", ENV_YAML_PATH)
        return 1

    cfg = OmegaConf.load(ENV_YAML_PATH)
    env_args_cfg = cfg.env_init_args

    log.info("Instantiating robot_interface, sensor_client, observation_adapter")
    robot = instantiate(env_args_cfg.robot_interface)
    sensor = instantiate(env_args_cfg.sensor_client)
    adapter = instantiate(env_args_cfg.observation_adapter)

    env_kwargs = OmegaConf.to_container(env_args_cfg, resolve=True)
    env_kwargs["robot_interface"] = robot
    env_kwargs["sensor_client"] = sensor
    env_kwargs["observation_adapter"] = adapter
    env_kwargs["require_object_swap_confirmation"] = False

    env = RealWorldLite6A010Environment(**env_kwargs)

    os.environ["MONTY_ADAPTER_DEBUG_DUMP"] = "1"
    log.info("MONTY_ADAPTER_DEBUG_DUMP=1; capturing one frame at home pose")

    try:
        env.reset()
        log.info("Frame captured. See ~/monty_diag/ for the .npz dump.")
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
